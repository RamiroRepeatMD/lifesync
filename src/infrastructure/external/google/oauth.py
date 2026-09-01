"""Adaptador OAuth2 contra Google (PB-009).

Implementa `AutorizadorGoogle` hablando con los dos endpoints públicos de
Google: el de consentimiento, al que va la persona con el navegador, y el de
tokens, al que vamos nosotros por detrás.

Regla dura del módulo, igual que en el repositorio de tokens: **acá no se
loguea ninguna credencial**, ni el código, ni el access_token, ni el refresh.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import httpx
import structlog

from src.application.ports.autorizador_google import AutorizadorGoogle, CredencialesGoogle
from src.domain.exceptions import AutorizacionFallidaError, ServiceUnavailableError
from src.infrastructure.config.settings import Settings
from src.infrastructure.external.google import estado as estado_oauth

logger = structlog.get_logger(__name__)

URL_CONSENTIMIENTO = "https://accounts.google.com/o/oauth2/v2/auth"
URL_TOKENS = "https://oauth2.googleapis.com/token"

# Lectura + escritura de eventos (PB-015 · PB-016). El readonly se conserva
# aunque parezca redundante: `calendarList` —la lista de calendarios que lee
# PB-015— lo necesita; `calendar.events` sólo cubre los eventos.
#
# OJO: una cuenta conectada antes de PB-016 tiene guardado sólo el readonly.
# La lectura le sigue andando; la primera escritura devuelve 403 y el bot le
# pide reconectar con /conectar (ver PermisoInsuficienteError).
SCOPES = (
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
)

TIMEOUT_SEGUNDOS = 10.0
TIMEOUT_CONEXION_SEGUNDOS = 5.0

# Margen que se le resta al vencimiento que informa Google. Sin esto, un token
# que vence en dos segundos se da por vigente y la llamada siguiente falla con
# 401 en pleno pedido del usuario.
MARGEN_DE_VENCIMIENTO_SEGUNDOS = 60


class OAuthGoogle(AutorizadorGoogle):
    """Implementación de `AutorizadorGoogle` contra los endpoints de Google."""

    def __init__(self, cliente: httpx.AsyncClient, settings: Settings, clave_estado: bytes) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._cliente = cliente
        self._settings = settings
        self._clave_estado = clave_estado

    # --- Ida: mandar a la persona a Google -------------------------------

    def url_de_autorizacion(self, usuario_id: UUID, ahora: datetime) -> str:
        """Arma la URL de consentimiento con el `state` ya firmado."""
        parametros = {
            "client_id": self._settings.google_client_id or "",
            "redirect_uri": self._settings.google_redirect_uri or "",
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "state": estado_oauth.firmar(usuario_id, self._clave_estado, ahora),
            # Los dos parámetros que deciden si vamos a poder renovar:
            #
            # access_type=offline es lo único que hace que Google emita un
            # refresh_token. Sin él la conexión se muere en una hora y hay que
            # volver a molestar a la persona.
            #
            # prompt=consent lo fuerza en CADA autorización. Google lo emite
            # sólo la primera vez salvo que se lo pidas así, de modo que sin
            # esto una reconexión nos dejaría sin refresh token y sin aviso.
            "access_type": "offline",
            "prompt": "consent",
            # Si la persona destilda un permiso, que falle acá y no más tarde
            # con un 403 en medio de una consulta.
            "include_granted_scopes": "false",
        }
        return f"{URL_CONSENTIMIENTO}?{urlencode(parametros)}"

    def usuario_del_estado(self, estado: str, ahora: datetime) -> UUID:
        """Verifica el `state` de la vuelta y extrae el usuario."""
        return estado_oauth.verificar(estado, self._clave_estado, ahora)

    # --- Vuelta: canjear y renovar ---------------------------------------

    async def canjear_codigo(self, codigo: str) -> CredencialesGoogle:
        """Cambia el código de un solo uso por credenciales."""
        return await self._pedir_tokens(
            {
                "code": codigo,
                "grant_type": "authorization_code",
                "redirect_uri": self._settings.google_redirect_uri or "",
            },
            operacion="canje",
        )

    async def refrescar(self, refresh_token: str) -> CredencialesGoogle:
        """Pide un access_token nuevo con el refresh_token."""
        return await self._pedir_tokens(
            {"refresh_token": refresh_token, "grant_type": "refresh_token"},
            operacion="refresco",
        )

    async def _pedir_tokens(self, cuerpo: dict[str, str], *, operacion: str) -> CredencialesGoogle:
        """Postea al endpoint de tokens y traduce lo que vuelva."""
        completo = {
            **cuerpo,
            "client_id": self._settings.google_client_id or "",
            "client_secret": self._secreto(),
        }

        try:
            respuesta = await self._cliente.post(URL_TOKENS, data=completo)
        except httpx.HTTPError as exc:
            logger.error(
                "google.oauth.error_transporte",
                operacion=operacion,
                tipo=type(exc).__name__,
            )
            raise AutorizacionFallidaError("No se pudo contactar a Google.") from None

        if respuesta.status_code >= httpx.codes.BAD_REQUEST:
            # Google explica el rechazo en `error` y `error_description`, y son
            # diagnósticos de la API, no datos de la persona: se loguean. El
            # cuerpo completo NO, porque en el canje incluye el código.
            datos = _json_o_vacio(respuesta)
            logger.error(
                "google.oauth.rechazado",
                operacion=operacion,
                status_code=respuesta.status_code,
                error=datos.get("error"),
                descripcion=datos.get("error_description"),
            )
            raise AutorizacionFallidaError(f"Google rechazó el {operacion}.")

        return self._a_credenciales(_json_o_vacio(respuesta), operacion=operacion)

    def _secreto(self) -> str:
        secreto = self._settings.google_client_secret
        return "" if secreto is None else secreto.get_secret_value()

    @staticmethod
    def _a_credenciales(datos: dict[str, Any], *, operacion: str) -> CredencialesGoogle:
        """Traduce la respuesta de Google a la credencial del dominio."""
        access_token = datos.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            logger.error("google.oauth.sin_access_token", operacion=operacion)
            raise AutorizacionFallidaError("Google no devolvió credenciales utilizables.")

        refresh_token = datos.get("refresh_token")
        scope = datos.get("scope")

        logger.info(
            "google.oauth.credenciales_obtenidas",
            operacion=operacion,
            # Si esto sale en False después de un canje, faltó access_type=offline.
            trajo_refresh=isinstance(refresh_token, str) and bool(refresh_token),
            cantidad_scopes=len(scope.split()) if isinstance(scope, str) else 0,
        )

        return CredencialesGoogle(
            access_token=access_token,
            refresh_token=refresh_token if isinstance(refresh_token, str) else None,
            expira_en=_vencimiento(datos.get("expires_in")),
            scopes=tuple(scope.split()) if isinstance(scope, str) else (),
        )


def _vencimiento(expires_in: Any) -> datetime | None:
    """Convierte los segundos de vida que informa Google en un instante."""
    if not isinstance(expires_in, int):
        return None
    segundos = max(expires_in - MARGEN_DE_VENCIMIENTO_SEGUNDOS, 0)
    return datetime.now(UTC) + timedelta(seconds=segundos)


def _json_o_vacio(respuesta: httpx.Response) -> dict[str, Any]:
    """Devuelve el cuerpo como dict, o vacío si no es JSON con esa forma."""
    try:
        datos = respuesta.json()
    except ValueError:
        return {}
    return datos if isinstance(datos, dict) else {}


def create_google_oauth_client() -> httpx.AsyncClient:
    """Abre el cliente HTTP hacia Google, con timeouts explícitos.

    Sin auth por defecto: las credenciales van en el cuerpo de cada POST, no en
    una cabecera compartida.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT_SEGUNDOS, connect=TIMEOUT_CONEXION_SEGUNDOS)
    )


def create_autorizador_google(cliente: httpx.AsyncClient, settings: Settings) -> OAuthGoogle:
    """Arma el autorizador, derivando la clave de firmado del `state`.

    Raises:
        ServiceUnavailableError: Si falta configuración de OAuth o la clave de
            cifrado de la que se deriva la del `state`.
    """
    clave_maestra = settings.token_encryption_key
    if not settings.google_oauth_configurado or clave_maestra is None:
        raise ServiceUnavailableError(
            "Faltan GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI "
            "o TOKEN_ENCRYPTION_KEY para conectar cuentas de Google."
        )

    return OAuthGoogle(
        cliente,
        settings,
        estado_oauth.derivar_clave(clave_maestra.get_secret_value()),
    )


async def close_google_oauth_client(cliente: httpx.AsyncClient | None) -> None:
    """Cierra el cliente HTTP. Tolera None y errores de cierre."""
    if cliente is None:
        return
    try:
        await cliente.aclose()
    except Exception as exc:  # fallar al apagar no rompe el shutdown
        logger.warning("google.oauth.cierre_con_error", tipo=type(exc).__name__)
