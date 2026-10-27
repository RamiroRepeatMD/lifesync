"""Adaptador OAuth2 contra Notion (PB-014).

Implementa `AutorizadorNotion` con los endpoints públicos de Notion: el de
consentimiento, al que va la persona con el navegador, y los de tokens y
revocación, a los que vamos nosotros por detrás. Verificado contra la
referencia oficial el 27/10/2026.

Lo que lo separa del de Google:

- **La app se autentica con HTTP Basic** (`client_id:client_secret`) y el
  cuerpo va en JSON, no como formulario.
- **Todo pedido lleva `Notion-Version`**, también los de OAuth.
- **El `state` se firma igual que el de Google** (`google/estado.py` es nuestro,
  no de Google), pero con una etiqueta de derivación propia: uno firmado para
  Google no vale en este callback, ni al revés.

Regla dura, como en Google: **acá no se loguea ninguna credencial** —ni el
código, ni los tokens, ni el secret—, y tampoco el `owner` de la respuesta, que
puede traer el correo de la persona.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import httpx
import structlog

from src.application.ports.autorizador_notion import AutorizadorNotion, CredencialesNotion
from src.domain.exceptions import AutorizacionFallidaError, ServiceUnavailableError
from src.infrastructure.config.settings import Settings
from src.infrastructure.external.google import estado as estado_oauth
from src.infrastructure.external.notion.transporte import ENCABEZADOS, codigo_de, json_o_vacio

logger = structlog.get_logger(__name__)

API = "https://api.notion.com/v1"
URL_CONSENTIMIENTO = f"{API}/oauth/authorize"
URL_TOKENS = f"{API}/oauth/token"
URL_REVOCACION = f"{API}/oauth/revoke"
# La de Google es `lifesync.oauth.state.v1`: con otra etiqueta, otra clave.
ETIQUETA_DEL_STATE = b"lifesync.oauth.state.notion.v1"

TIMEOUT_SEGUNDOS = 10.0
TIMEOUT_CONEXION_SEGUNDOS = 5.0


class OAuthNotion(AutorizadorNotion):
    """Implementación de `AutorizadorNotion` contra los endpoints de Notion."""

    def __init__(self, cliente: httpx.AsyncClient, settings: Settings, clave_estado: bytes) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._cliente = cliente
        self._settings = settings
        self._clave_estado = clave_estado

    # --- Ida: mandar a la persona a Notion -------------------------------

    def url_de_autorizacion(self, usuario_id: UUID, ahora: datetime) -> str:
        """Arma la URL de consentimiento con el `state` ya firmado."""
        parametros = {
            "client_id": self._settings.notion_client_id or "",
            "redirect_uri": self._settings.notion_redirect_uri or "",
            "response_type": "code",
            # La conexión es de la persona, no de un workspace entero ("Always
            # use user", dice la referencia).
            "owner": "user",
            "state": estado_oauth.firmar(usuario_id, self._clave_estado, ahora),
        }
        return f"{URL_CONSENTIMIENTO}?{urlencode(parametros)}"

    def usuario_del_estado(self, estado: str, ahora: datetime) -> UUID:
        """Verifica el `state` de la vuelta y extrae el usuario."""
        return estado_oauth.verificar(estado, self._clave_estado, ahora)

    # --- Vuelta: canjear, renovar y revocar ------------------------------

    async def canjear_codigo(self, codigo: str) -> CredencialesNotion:
        """Cambia el código de un solo uso por credenciales."""
        return await self._pedir_tokens(
            {
                "grant_type": "authorization_code",
                "code": codigo,
                "redirect_uri": self._settings.notion_redirect_uri or "",
            },
            operacion="canje",
        )

    async def refrescar(self, refresh_token: str) -> CredencialesNotion:
        """Pide credenciales nuevas con el refresh_token."""
        return await self._pedir_tokens(
            {"grant_type": "refresh_token", "refresh_token": refresh_token},
            operacion="refresco",
        )

    async def revocar(self, token: str) -> bool:
        """Invalida una credencial en Notion. Best-effort: nunca lanza.

        Un 400 `invalid_grant` cuenta como éxito: la credencial ya no valía, y
        el estado final es el mismo que el de revocarla.
        """
        try:
            respuesta = await self._cliente.post(
                URL_REVOCACION,
                json={"token": token},
                auth=self._credenciales_de_la_app(),
                headers=ENCABEZADOS,
            )
        except httpx.HTTPError as exc:
            logger.warning("notion.oauth.revocacion_fallida", tipo=type(exc).__name__)
            return False

        codigo = codigo_de(respuesta)
        ya_no_valia = respuesta.status_code == httpx.codes.BAD_REQUEST and codigo == "invalid_grant"
        exito = respuesta.status_code == httpx.codes.OK or ya_no_valia
        logger.info(
            "notion.oauth.revocado" if exito else "notion.oauth.revocacion_rechazada",
            status_code=respuesta.status_code,
            codigo=codigo,
        )
        return exito

    async def _pedir_tokens(self, cuerpo: dict[str, str], *, operacion: str) -> CredencialesNotion:
        """Postea al endpoint de tokens y traduce lo que vuelva."""
        try:
            respuesta = await self._cliente.post(
                URL_TOKENS, json=cuerpo, auth=self._credenciales_de_la_app(), headers=ENCABEZADOS
            )
        except httpx.HTTPError as exc:
            logger.error(
                "notion.oauth.error_transporte", operacion=operacion, tipo=type(exc).__name__
            )
            raise AutorizacionFallidaError("No se pudo contactar a Notion.") from None

        if respuesta.status_code >= httpx.codes.BAD_REQUEST:
            # El código del rechazo es un diagnóstico de la API y se loguea; el
            # cuerpo entero no, porque en el canje repite el código de un solo uso.
            logger.error(
                "notion.oauth.rechazado",
                operacion=operacion,
                status_code=respuesta.status_code,
                codigo=codigo_de(respuesta),
            )
            raise AutorizacionFallidaError(f"Notion rechazó el {operacion}.")

        return _a_credenciales(json_o_vacio(respuesta), operacion=operacion)

    def _credenciales_de_la_app(self) -> httpx.BasicAuth:
        secreto = self._settings.notion_client_secret
        return httpx.BasicAuth(
            self._settings.notion_client_id or "",
            "" if secreto is None else secreto.get_secret_value(),
        )


def _a_credenciales(datos: dict[str, Any], *, operacion: str) -> CredencialesNotion:
    """Traduce la respuesta de Notion a la credencial del dominio.

    Del resto de la respuesta (`workspace_name`, `bot_id`, `owner`…) no se usa
    nada: el `owner` puede traer el correo de la persona.
    """
    access_token = datos.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        logger.error("notion.oauth.sin_access_token", operacion=operacion)
        raise AutorizacionFallidaError("Notion no devolvió credenciales utilizables.")

    refresh_token = datos.get("refresh_token")
    trajo_refresh = isinstance(refresh_token, str) and bool(refresh_token)
    logger.info(
        "notion.oauth.credenciales_obtenidas", operacion=operacion, trajo_refresh=trajo_refresh
    )
    return CredencialesNotion(
        access_token=access_token,
        refresh_token=refresh_token if trajo_refresh else None,
    )


def create_notion_client() -> httpx.AsyncClient:
    """Abre el cliente HTTP hacia Notion, con timeouts explícitos.

    Uno solo para el OAuth y para la API: las credenciales van en cada pedido
    (Basic para la app, Bearer para la persona), nunca como cabecera compartida.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT_SEGUNDOS, connect=TIMEOUT_CONEXION_SEGUNDOS)
    )


def create_autorizador_notion(cliente: httpx.AsyncClient, settings: Settings) -> OAuthNotion:
    """Arma el autorizador, derivando la clave de firmado del `state`.

    Raises:
        ServiceUnavailableError: Si falta la configuración de OAuth de Notion o
            la clave de cifrado de la que se deriva la del `state`.
    """
    clave_maestra = settings.token_encryption_key
    if not settings.notion_oauth_configurado or clave_maestra is None:
        raise ServiceUnavailableError(
            "Faltan NOTION_CLIENT_ID, NOTION_CLIENT_SECRET, NOTION_REDIRECT_URI "
            "o TOKEN_ENCRYPTION_KEY para conectar Notion."
        )
    return OAuthNotion(
        cliente,
        settings,
        estado_oauth.derivar_clave(clave_maestra.get_secret_value(), ETIQUETA_DEL_STATE),
    )


async def close_notion_client(cliente: httpx.AsyncClient | None) -> None:
    """Cierra el cliente HTTP. Tolera None y errores de cierre."""
    if cliente is None:
        return
    try:
        await cliente.aclose()
    except Exception as exc:  # fallar al apagar no rompe el shutdown
        logger.warning("notion.cierre_con_error", tipo=type(exc).__name__)
