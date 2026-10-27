"""Caso de uso: conectar el Notion de una persona (PB-014, RF-01).

El mismo cruce que `ConectarGoogle` —la persona está en WhatsApp y el
consentimiento pasa por el navegador—, con dos diferencias que dicta Notion:

- **No informa cuándo vence el access_token.** No hay refresco perezoso por
  fecha: se renueva cuando la API contesta 401 (`renovar`).
- **Puede rotar el par en cada renovación.** Dos renovaciones a la vez con el
  mismo refresh_token se invalidarían entre sí, y una que termina después de un
  `/desconectar notion` reviviría la conexión borrada. Por eso `completar`,
  `desconectar` y `renovar` toman el mismo candado por persona.

No se generalizó `ConectarGoogle` porque la semántica del token es otra y lo
usan tal cual cuatro adaptadores de Google: unificarlos queda como deuda.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import UUID
from weakref import WeakValueDictionary

import structlog

from src.application.ports.autorizador_notion import AutorizadorNotion, CredencialesNotion
from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import (
    AutorizacionFallidaError,
    NotionNoConectadoError,
    RepositoryError,
)
from src.domain.repositories.oauth_token_repository import OAuthTokenRepository
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth

logger = structlog.get_logger(__name__)

PROVEEDOR = ProveedorOAuth.NOTION

# El candado vive en el módulo y no en la instancia: los casos de uso se arman
# por pedido (`dependencies.py`), y los comandos no pasan por el candado del
# agente. Referencias débiles, para no acumular uno por persona para siempre:
# mientras alguien lo usa, la variable local lo mantiene vivo.
_CANDADOS: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()


def _candado_de(usuario_id: UUID) -> asyncio.Lock:
    candado = _CANDADOS.get(usuario_id)
    if candado is None:
        candado = asyncio.Lock()
        _CANDADOS[usuario_id] = candado
    return candado


class ConectarNotion:
    """Obtiene, guarda, renueva y revoca las credenciales de Notion."""

    def __init__(self, tokens: OAuthTokenRepository, autorizador: AutorizadorNotion) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._tokens = tokens
        self._autorizador = autorizador

    def link_de_autorizacion(self, usuario_id: UUID, ahora: datetime) -> str:
        """Devuelve la URL de consentimiento para mandarle a la persona."""
        return self._autorizador.url_de_autorizacion(usuario_id, ahora)

    def usuario_del_estado(self, estado: str, ahora: datetime) -> UUID:
        """Verifica un `state` firmado y devuelve de quién es.

        Raises:
            InvalidValueError: Si está deformado, mal firmado o vencido.
        """
        return self._autorizador.usuario_del_estado(estado, ahora)

    async def completar(self, codigo: str, estado: str, ahora: datetime) -> UUID:
        """Procesa la vuelta de Notion y deja las credenciales guardadas.

        Returns:
            El id del usuario que quedó conectado.

        Raises:
            InvalidValueError: Si el `state` es inválido o venció.
            AutorizacionFallidaError: Si Notion rechaza el canje.
        """
        # Primero el state: si no vale, ni se gasta el código ni se toca la base.
        usuario_id = self._autorizador.usuario_del_estado(estado, ahora)
        credenciales = await self._autorizador.canjear_codigo(codigo)

        candado = _candado_de(usuario_id)
        async with candado:
            # Una autorización nueva trae su propio par: no se conserva nada
            # del anterior, que es de otro consentimiento.
            await self._tokens.guardar(_token(usuario_id, credenciales))

        logger.info(
            "notion.cuenta_conectada",
            usuario_id=str(usuario_id),
            trajo_refresh=credenciales.refresh_token is not None,
        )
        return usuario_id

    async def desconectar(self, usuario_id: UUID) -> bool:
        """Revoca la autorización y borra las credenciales (RF-12).

        Se revocan las dos credenciales mientras todavía las tenemos, y se
        borra igual aunque Notion no confirme: quedarse con un token "para
        reintentar" sería conservar algo que la persona pidió eliminar.

        Returns:
            True si Notion confirmó la revocación; False si sólo se borró de
            nuestro lado (el llamador se lo cuenta a la persona).
        """
        candado = _candado_de(usuario_id)
        async with candado:
            token = await self._tokens.obtener(usuario_id, PROVEEDOR)
            if token is None:
                return True  # no había nada: el estado final es el pedido

            revocado = await self._autorizador.revocar(token.access_token)
            if token.refresh_token is not None:
                refresh_revocado = await self._autorizador.revocar(token.refresh_token)
                revocado = revocado and refresh_revocado
            await self._tokens.eliminar(usuario_id, PROVEEDOR)

        logger.info(
            "notion.cuenta_desconectada",
            usuario_id=str(usuario_id),
            revocado_en_notion=revocado,
        )
        return revocado

    async def esta_conectado(self, usuario_id: UUID) -> bool:
        """Indica si la persona ya autorizó su Notion (RF-12)."""
        return await self._tokens.obtener(usuario_id, PROVEEDOR) is not None

    async def credencial(self, usuario_id: UUID) -> OAuthToken:
        """El token guardado, tal cual: Notion no informa cuándo vence.

        Raises:
            NotionNoConectadoError: Si la persona no conectó su Notion.
        """
        token = await self._tokens.obtener(usuario_id, PROVEEDOR)
        if token is None:
            raise NotionNoConectadoError
        return token

    async def renovar(self, usuario_id: UUID, vencido: str) -> OAuthToken:
        """Renueva la credencial que Notion acaba de rechazar con un 401.

        Args:
            vencido: El access_token rechazado. Si el guardado ya es otro, lo
                renovó otro pedido mientras éste esperaba el candado: se usa ése
                y no se gasta el refresh_token.

        Raises:
            NotionNoConectadoError: Si la persona desconectó Notion mientras
                tanto (no se revive una conexión borrada).
            AutorizacionFallidaError: Si no hay con qué renovar o Notion lo
                rechaza; el remedio es /conectar notion.
        """
        candado = _candado_de(usuario_id)
        async with candado:
            token = await self._tokens.obtener(usuario_id, PROVEEDOR)
            if token is None:
                raise NotionNoConectadoError
            if token.access_token != vencido:
                return token
            if token.refresh_token is None:
                logger.warning("notion.token_rechazado_sin_refresh", usuario_id=str(usuario_id))
                raise AutorizacionFallidaError(
                    "La conexión con Notion venció y hay que volver a autorizarla."
                )

            try:
                credenciales = await self._autorizador.refrescar(token.refresh_token)
            except AutorizacionFallidaError:
                # Durante un deploy conviven dos contenedores, y el candado es
                # de cada proceso: si el otro ya renovó con el mismo refresh, el
                # rechazo es por eso y vale lo que guardó.
                actual = await self._tokens.obtener(usuario_id, PROVEEDOR)
                if actual is not None and actual.access_token != vencido:
                    return actual
                raise

            renovado = _token(usuario_id, credenciales, anterior=token)
            try:
                guardado = await self._tokens.guardar(renovado)
            except RepositoryError:
                # Notion ya pudo haber invalidado el refresh viejo: perder el
                # nuevo obligaría a reconectar. Un reintento, y que quede a la vista.
                logger.error("notion.renovacion_sin_guardar", usuario_id=str(usuario_id))
                guardado = await self._tokens.guardar(renovado)

        logger.info(
            "notion.token_renovado",
            usuario_id=str(usuario_id),
            refresh_nuevo=credenciales.refresh_token is not None,
        )
        return guardado


def _token(
    usuario_id: UUID, credenciales: CredencialesNotion, anterior: OAuthToken | None = None
) -> OAuthToken:
    """El token a guardar. En una renovación, un refresh_token vacío no pisa el que había.

    Es la lección de `ConectarGoogle._guardar`: escribir None sobre el único
    refresh_token deja la conexión muerta hasta reconectar a mano.
    """
    refresh = credenciales.refresh_token
    if refresh is None and anterior is not None:
        refresh = anterior.refresh_token
    return OAuthToken(
        usuario_id=usuario_id,
        proveedor=PROVEEDOR,
        access_token=credenciales.access_token,
        refresh_token=refresh,
    )
