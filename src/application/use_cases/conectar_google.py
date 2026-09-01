"""Caso de uso: conectar la cuenta de Google de una persona (PB-009, RF-01).

Cruza los dos mundos del flujo: la persona está en WhatsApp y el consentimiento
pasa por el navegador. Acá viven las tres operaciones que eso implica —armar el
link, procesar la vuelta y mantener la credencial vigente— sin que ninguna sepa
que del otro lado hay HTTP.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import structlog

from src.application.ports.autorizador_google import AutorizadorGoogle, CredencialesGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import AutorizacionFallidaError
from src.domain.repositories.oauth_token_repository import OAuthTokenRepository
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth

logger = structlog.get_logger(__name__)

PROVEEDOR = ProveedorOAuth.GOOGLE


class ConectarGoogle:
    """Obtiene, guarda y mantiene vigentes las credenciales de Google."""

    def __init__(self, tokens: OAuthTokenRepository, autorizador: AutorizadorGoogle) -> None:
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
        """Procesa la vuelta de Google y deja las credenciales guardadas.

        Returns:
            El id del usuario que quedó conectado.

        Raises:
            InvalidValueError: Si el `state` es inválido o venció.
            AutorizacionFallidaError: Si Google rechaza el canje.
        """
        # Primero el state: si no es válido no hay que gastar un canje ni
        # tocar la base. Y si es inválido, no sabemos de quién es el código.
        usuario_id = self._autorizador.usuario_del_estado(estado, ahora)

        credenciales = await self._autorizador.canjear_codigo(codigo)
        token = await self._guardar(usuario_id, credenciales)

        logger.info(
            "google.cuenta_conectada",
            usuario_id=str(usuario_id),
            cantidad_scopes=len(token.scopes),
        )
        return usuario_id

    async def esta_conectado(self, usuario_id: UUID) -> bool:
        """Indica si la persona ya autorizó su cuenta de Google (RF-12)."""
        return await self._tokens.obtener(usuario_id, PROVEEDOR) is not None

    async def credencial_vigente(self, usuario_id: UUID, ahora: datetime) -> OAuthToken | None:
        """Devuelve un token utilizable, renovándolo si hizo falta.

        Es el **refresco perezoso**: en vez de una tarea que revisa
        vencimientos, se renueva recién cuando alguien va a usar la credencial.
        Sin esto la conexión sirve una hora y después empieza a fallar.

        Returns:
            El token vigente, o None si la persona nunca conectó su cuenta.

        Raises:
            AutorizacionFallidaError: Si venció y no se puede renovar —porque
                no hay refresh_token o porque la persona revocó el acceso—.
        """
        token = await self._tokens.obtener(usuario_id, PROVEEDOR)
        if token is None:
            return None
        if not token.esta_vencido(ahora):
            return token

        if not token.puede_renovarse():
            logger.warning("google.token_vencido_sin_refresh", usuario_id=str(usuario_id))
            raise AutorizacionFallidaError(
                "La conexión con Google venció y hay que volver a autorizarla."
            )

        # mypy: puede_renovarse() ya garantizó que no es None.
        assert token.refresh_token is not None  # noqa: S101
        logger.info("google.renovando_token", usuario_id=str(usuario_id))
        credenciales = await self._autorizador.refrescar(token.refresh_token)
        return await self._guardar(usuario_id, credenciales, anterior=token)

    async def _guardar(
        self,
        usuario_id: UUID,
        credenciales: CredencialesGoogle,
        anterior: OAuthToken | None = None,
    ) -> OAuthToken:
        """Persiste las credenciales conservando el refresh_token que ya había.

        **Éste es el punto delicado de todo el PB.** Google emite el
        refresh_token en el primer consentimiento y no lo repite: ni al
        renovar, ni necesariamente al reconectar. Si se guardara tal cual lo
        que vuelve, la primera renovación escribiría NULL sobre el único
        refresh_token que teníamos y la conexión quedaría muerta hasta que la
        persona volviera a autorizar a mano.

        Está advertido en el docstring de
        `SupabaseOAuthTokenRepository.guardar`, que es donde se descubrió.
        """
        guardado = anterior
        if credenciales.refresh_token is None and guardado is None:
            guardado = await self._tokens.obtener(usuario_id, PROVEEDOR)

        refresh = credenciales.refresh_token or (guardado.refresh_token if guardado else None)
        if credenciales.refresh_token is None and refresh is not None:
            logger.info("google.refresh_token_conservado", usuario_id=str(usuario_id))

        return await self._tokens.guardar(
            OAuthToken(
                usuario_id=usuario_id,
                proveedor=PROVEEDOR,
                access_token=credenciales.access_token,
                refresh_token=refresh,
                expira_en=credenciales.expira_en,
                # Los scopes que valen son los que Google concedió, que pueden
                # ser menos que los pedidos si la persona destildó alguno.
                scopes=credenciales.scopes or (guardado.scopes if guardado else ()),
            )
        )
