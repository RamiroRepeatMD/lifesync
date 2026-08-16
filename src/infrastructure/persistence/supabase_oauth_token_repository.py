"""Implementación Supabase del repositorio de tokens OAuth2 (RF-01 + RF-18).

Acá vive el borde del cifrado: los tokens se cifran justo antes de salir hacia
PostgreSQL y se descifran justo después de leerlos. Hacia el dominio, el
repositorio devuelve siempre entidades con el token en texto plano.

Regla dura de este módulo: **nunca se loguea un token**, ni cifrado ni plano.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from supabase import AsyncClient

from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import RepositoryError
from src.domain.repositories.oauth_token_repository import OAuthTokenRepository
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.persistence.encryption import TokenCipher
from src.infrastructure.persistence.error_translation import traducir_errores
from src.infrastructure.persistence.mapeo import (
    a_datetime,
    a_texto,
    a_texto_opcional,
    a_tupla_de_textos,
    a_uuid,
    filas_de,
    formatear_fecha,
)

logger = structlog.get_logger(__name__)

TABLA = "oauth_tokens"
CLAVE_UNICA = "usuario_id,proveedor"


class SupabaseOAuthTokenRepository(OAuthTokenRepository):
    """Guarda tokens OAuth2 cifrados en la tabla `oauth_tokens` de Supabase."""

    def __init__(self, cliente: AsyncClient, cipher: TokenCipher) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._cliente = cliente
        self._cipher = cipher

    async def guardar(self, token: OAuthToken) -> OAuthToken:
        """Crea o reemplaza el token del par (usuario, proveedor).

        El repositorio persiste exactamente lo que dice la entidad: si
        `refresh_token` es None, la columna queda en NULL. Ojo en PB-009:
        Google emite el refresh_token sólo en el primer consentimiento, así
        que el caso de uso debe conservar el guardado en vez de pisarlo con
        None cuando el proveedor no lo devuelve.
        """
        fila = self._a_fila(token)

        async with traducir_errores("oauth_tokens.upsert"):
            respuesta = await (
                self._cliente.table(TABLA).upsert(fila, on_conflict=CLAVE_UNICA).execute()
            )

        filas = filas_de(respuesta)
        if not filas:
            raise RepositoryError("El upsert del token no devolvió la fila guardada.")

        guardado = self._a_entidad(filas[0])
        logger.info(
            "oauth_token.guardado",
            usuario_id=str(token.usuario_id),
            proveedor=token.proveedor.value,
            cantidad_scopes=len(token.scopes),
        )
        return guardado

    async def obtener(self, usuario_id: UUID, proveedor: ProveedorOAuth) -> OAuthToken | None:
        """Devuelve el token del usuario para ese proveedor, o None si no hay."""
        async with traducir_errores("oauth_tokens.select"):
            respuesta = await (
                self._cliente.table(TABLA)
                .select("*")
                .eq("usuario_id", str(usuario_id))
                .eq("proveedor", proveedor.value)
                .limit(1)
                .execute()
            )

        filas = filas_de(respuesta)
        return self._a_entidad(filas[0]) if filas else None

    async def listar_por_usuario(self, usuario_id: UUID) -> list[OAuthToken]:
        """Devuelve todos los tokens del usuario, uno por proveedor conectado."""
        async with traducir_errores("oauth_tokens.select_por_usuario"):
            respuesta = await (
                self._cliente.table(TABLA).select("*").eq("usuario_id", str(usuario_id)).execute()
            )

        return [self._a_entidad(fila) for fila in filas_de(respuesta)]

    async def eliminar(self, usuario_id: UUID, proveedor: ProveedorOAuth) -> None:
        """Borra el token, desconectando la integración (RF-12)."""
        async with traducir_errores("oauth_tokens.delete"):
            await (
                self._cliente.table(TABLA)
                .delete()
                .eq("usuario_id", str(usuario_id))
                .eq("proveedor", proveedor.value)
                .execute()
            )

        logger.info(
            "oauth_token.eliminado",
            usuario_id=str(usuario_id),
            proveedor=proveedor.value,
        )

    # --- Conversión entidad <-> fila -------------------------------------

    def _a_fila(self, token: OAuthToken) -> dict[str, Any]:
        """Serializa la entidad, cifrando las credenciales.

        Sólo emite tipos serializables a JSON: httpx usa `json.dumps` sin
        encoder propio, así que un `UUID` o un `datetime` crudos explotarían
        al momento de enviar la request.

        Nunca se envía `id`: la clave primaria la asigna y la conserva la
        base. Mandarla haría que un upsert con un UUID recién generado
        reescribiera la PK de la fila existente en cada guardado.
        """
        return {
            "usuario_id": str(token.usuario_id),
            "proveedor": token.proveedor.value,
            "access_token_cifrado": self._cipher.cifrar(token.access_token),
            "refresh_token_cifrado": (
                None if token.refresh_token is None else self._cipher.cifrar(token.refresh_token)
            ),
            "expira_en": formatear_fecha(token.expira_en),
            "scopes": list(token.scopes),
        }

    def _a_entidad(self, fila: dict[str, Any]) -> OAuthToken:
        """Reconstruye la entidad, descifrando las credenciales."""
        refresh_cifrado = a_texto_opcional(
            fila.get("refresh_token_cifrado"), "refresh_token_cifrado"
        )

        return OAuthToken(
            id=a_uuid(fila.get("id"), "id"),
            usuario_id=a_uuid(fila.get("usuario_id"), "usuario_id"),
            proveedor=ProveedorOAuth(a_texto(fila.get("proveedor"), "proveedor")),
            access_token=self._cipher.descifrar(
                a_texto(fila.get("access_token_cifrado"), "access_token_cifrado")
            ),
            refresh_token=(
                None if refresh_cifrado is None else self._cipher.descifrar(refresh_cifrado)
            ),
            expira_en=a_datetime(fila.get("expira_en"), "expira_en"),
            scopes=a_tupla_de_textos(fila.get("scopes"), "scopes"),
            creado_en=a_datetime(fila.get("creado_en"), "creado_en"),
            actualizado_en=a_datetime(fila.get("actualizado_en"), "actualizado_en"),
        )
