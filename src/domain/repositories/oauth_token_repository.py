"""Puerto de persistencia de tokens OAuth2 (RF-01).

Define QUÉ necesita el negocio, nunca CÓMO se guarda. La implementación contra
Supabase vive en `src/infrastructure/persistence/`.

Los tokens entran y salen de este puerto **en texto plano**: cifrarlos al
escribir y descifrarlos al leer es responsabilidad de la implementación.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.oauth_token import OAuthToken
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth


class OAuthTokenRepository(ABC):
    """Almacén de credenciales OAuth2 por usuario y proveedor."""

    @abstractmethod
    async def guardar(self, token: OAuthToken) -> OAuthToken:
        """Crea o reemplaza el token del par (usuario, proveedor).

        Returns:
            El token persistido, con `id` y timestamps ya asignados.
        """

    @abstractmethod
    async def obtener(self, usuario_id: UUID, proveedor: ProveedorOAuth) -> OAuthToken | None:
        """Devuelve el token del usuario para ese proveedor, o None si no hay."""

    @abstractmethod
    async def listar_por_usuario(self, usuario_id: UUID) -> list[OAuthToken]:
        """Devuelve todos los tokens del usuario, uno por proveedor conectado."""

    @abstractmethod
    async def eliminar(self, usuario_id: UUID, proveedor: ProveedorOAuth) -> None:
        """Borra el token, desconectando la integración (RF-12).

        Es idempotente: borrar algo que no existe no es un error.
        """
