"""Puerto de persistencia de usuarios.

Define QUÉ necesita el negocio, nunca CÓMO se guarda. La implementación contra
Supabase vive en `src/infrastructure/persistence/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.usuario import Usuario


class UsuarioRepository(ABC):
    """Almacén de los usuarios registrados del sistema."""

    @abstractmethod
    async def crear(self, usuario: Usuario) -> Usuario:
        """Da de alta un usuario.

        Returns:
            El usuario persistido, con `id` y timestamps ya asignados.

        Raises:
            RepositoryError: Si el teléfono ya está registrado.
        """

    @abstractmethod
    async def obtener_por_id(self, usuario_id: UUID) -> Usuario | None:
        """Devuelve el usuario con ese id, o None si no existe."""

    @abstractmethod
    async def obtener_por_telefono(self, telefono: str) -> Usuario | None:
        """Devuelve el usuario con ese número de WhatsApp, o None si no existe.

        Es la búsqueda que va a usar el webhook de WhatsApp (PB-004) para
        resolver quién está escribiendo.
        """
