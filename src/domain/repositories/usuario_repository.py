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
            InvalidValueError: Si el teléfono ya está registrado. Es
                `InvalidValueError` y no `RepositoryError`: son ramas hermanas
                de la jerarquía, así que un `except RepositoryError` escrito
                leyendo mal este docstring no atraparía nada.
        """

    @abstractmethod
    async def obtener_o_crear(self, telefono: str, nombre: str | None = None) -> Usuario:
        """Devuelve el usuario de ese teléfono, dándolo de alta si no existe.

        Es la operación que necesita el webhook de WhatsApp: cuando alguien
        escribe por primera vez, hay que registrarlo sin que eso sea un caso
        especial para el llamador.

        Debe ser segura ante dos mensajes concurrentes del mismo usuario nuevo:
        la unicidad la garantiza la base, no una comprobación previa.
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
