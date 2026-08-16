"""Implementación Supabase del repositorio de usuarios (PB-003).

La búsqueda por teléfono es la que va a usar el webhook de WhatsApp (PB-004)
para resolver quién está escribiendo.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from supabase import AsyncClient

from src.domain.entities.usuario import Usuario
from src.domain.exceptions import RepositoryError
from src.domain.repositories.usuario_repository import UsuarioRepository
from src.infrastructure.persistence.error_translation import traducir_errores
from src.infrastructure.persistence.mapeo import (
    a_datetime,
    a_texto,
    a_texto_opcional,
    a_uuid,
    filas_de,
)

logger = structlog.get_logger(__name__)

TABLA = "usuarios"


class SupabaseUsuarioRepository(UsuarioRepository):
    """Persiste usuarios en la tabla `usuarios` de Supabase."""

    def __init__(self, cliente: AsyncClient) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._cliente = cliente

    async def crear(self, usuario: Usuario) -> Usuario:
        """Da de alta un usuario."""
        fila: dict[str, Any] = {
            "telefono_whatsapp": usuario.telefono_whatsapp,
            "nombre": usuario.nombre,
        }

        async with traducir_errores("usuarios.insert"):
            respuesta = await self._cliente.table(TABLA).insert(fila).execute()

        filas = filas_de(respuesta)
        if not filas:
            raise RepositoryError("El alta del usuario no devolvió la fila creada.")

        creado = self._a_entidad(filas[0])
        logger.info("usuario.creado", usuario_id=str(creado.id))
        return creado

    async def obtener_por_id(self, usuario_id: UUID) -> Usuario | None:
        """Devuelve el usuario con ese id, o None si no existe."""
        async with traducir_errores("usuarios.select_por_id"):
            respuesta = await (
                self._cliente.table(TABLA).select("*").eq("id", str(usuario_id)).limit(1).execute()
            )

        filas = filas_de(respuesta)
        return self._a_entidad(filas[0]) if filas else None

    async def obtener_por_telefono(self, telefono: str) -> Usuario | None:
        """Devuelve el usuario con ese número de WhatsApp, o None si no existe."""
        async with traducir_errores("usuarios.select_por_telefono"):
            respuesta = await (
                self._cliente.table(TABLA)
                .select("*")
                .eq("telefono_whatsapp", telefono)
                .limit(1)
                .execute()
            )

        filas = filas_de(respuesta)
        return self._a_entidad(filas[0]) if filas else None

    @staticmethod
    def _a_entidad(fila: dict[str, Any]) -> Usuario:
        """Reconstruye la entidad a partir de una fila de PostgREST."""
        return Usuario(
            id=a_uuid(fila.get("id"), "id"),
            telefono_whatsapp=a_texto(fila.get("telefono_whatsapp"), "telefono_whatsapp"),
            nombre=a_texto_opcional(fila.get("nombre"), "nombre"),
            creado_en=a_datetime(fila.get("creado_en"), "creado_en"),
            actualizado_en=a_datetime(fila.get("actualizado_en"), "actualizado_en"),
        )
