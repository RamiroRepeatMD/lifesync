"""Implementación Supabase del repositorio de recordatorios (PB-030).

El texto se cifra acá, en el borde, con el mismo `TokenCipher` de los tokens
OAuth: a PostgreSQL sólo llega el Fernet (y la migración 004 tiene un CHECK
que rechaza cualquier otra cosa). Por eso las búsquedas por texto —cancelar
"el de la pizza"— se hacen en Python sobre los pendientes ya descifrados.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from supabase import AsyncClient

from src.domain.entities.recordatorio import EstadoDeRecordatorio, Recordatorio
from src.domain.exceptions import RepositoryError
from src.domain.repositories.recordatorio_repository import RecordatorioRepository
from src.infrastructure.persistence.encryption import TokenCipher
from src.infrastructure.persistence.error_translation import traducir_errores
from src.infrastructure.persistence.mapeo import a_datetime, a_texto, a_uuid, filas_de

logger = structlog.get_logger(__name__)

TABLA = "recordatorios"


class SupabaseRecordatorioRepository(RecordatorioRepository):
    """Persiste recordatorios en la tabla `recordatorios` de Supabase."""

    def __init__(self, cliente: AsyncClient, cifrador: TokenCipher) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._cliente = cliente
        self._cifrador = cifrador

    async def crear(self, recordatorio: Recordatorio) -> Recordatorio:
        """Guarda un recordatorio nuevo, con el texto cifrado."""
        fila: dict[str, Any] = {
            "usuario_id": str(recordatorio.usuario_id),
            "texto_cifrado": self._cifrador.cifrar(recordatorio.texto),
            "momento": recordatorio.momento.isoformat(),
            "estado": recordatorio.estado.value,
        }

        async with traducir_errores("recordatorios.insert"):
            respuesta = await self._cliente.table(TABLA).insert(fila).execute()

        filas = filas_de(respuesta)
        if not filas:
            raise RepositoryError("El alta del recordatorio no devolvió la fila creada.")

        creado = self._a_entidad(filas[0])
        # Nunca el texto: el id alcanza para seguirlo en los logs (RF-18).
        logger.info(
            "recordatorio.creado",
            recordatorio_id=str(creado.id),
            usuario_id=str(creado.usuario_id),
        )
        return creado

    async def pendientes_de(self, usuario_id: UUID) -> list[Recordatorio]:
        """Los pendientes de una persona, del más próximo al más lejano."""
        async with traducir_errores("recordatorios.select_pendientes"):
            respuesta = await (
                self._cliente.table(TABLA)
                .select("*")
                .eq("usuario_id", str(usuario_id))
                .eq("estado", EstadoDeRecordatorio.PENDIENTE.value)
                .order("momento")
                .execute()
            )
        return [self._a_entidad(fila) for fila in filas_de(respuesta)]

    async def vencidos(self, hasta: datetime, limite: int) -> list[Recordatorio]:
        """Los pendientes cuyo momento ya llegó, los más viejos primero."""
        async with traducir_errores("recordatorios.select_vencidos"):
            respuesta = await (
                self._cliente.table(TABLA)
                .select("*")
                .eq("estado", EstadoDeRecordatorio.PENDIENTE.value)
                .lte("momento", hasta.isoformat())
                .order("momento")
                .limit(limite)
                .execute()
            )
        return [self._a_entidad(fila) for fila in filas_de(respuesta)]

    async def cambiar_estado(
        self,
        recordatorio_id: UUID,
        desde: EstadoDeRecordatorio,
        hacia: EstadoDeRecordatorio,
    ) -> bool:
        """Compare-and-set: `UPDATE … WHERE id = … AND estado = desde`.

        PostgREST devuelve las filas que cambió: ninguna quiere decir que el
        recordatorio ya no estaba en `desde`, y otro llamado ganó.
        """
        cambios: dict[str, Any] = {"estado": hacia.value}
        if hacia is EstadoDeRecordatorio.ENVIADO:
            cambios["enviado_en"] = datetime.now(UTC).isoformat()

        async with traducir_errores("recordatorios.update_estado"):
            respuesta = await (
                self._cliente.table(TABLA)
                .update(cambios)
                .eq("id", str(recordatorio_id))
                .eq("estado", desde.value)
                .execute()
            )
        return bool(filas_de(respuesta))

    def _a_entidad(self, fila: dict[str, Any]) -> Recordatorio:
        """Reconstruye la entidad a partir de una fila de PostgREST."""
        momento = a_datetime(fila.get("momento"), "momento")
        if momento is None:
            raise RepositoryError("Falta la columna 'momento' en la respuesta.")
        try:
            estado = EstadoDeRecordatorio(a_texto(fila.get("estado"), "estado"))
        except ValueError as exc:
            raise RepositoryError("La columna 'estado' trae un valor desconocido.") from exc

        return Recordatorio(
            usuario_id=a_uuid(fila.get("usuario_id"), "usuario_id"),
            texto=self._cifrador.descifrar(a_texto(fila.get("texto_cifrado"), "texto_cifrado")),
            momento=momento,
            estado=estado,
            id=a_uuid(fila.get("id"), "id"),
        )
