"""Puerto de persistencia de los recordatorios (PB-030).

Define QUÉ necesitan el agente y el despachador, nunca CÓMO se guarda. La
implementación contra Supabase vive en `src/infrastructure/persistence/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from src.domain.entities.recordatorio import EstadoDeRecordatorio, Recordatorio


class RecordatorioRepository(ABC):
    """Almacén de los recordatorios programados."""

    @abstractmethod
    async def crear(self, recordatorio: Recordatorio) -> Recordatorio:
        """Guarda un recordatorio nuevo.

        Returns:
            El recordatorio persistido, con `id` ya asignado.
        """

    @abstractmethod
    async def pendientes_de(self, usuario_id: UUID) -> list[Recordatorio]:
        """Los pendientes de una persona, del más próximo al más lejano."""

    @abstractmethod
    async def vencidos(self, hasta: datetime, limite: int) -> list[Recordatorio]:
        """Los pendientes de todas las personas cuyo momento ya llegó.

        Son los que tienen `momento <= hasta`, los más viejos primero, como
        mucho `limite`. Es la consulta del despachador.
        """

    @abstractmethod
    async def cambiar_estado(
        self,
        recordatorio_id: UUID,
        desde: EstadoDeRecordatorio,
        hacia: EstadoDeRecordatorio,
    ) -> bool:
        """Pasa el recordatorio de `desde` a `hacia`, sólo si sigue en `desde`.

        Es un *compare-and-set*: la condición y el cambio son UNA operación en
        la base. Así, si dos despachadores reclaman el mismo recordatorio,
        exactamente uno gana.

        Returns:
            True si este llamado hizo el cambio; False si el recordatorio ya
            no estaba en `desde` (otro lo reclamó, ya salió o se canceló).
        """
