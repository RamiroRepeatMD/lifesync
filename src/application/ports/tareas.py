"""Puerto de acceso a las tareas de la persona (PB-028).

Vive en `application/ports` y no en `domain/repositories` por la misma regla
que `Calendario`: no es la persistencia de nuestras entidades sino una
capacidad externa — las tareas viven en el proveedor y esta interfaz sólo
describe qué necesita el agente de ellas.

Alcance deliberado del PB: la lista por defecto (`@default`) solamente, y las
operaciones listar, crear y completar. Posponer y eliminar son PB-029.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.tarea import Tarea


class Tareas(ABC):
    """Lo que el agente puede hacer con las tareas de quien escribe."""

    @abstractmethod
    async def pendientes(self, usuario_id: UUID) -> tuple[Tarea, ...]:
        """Devuelve las tareas no completadas de la lista principal.

        Raises:
            CuentaNoConectadaError: Si la persona nunca autorizó su cuenta.
            PermisoInsuficienteError: Si el token no tiene el scope de tareas.
            AutorizacionFallidaError: Si la autorización venció o fue revocada.
            ServiceUnavailableError: Si el proveedor no responde.
        """

    @abstractmethod
    async def crear(self, usuario_id: UUID, tarea: Tarea) -> Tarea:
        """Crea la tarea en la lista principal y la devuelve con su id.

        Raises:
            Las mismas de `pendientes`.
        """

    @abstractmethod
    async def completar(self, usuario_id: UUID, tarea_id: str) -> None:
        """Marca la tarea como hecha.

        Raises:
            EntityNotFoundError: Si la tarea ya no existe.
            Las mismas de `pendientes`.
        """
