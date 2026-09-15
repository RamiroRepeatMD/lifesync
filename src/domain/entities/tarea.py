"""La tarea pendiente de una persona (PB-028).

Modelo mínimo alineado con lo que Google Tasks realmente guarda. El detalle
que importa: el vencimiento es un `date`, no un `datetime` — la API descarta
la parte horaria del campo `due` (documentado por Google). Modelarlo con hora
sería prometer una precisión que el proveedor no conserva; si la persona da
una hora, eso es un evento de calendario, no una tarea.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class Tarea:
    """Una tarea de la lista de la persona.

    Attributes:
        titulo: Qué hay que hacer. Fuera del repr: es contenido de la persona.
        vencimiento: Fecha límite, si la tiene. Sin hora, a propósito.
        notas: Detalle libre. Fuera del repr por el mismo motivo que el título.
        completada: Si ya se hizo.
        id: Identificador del proveedor. `None` para una tarea aún no creada.
    """

    titulo: str = field(repr=False)
    vencimiento: date | None = None
    notas: str | None = field(default=None, repr=False)
    completada: bool = False
    id: str | None = None
