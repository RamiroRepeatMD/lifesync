"""Entidad Evento: una cita en el calendario de una persona (RF-03).

Es lo que LifeSync entiende por "evento", independiente de que hoy venga de
Google Calendar. Si mañana se suma otro proveedor, esta entidad no cambia: lo
que cambia es el adaptador que la construye.

Sólo stdlib, como todo `domain/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.domain.exceptions import InvalidValueError

# Google permite eventos sin título. Mostrar una línea vacía sería peor que
# decir que no tiene nombre.
SIN_TITULO = "(sin título)"


@dataclass(frozen=True, slots=True)
class Evento:
    """Un evento del calendario, ya normalizado.

    Attributes:
        titulo: Nombre del evento. `repr=False` porque es dato personal: no
            debe aparecer en un traceback ni en un assert de pytest (RF-18).
        inicio: Cuándo empieza, **siempre con zona horaria**.
        fin: Cuándo termina; puede faltar.
        todo_el_dia: Los eventos de día completo llegan de Google con `date`
            en vez de `dateTime`. Se marca para poder mostrarlos como "todo el
            día" en lugar de inventar un horario de 00:00.
        calendario: De qué calendario salió. Con varios calendarios en juego,
            sin esto no se puede distinguir el cumpleaños del turno médico.
    """

    titulo: str = field(repr=False)
    inicio: datetime
    fin: datetime | None = None
    todo_el_dia: bool = False
    calendario: str | None = None

    def __post_init__(self) -> None:
        """Valida las invariantes de la entidad al construirla."""
        if self.inicio.tzinfo is None:
            raise InvalidValueError(
                "El inicio del evento debe tener zona horaria: comparar fechas naive "
                "mezcla eventos de días distintos según dónde corra el proceso."
            )
        if self.fin is not None and self.fin.tzinfo is None:
            raise InvalidValueError("El fin del evento debe tener zona horaria.")
        if self.fin is not None and self.fin < self.inicio:
            raise InvalidValueError("Un evento no puede terminar antes de empezar.")

    @property
    def titulo_visible(self) -> str:
        """Título para mostrar, contemplando los eventos sin nombre."""
        return self.titulo.strip() or SIN_TITULO
