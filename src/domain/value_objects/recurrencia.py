"""Cómo se repite un evento (PB-025).

Un value object y no un string con la regla de iCalendar: la regla (`RRULE`)
es un formato del proveedor, y la arma el adaptador a partir de esto — nunca
el modelo. Así el dominio valida lo que tiene sentido para una persona
("los lunes y miércoles, 4 veces") y no le llega texto arbitrario a Google.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from src.domain.exceptions import InvalidValueError

# Índice 0 = lunes, como `date.weekday()`.
NOMBRES_DE_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_PLURALES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábados", "domingos")
MAX_VECES = 365


class Frecuencia(StrEnum):
    """Cada cuánto se repite."""

    DIARIA = "diaria"
    SEMANAL = "semanal"
    MENSUAL = "mensual"


@dataclass(frozen=True, slots=True)
class Recurrencia:
    """Una regla de repetición simple, ya validada.

    Attributes:
        frecuencia: Diaria, semanal o mensual.
        dias: Días de la semana (0 = lunes … 6 = domingo). Sólo en la semanal,
            donde es obligatorio.
        hasta: Último día en que puede haber una repetición, si termina en
            una fecha.
        veces: Cuántas repeticiones hay en total, si termina por cantidad.
            `hasta` y `veces` son excluyentes; sin ninguno, no termina.
    """

    frecuencia: Frecuencia
    dias: tuple[int, ...] = ()
    hasta: date | None = None
    veces: int | None = None

    def __post_init__(self) -> None:
        """Valida las invariantes al construirla."""
        if self.hasta is not None and self.veces is not None:
            raise InvalidValueError(
                "Una repetición termina en una fecha o después de cierta cantidad de veces, "
                "no las dos cosas."
            )
        if self.veces is not None and not 1 <= self.veces <= MAX_VECES:
            raise InvalidValueError(f"La cantidad de repeticiones va de 1 a {MAX_VECES}.")
        if self.dias and self.frecuencia is not Frecuencia.SEMANAL:
            raise InvalidValueError("Los días de la semana sólo sirven para repetir cada semana.")
        if any(not 0 <= dia <= 6 for dia in self.dias):
            raise InvalidValueError("Los días de la semana van del lunes al domingo.")
        if len(set(self.dias)) != len(self.dias):
            raise InvalidValueError("Hay días de la semana repetidos.")
        if self.frecuencia is Frecuencia.SEMANAL and not self.dias:
            raise InvalidValueError("Para repetir cada semana hace falta al menos un día.")

    def describir(self) -> str:
        """La regla en castellano, sin el final: "todos los lunes y miércoles"."""
        if self.frecuencia is Frecuencia.DIARIA:
            return "todos los días"
        if self.frecuencia is Frecuencia.MENSUAL:
            return "todos los meses"
        return "todos los " + _enumerar([_PLURALES[dia] for dia in sorted(self.dias)])


def _enumerar(partes: list[str]) -> str:
    """["lunes", "martes", "jueves"] → "lunes, martes y jueves"."""
    if len(partes) == 1:
        return partes[0]
    return f"{', '.join(partes[:-1])} y {partes[-1]}"
