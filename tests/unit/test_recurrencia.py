"""Tests del value object `Recurrencia` (PB-025)."""

from __future__ import annotations

from datetime import date

import pytest

from src.domain.exceptions import InvalidValueError
from src.domain.value_objects.recurrencia import Frecuencia, Recurrencia


@pytest.mark.parametrize(
    ("recurrencia", "texto"),
    [
        (Recurrencia(Frecuencia.DIARIA), "todos los días"),
        (Recurrencia(Frecuencia.MENSUAL), "todos los meses"),
        (Recurrencia(Frecuencia.SEMANAL, (0,)), "todos los lunes"),
        (Recurrencia(Frecuencia.SEMANAL, (2, 0)), "todos los lunes y miércoles"),
        (Recurrencia(Frecuencia.SEMANAL, (5, 6)), "todos los sábados y domingos"),
        (Recurrencia(Frecuencia.SEMANAL, (0, 2, 4)), "todos los lunes, miércoles y viernes"),
    ],
)
def test_se_describe_en_castellano(recurrencia: Recurrencia, texto: str) -> None:
    assert recurrencia.describir() == texto


@pytest.mark.parametrize(
    "armar",
    [
        # Termina en una fecha O después de N veces: no las dos cosas.
        lambda: Recurrencia(Frecuencia.DIARIA, hasta=date(2026, 12, 31), veces=10),
        lambda: Recurrencia(Frecuencia.DIARIA, veces=0),
        lambda: Recurrencia(Frecuencia.DIARIA, veces=366),
        # Los días de la semana sólo tienen sentido repitiendo cada semana.
        lambda: Recurrencia(Frecuencia.DIARIA, (0,)),
        lambda: Recurrencia(Frecuencia.MENSUAL, (0,)),
        lambda: Recurrencia(Frecuencia.SEMANAL),  # semanal sin días
        lambda: Recurrencia(Frecuencia.SEMANAL, (7,)),  # fuera de rango
        lambda: Recurrencia(Frecuencia.SEMANAL, (0, 0)),  # repetidos
    ],
    ids=[
        "hasta-y-veces",
        "cero-veces",
        "demasiadas-veces",
        "dias-en-diaria",
        "dias-en-mensual",
        "semanal-sin-dias",
        "dia-fuera-de-rango",
        "dias-repetidos",
    ],
)
def test_las_reglas_sin_sentido_no_se_pueden_construir(armar: object) -> None:
    with pytest.raises(InvalidValueError):
        armar()  # type: ignore[operator]


def test_una_regla_valida_se_construye() -> None:
    regla = Recurrencia(Frecuencia.SEMANAL, (0, 2), veces=4)

    assert regla.dias == (0, 2)
    assert regla.veces == 4
