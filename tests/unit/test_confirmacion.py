"""Tests del clasificador de confirmaciones (PB-016, RF-08).

Es una función pura y se testea con una tabla. La propiedad que importa: **sólo
un sí explícito aprueba**; todo lo demás cancela. Un falso positivo acá es una
acción ejecutada sin consentimiento — por eso los casos negativos son los que
más pesan.
"""

from __future__ import annotations

import pytest

from src.infrastructure.llm.confirmacion import Decision, clasificar


@pytest.mark.parametrize(
    "texto",
    [
        "sí",
        "si",
        "Si!",
        "SÍ",
        "dale",
        "ok",
        "OK.",
        "confirmo",
        "confirmar",
        "listo",
        "hacelo",
        "mandale",
        "de una",
        "¡Sí!",
        "si, dale",
        "s",
    ],
)
def test_un_si_explicito_aprueba(texto: str) -> None:
    assert clasificar(texto) is Decision.APRUEBA


@pytest.mark.parametrize(
    "texto",
    [
        "no",
        "No.",
        "NO",
        "nop",
        "cancelar",
        "cancela",
        "no gracias",
        "No, gracias",
        "mejor no",
        "dejalo",
        "pará",
        "para",
    ],
)
def test_un_no_explicito_rechaza(texto: str) -> None:
    assert clasificar(texto) is Decision.RECHAZA


@pytest.mark.parametrize(
    "texto",
    [
        "si me parece raro",  # empieza con "si" pero no es un sí
        "no sé, ¿a qué hora era?",  # empieza con "no" pero es una pregunta
        "mmm bueno",  # tibio: no alcanza para ejecutar
        "y que hora era?",
        "qué tengo mañana?",
        "puede ser",
        "supongo",
        "",
        "   ",
        "sí pero a las 15",  # un sí condicionado NO es un sí
    ],
)
def test_todo_lo_demas_es_otra_cosa(texto: str) -> None:
    """La zona gris cancela: nadie crea ni borra nada por un mensaje tibio."""
    assert clasificar(texto) is Decision.OTRA_COSA


def test_un_si_con_relleno_no_aprueba() -> None:
    """Match exacto, nunca substring: es la defensa contra el falso positivo."""
    assert clasificar("si tengo tiempo lo hago") is Decision.OTRA_COSA
