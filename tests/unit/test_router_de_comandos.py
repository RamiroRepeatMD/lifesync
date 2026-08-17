"""Tests de los comandos de respuesta fija (RF-11).

Es una función pura, así que estos tests no necesitan ningún doble.

Desde PB-005 la función devuelve `str | None`: `None` significa "esto es
lenguaje natural" y lo maneja el agente. Los tests de abajo son, sobre todo,
los que impiden que un mensaje libre vuelva a caer en una respuesta enlatada.
"""

from __future__ import annotations

import pytest

from src.application.services.router_de_comandos import AYUDA, ESTADO, respuesta_fija


@pytest.mark.parametrize(
    "texto",
    ["/ayuda", "/AYUDA", "  /Ayuda  ", "/ayuda por favor"],
    ids=["exacto", "mayusculas", "con_espacios", "con_cola"],
)
def test_el_comando_de_ayuda_se_reconoce(texto: str) -> None:
    assert respuesta_fija(texto) == AYUDA


def test_el_comando_de_estado_se_reconoce() -> None:
    assert respuesta_fija("/estado") == ESTADO


@pytest.mark.parametrize(
    "texto",
    ["hola", "agendame una reunion el jueves", "", "   ", "ayuda", "/desconocido", "😀"],
    ids=["saludo", "lenguaje_natural", "vacio", "espacios", "sin_barra", "otro_comando", "emoji"],
)
def test_lo_que_no_es_comando_queda_para_el_agente(texto: str) -> None:
    """None no es "no entendí": es "esto no me toca a mí"."""
    assert respuesta_fija(texto) is None


def test_las_respuestas_no_estan_vacias() -> None:
    for respuesta in (AYUDA, ESTADO):
        assert respuesta.strip()


def test_la_ayuda_lista_los_comandos_que_existen() -> None:
    assert "/ayuda" in AYUDA
    assert "/estado" in AYUDA


def test_la_ayuda_ya_no_dice_que_no_entiende_lenguaje_natural() -> None:
    """Con el agente andando, ese texto pasó a ser mentira."""
    assert "lenguaje natural" in AYUDA.lower()
    assert "todavía no" not in AYUDA.lower().split("comandos fijos")[0]
