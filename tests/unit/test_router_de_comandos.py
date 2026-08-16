"""Tests del router de comandos (PB-004, base de RF-11).

Es una función pura, así que estos tests no necesitan ningún doble.
"""

from __future__ import annotations

import pytest

from src.application.services.router_de_comandos import (
    AYUDA,
    ESTADO,
    NO_ENTIENDO,
    decidir_respuesta,
)


@pytest.mark.parametrize(
    "texto",
    ["/ayuda", "/AYUDA", "  /Ayuda  ", "/ayuda por favor"],
    ids=["exacto", "mayusculas", "con_espacios", "con_cola"],
)
def test_el_comando_de_ayuda_se_reconoce(texto: str) -> None:
    assert decidir_respuesta(texto) == AYUDA


def test_el_comando_de_estado_se_reconoce() -> None:
    assert decidir_respuesta("/estado") == ESTADO


@pytest.mark.parametrize(
    "texto",
    ["hola", "agendame una reunion el jueves", "", "   ", "ayuda", "/desconocido", "😀"],
    ids=["saludo", "lenguaje_natural", "vacio", "espacios", "sin_barra", "otro_comando", "emoji"],
)
def test_lo_demas_cae_en_el_fallback(texto: str) -> None:
    assert decidir_respuesta(texto) == NO_ENTIENDO


def test_las_respuestas_no_estan_vacias() -> None:
    for respuesta in (AYUDA, ESTADO, NO_ENTIENDO):
        assert respuesta.strip()


def test_el_fallback_orienta_hacia_la_ayuda() -> None:
    """RF-11: el usuario tiene que poder descubrir qué puede hacer."""
    assert "/ayuda" in NO_ENTIENDO


def test_la_ayuda_lista_los_comandos_que_existen() -> None:
    assert "/ayuda" in AYUDA
    assert "/estado" in AYUDA
