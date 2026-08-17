"""Tests de las herramientas del agente (PB-005)."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import structlog

from src.infrastructure.llm.herramientas import (
    HERRAMIENTAS,
    ZONA_HORARIA,
    fecha_y_hora_actual,
)


def _invocar() -> str:
    resultado: str = fecha_y_hora_actual.invoke({})
    return resultado


def test_devuelve_la_fecha_de_hoy_en_hora_argentina() -> None:
    esperado = datetime.now(UTC).astimezone(ZONA_HORARIA)

    respuesta = _invocar()

    assert str(esperado.day) in respuesta
    assert str(esperado.year) in respuesta
    assert "hora de Argentina" in respuesta


def test_el_dia_y_el_mes_salen_en_espanol() -> None:
    """`strftime("%A")` depende del locale, y en el contenedor es inglés."""
    respuesta = _invocar().lower()

    dias = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
    assert any(dia in respuesta for dia in dias)
    assert not re.search(r"monday|tuesday|january|august", respuesta)


def test_incluye_la_hora() -> None:
    assert re.search(r"\d{2}:\d{2}", _invocar())


def test_el_modelo_ve_un_docstring_que_explica_cuando_usarla() -> None:
    """La descripción es lo único que el modelo lee para decidir si la llama."""
    descripcion = fecha_y_hora_actual.description.lower()

    assert "fecha" in descripcion
    assert "mañana" in descripcion


def test_esta_registrada_para_el_grafo() -> None:
    assert [h.name for h in HERRAMIENTAS] == ["fecha_y_hora_actual"]


def test_se_loguea_la_invocacion() -> None:
    """Sin esto no hay forma de saber si el modelo está usando la herramienta."""
    with structlog.testing.capture_logs() as eventos:
        _invocar()

    invocaciones = [e for e in eventos if e["event"] == "agente.herramienta.invocada"]
    assert len(invocaciones) == 1
    assert invocaciones[0]["herramienta"] == "fecha_y_hora_actual"
