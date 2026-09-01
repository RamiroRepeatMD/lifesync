"""Tests de la entidad Evento (PB-015)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from src.domain.entities.evento import SIN_TITULO, Evento
from src.domain.exceptions import InvalidValueError

INICIO = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def test_se_construye_con_lo_minimo() -> None:
    evento = Evento(titulo="Reunión", inicio=INICIO)

    assert evento.titulo == "Reunión"
    assert evento.fin is None
    assert evento.todo_el_dia is False


def test_rechaza_un_inicio_sin_zona_horaria() -> None:
    """Comparar fechas naive mezcla días según dónde corra el proceso."""
    with pytest.raises(InvalidValueError):
        Evento(titulo="Reunión", inicio=datetime(2026, 9, 1, 10, 0))


def test_rechaza_un_fin_sin_zona_horaria() -> None:
    with pytest.raises(InvalidValueError):
        Evento(titulo="Reunión", inicio=INICIO, fin=datetime(2026, 9, 1, 11, 0))


def test_rechaza_terminar_antes_de_empezar() -> None:
    with pytest.raises(InvalidValueError):
        Evento(titulo="Reunión", inicio=INICIO, fin=INICIO - timedelta(hours=1))


@pytest.mark.parametrize("titulo", ["", "   ", "\t"], ids=["vacio", "espacios", "tab"])
def test_un_evento_sin_titulo_igual_se_puede_mostrar(titulo: str) -> None:
    """Google permite eventos sin nombre; una línea vacía sería peor."""
    assert Evento(titulo=titulo, inicio=INICIO).titulo_visible == SIN_TITULO


def test_el_titulo_no_aparece_en_el_repr() -> None:
    """Es dato personal: no puede filtrarse por un traceback o un assert (RF-18)."""
    evento = Evento(titulo="Terapia", inicio=INICIO)

    assert "Terapia" not in repr(evento)


def test_es_inmutable() -> None:
    evento = Evento(titulo="Reunión", inicio=INICIO)

    with pytest.raises(FrozenInstanceError):
        evento.titulo = "Otra"  # type: ignore[misc]
