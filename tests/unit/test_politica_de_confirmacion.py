"""La política de RF-08 (segunda versión): qué necesita el sí de la persona."""

from __future__ import annotations

import pytest

from src.domain.services.politica_de_confirmacion import (
    TipoDeEscritura,
    requiere_confirmacion,
)

T = TipoDeEscritura


@pytest.mark.parametrize("tipo", [T.CREAR, T.MODIFICAR, T.COMPLETAR, T.POSPONER])
def test_una_accion_reversible_sale_directo(tipo: TipoDeEscritura) -> None:
    assert requiere_confirmacion([tipo]) is False


@pytest.mark.parametrize("tipo", [T.ELIMINAR, T.ENVIAR])
def test_una_accion_irreversible_confirma(tipo: TipoDeEscritura) -> None:
    assert requiere_confirmacion([tipo]) is True


def test_dos_acciones_juntas_confirman_aunque_sean_reversibles() -> None:
    assert requiere_confirmacion([T.CREAR, T.CREAR]) is True


def test_con_texto_de_terceros_a_la_vista_todo_confirma() -> None:
    assert requiere_confirmacion([T.CREAR], hay_texto_de_terceros=True) is True


def test_sin_escrituras_no_hay_nada_que_confirmar() -> None:
    assert requiere_confirmacion([], hay_texto_de_terceros=True) is False
