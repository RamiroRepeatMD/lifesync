"""Tests del value object del número de WhatsApp (PB-004)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.domain.exceptions import InvalidValueError
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp


@pytest.mark.parametrize(
    "wa_id",
    ["5491141234567", "16505551234", "12345678", "123456789012345"],
    ids=["argentino", "estadounidense", "minimo_8", "maximo_15"],
)
def test_wa_id_valido_se_convierte_a_e164(wa_id: str) -> None:
    assert NumeroWhatsApp.desde_wa_id(wa_id).valor == f"+{wa_id}"


def test_acepta_un_wa_id_que_ya_venga_con_mas() -> None:
    """Meta no lo manda así, pero recibirlo no debe duplicar el signo."""
    assert NumeroWhatsApp.desde_wa_id("+5491141234567").valor == "+5491141234567"


def test_ignora_espacios_alrededor() -> None:
    assert NumeroWhatsApp.desde_wa_id("  5491141234567  ").valor == "+5491141234567"


def test_no_inventa_el_9_argentino() -> None:
    """Es verbatim: no agrega ni quita dígitos.

    Forzar el 9 se apoya en evidencia que Meta nunca confirmó, y sería
    reescribir la identidad del usuario si la premisa fuera falsa.
    """
    assert NumeroWhatsApp.desde_wa_id("541141234567").valor == "+541141234567"


@pytest.mark.parametrize(
    "invalido",
    ["", "   ", "1234567", "1234567890123456", "0491141234567", "549-114-123", "abc"],
    ids=["vacio", "espacios", "corto_7", "largo_16", "empieza_con_cero", "con_guiones", "letras"],
)
def test_wa_id_invalido_es_rechazado(invalido: str) -> None:
    with pytest.raises(InvalidValueError):
        NumeroWhatsApp.desde_wa_id(invalido)


def test_rechaza_digitos_unicode() -> None:
    """`\\d` en Python matchea dígitos árabes: por eso el regex usa [0-9]."""
    with pytest.raises(InvalidValueError):
        NumeroWhatsApp.desde_wa_id("١٢٣٤٥٦٧٨٩")


def test_el_error_no_expone_el_numero() -> None:
    """El mensaje puede terminar en un log o en una respuesta HTTP (RF-18)."""
    with pytest.raises(InvalidValueError) as capturado:
        NumeroWhatsApp.desde_wa_id("0491141234567")

    assert "0491141234567" not in str(capturado.value)


def test_es_inmutable() -> None:
    numero = NumeroWhatsApp.desde_wa_id("5491141234567")

    with pytest.raises(FrozenInstanceError):
        numero.valor = "+1"  # type: ignore[misc]


def test_dos_numeros_iguales_son_iguales() -> None:
    """La igualdad estructural es lo que permite usarlo como clave."""
    uno = NumeroWhatsApp.desde_wa_id("5491141234567")
    otro = NumeroWhatsApp.desde_wa_id("+5491141234567")

    assert uno == otro
    assert len({uno, otro}) == 1


def test_str_devuelve_la_forma_canonica_con_mas() -> None:
    """El `+` es obligatorio: sin él Meta antepone nuestro código de país."""
    assert str(NumeroWhatsApp.desde_wa_id("5491141234567")) == "+5491141234567"
