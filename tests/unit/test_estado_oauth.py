"""Tests del `state` firmado del flujo OAuth2 (PB-009).

Es el control de seguridad del PB: sin firma, cualquiera podría fabricar un
callback con el id de otra persona y dejar su cuenta de Google enganchada al
usuario de LifeSync que elija. Por eso las aserciones que importan acá son las
**negativas**.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from src.domain.exceptions import InvalidValueError
from src.infrastructure.external.google import estado

CLAVE = estado.derivar_clave("clave-fernet-de-prueba")
OTRA_CLAVE = estado.derivar_clave("otra-clave-distinta")
AHORA = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def test_ida_y_vuelta() -> None:
    usuario_id = uuid4()

    firmado = estado.firmar(usuario_id, CLAVE, AHORA)

    assert estado.verificar(firmado, CLAVE, AHORA) == usuario_id


def test_dos_estados_del_mismo_usuario_son_iguales_en_el_mismo_instante() -> None:
    """No hay aleatoriedad: el state es determinístico y eso está bien.

    La imposibilidad de falsificarlo viene de la clave, no de que sea
    impredecible; agregarle un nonce obligaría a guardarlo en algún lado.
    """
    usuario_id = uuid4()

    assert estado.firmar(usuario_id, CLAVE, AHORA) == estado.firmar(usuario_id, CLAVE, AHORA)


# --- Lo que tiene que rechazar ----------------------------------------------


def test_rechaza_una_firma_alterada() -> None:
    firmado = estado.firmar(uuid4(), CLAVE, AHORA)

    with pytest.raises(InvalidValueError):
        estado.verificar(firmado[:-4] + "AAAA", CLAVE, AHORA)


def test_rechaza_una_carga_alterada() -> None:
    """Cambiar el usuario_id invalida la firma: es el ataque que esto previene."""
    firmado = estado.firmar(uuid4(), CLAVE, AHORA)
    carga, _, firma = firmado.partition(".")
    otro = estado.firmar(uuid4(), CLAVE, AHORA).partition(".")[0]

    with pytest.raises(InvalidValueError):
        estado.verificar(f"{otro}.{firma}", CLAVE, AHORA)
    assert carga != otro


def test_rechaza_una_firma_de_otra_clave() -> None:
    firmado = estado.firmar(uuid4(), OTRA_CLAVE, AHORA)

    with pytest.raises(InvalidValueError):
        estado.verificar(firmado, CLAVE, AHORA)


def test_rechaza_un_estado_vencido() -> None:
    firmado = estado.firmar(uuid4(), CLAVE, AHORA)
    despues = AHORA + timedelta(seconds=estado.VIGENCIA_SEGUNDOS + 1)

    with pytest.raises(InvalidValueError):
        estado.verificar(firmado, CLAVE, despues)


def test_sigue_siendo_valido_justo_antes_de_vencer() -> None:
    usuario_id = uuid4()
    firmado = estado.firmar(usuario_id, CLAVE, AHORA)
    casi = AHORA + timedelta(seconds=estado.VIGENCIA_SEGUNDOS - 1)

    assert estado.verificar(firmado, CLAVE, casi) == usuario_id


@pytest.mark.parametrize(
    "basura",
    ["", ".", "sin-punto", "a.b.c", "!!!.???", "eyJhbGciOiJIUzI1NiJ9.abc"],
    ids=["vacio", "solo_punto", "sin_separador", "tres_partes", "no_base64", "parece_jwt"],
)
def test_rechaza_cualquier_basura(basura: str) -> None:
    """Nunca lanza algo distinto de InvalidValueError: el endpoint cuenta con eso."""
    with pytest.raises(InvalidValueError):
        estado.verificar(basura, CLAVE, AHORA)


# --- Separación de claves ----------------------------------------------------


def test_la_clave_del_estado_no_es_la_de_cifrado() -> None:
    """Reutilizar la clave de cifrado para firmar mezclaría dos propósitos."""
    maestra = "clave-fernet-de-prueba"

    assert estado.derivar_clave(maestra) != maestra.encode("utf-8")


def test_claves_maestras_distintas_derivan_claves_distintas() -> None:
    assert estado.derivar_clave("una") != estado.derivar_clave("otra")


def test_el_estado_no_lleva_el_usuario_en_claro() -> None:
    """Va en base64, no cifrado; lo que lo protege es la firma, no el secreto."""
    usuario_id = uuid4()

    firmado = estado.firmar(usuario_id, CLAVE, AHORA)

    assert str(usuario_id) not in firmado
