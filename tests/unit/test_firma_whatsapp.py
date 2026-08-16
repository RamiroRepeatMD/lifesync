"""Tests de la validación de firma de los webhooks de Meta (RF-18)."""

from __future__ import annotations

import pytest

from src.infrastructure.external.whatsapp.firma import firma_valida, firmar

SECRETO = "app-secret-de-meta"
CUERPO = b'{"object":"whatsapp_business_account","entry":[]}'


def test_una_firma_bien_calculada_es_valida() -> None:
    assert firma_valida(CUERPO, firmar(CUERPO, SECRETO), SECRETO) is True


def test_la_firma_tiene_el_prefijo_esperado() -> None:
    assert firmar(CUERPO, SECRETO).startswith("sha256=")


@pytest.mark.parametrize(
    ("descripcion", "cuerpo", "cabecera"),
    [
        ("cuerpo alterado", CUERPO + b" ", firmar(CUERPO, SECRETO)),
        ("firma de otro secreto", CUERPO, firmar(CUERPO, "otro-secreto")),
        ("sin cabecera", CUERPO, None),
        ("cabecera vacía", CUERPO, ""),
        ("sin el prefijo sha256=", CUERPO, firmar(CUERPO, SECRETO).removeprefix("sha256=")),
        ("cabecera con basura", CUERPO, "sha256=no-es-un-hexdigest"),
        ("intento de downgrade a sha1", CUERPO, "sha1=" + firmar(CUERPO, SECRETO)[7:]),
        ("firma de otro cuerpo", CUERPO, firmar(b"{}", SECRETO)),
    ],
)
def test_firmas_invalidas_se_rechazan(
    descripcion: str, cuerpo: bytes, cabecera: str | None
) -> None:
    assert firma_valida(cuerpo, cabecera, SECRETO) is False, descripcion


def test_el_cuerpo_vacio_tambien_se_firma() -> None:
    """Meta puede mandar un cuerpo vacío; el HMAC sigue estando definido."""
    assert firma_valida(b"", firmar(b"", SECRETO), SECRETO) is True


def test_un_cuerpo_con_emoji_valida_sobre_los_bytes_crudos() -> None:
    """Es el caso que rompe si alguien firma el JSON reserializado."""
    cuerpo = '{"text":{"body":"hola 😀"}}'.encode()

    assert firma_valida(cuerpo, firmar(cuerpo, SECRETO), SECRETO) is True
