"""Tests de las entidades y value objects del dominio."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from src.domain.entities.oauth_token import OAuthToken
from src.domain.entities.usuario import Usuario
from src.domain.exceptions import InvalidValueError
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth

TOKEN_DE_PRUEBA = "ya29.token-secreto-de-ejemplo"
AHORA = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


def _token(**overrides: object) -> OAuthToken:
    """Construye un token válido, con los campos que se quieran cambiar."""
    campos: dict[str, object] = {
        "usuario_id": uuid4(),
        "proveedor": ProveedorOAuth.GOOGLE,
        "access_token": TOKEN_DE_PRUEBA,
    }
    campos.update(overrides)
    return OAuthToken(**campos)  # type: ignore[arg-type]


# --- ProveedorOAuth ---------------------------------------------------------


def test_los_proveedores_coinciden_con_el_check_de_la_migracion() -> None:
    """Si se agrega un proveedor, hay que escribir la migración que amplía el CHECK."""
    assert {proveedor.value for proveedor in ProveedorOAuth} == {"google", "notion"}


# --- Usuario ----------------------------------------------------------------


def test_usuario_valido_se_construye() -> None:
    usuario = Usuario(telefono_whatsapp="+5491122334455", nombre="Ramiro")

    assert usuario.telefono_whatsapp == "+5491122334455"
    assert usuario.id is None


@pytest.mark.parametrize("telefono", ["", "   "], ids=["vacio", "espacios"])
def test_usuario_sin_telefono_es_invalido(telefono: str) -> None:
    with pytest.raises(InvalidValueError):
        Usuario(telefono_whatsapp=telefono)


def test_usuario_es_inmutable() -> None:
    usuario = Usuario(telefono_whatsapp="+5491122334455")

    with pytest.raises(FrozenInstanceError):
        usuario.nombre = "otro"  # type: ignore[misc]


# --- OAuthToken: invariantes ------------------------------------------------


@pytest.mark.parametrize("valor", ["", "   "], ids=["vacio", "espacios"])
def test_token_sin_access_token_es_invalido(valor: str) -> None:
    with pytest.raises(InvalidValueError, match="access_token"):
        _token(access_token=valor)


def test_token_con_fecha_sin_zona_horaria_es_invalido() -> None:
    """Comparar fechas naive haría renovar tokens antes o después de tiempo."""
    with pytest.raises(InvalidValueError, match="zona horaria"):
        _token(expira_en=datetime(2026, 8, 16, 12, 0))  # naive a propósito


def test_token_es_inmutable() -> None:
    token = _token()

    with pytest.raises(FrozenInstanceError):
        token.access_token = "otro"  # type: ignore[misc]


# --- OAuthToken: RF-18, el token no debe filtrarse en representaciones ------


def test_el_repr_no_expone_el_access_token() -> None:
    """Sin repr=False, el token aparecería en cada traceback y cada log."""
    token = _token()

    assert TOKEN_DE_PRUEBA not in repr(token)


def test_el_repr_no_expone_el_refresh_token() -> None:
    token = _token(refresh_token="1//refresh-secreto")

    assert "1//refresh-secreto" not in repr(token)


def test_el_str_no_expone_las_credenciales() -> None:
    token = _token(refresh_token="1//refresh-secreto")

    assert TOKEN_DE_PRUEBA not in str(token)
    assert "1//refresh-secreto" not in str(token)


# --- OAuthToken: comportamiento --------------------------------------------


def test_token_sin_vencimiento_nunca_esta_vencido() -> None:
    """Notion emite credenciales sin expiración explícita."""
    assert _token(expira_en=None).esta_vencido(AHORA) is False


def test_token_con_vencimiento_futuro_no_esta_vencido() -> None:
    assert _token(expira_en=AHORA + timedelta(hours=1)).esta_vencido(AHORA) is False


def test_token_con_vencimiento_pasado_esta_vencido() -> None:
    assert _token(expira_en=AHORA - timedelta(seconds=1)).esta_vencido(AHORA) is True


def test_token_que_vence_justo_ahora_esta_vencido() -> None:
    assert _token(expira_en=AHORA).esta_vencido(AHORA) is True


def test_puede_renovarse_solo_si_hay_refresh_token() -> None:
    assert _token(refresh_token="1//refresh").puede_renovarse() is True
    assert _token(refresh_token=None).puede_renovarse() is False
