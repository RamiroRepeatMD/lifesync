"""Tests del repositorio de tokens contra un cliente de Supabase falso.

Acá vive la evidencia de RF-18: se afirma que lo que el repositorio le entrega
a la base va cifrado, y que el token nunca aparece en un log.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
import structlog
from postgrest import APIError
from supabase import AsyncClient

from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import (
    EncryptionError,
    EntityNotFoundError,
    InvalidValueError,
    RepositoryError,
    ServiceUnavailableError,
)
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.persistence.encryption import TokenCipher
from src.infrastructure.persistence.supabase_oauth_token_repository import (
    TABLA,
    SupabaseOAuthTokenRepository,
)
from tests.dobles import FakeSupabaseClient

TOKEN_PLANO = "ya29.access-token-super-secreto"
REFRESH_PLANO = "1//refresh-token-super-secreto"
USUARIO_ID = UUID("11111111-1111-1111-1111-111111111111")
TOKEN_ID = UUID("22222222-2222-2222-2222-222222222222")


def _token(**overrides: object) -> OAuthToken:
    campos: dict[str, object] = {
        "usuario_id": USUARIO_ID,
        "proveedor": ProveedorOAuth.GOOGLE,
        "access_token": TOKEN_PLANO,
    }
    campos.update(overrides)
    return OAuthToken(**campos)  # type: ignore[arg-type]


def _fila(cipher: TokenCipher, **overrides: Any) -> dict[str, Any]:
    """Fila como la devolvería PostgREST."""
    fila: dict[str, Any] = {
        "id": str(TOKEN_ID),
        "usuario_id": str(USUARIO_ID),
        "proveedor": "google",
        "access_token_cifrado": cipher.cifrar(TOKEN_PLANO),
        "refresh_token_cifrado": None,
        "expira_en": "2026-08-16T15:00:00+00:00",
        "scopes": ["calendar", "gmail"],
        "creado_en": "2026-08-16T12:00:00+00:00",
        "actualizado_en": "2026-08-16T12:00:00+00:00",
    }
    fila.update(overrides)
    return fila


def _repo(supabase_falso: FakeSupabaseClient, cipher: TokenCipher) -> SupabaseOAuthTokenRepository:
    """El doble imita la interfaz que el repositorio usa, no toda la de AsyncClient."""
    return SupabaseOAuthTokenRepository(cast("AsyncClient", supabase_falso), cipher)


# --- RF-18: lo que sale hacia la base va cifrado ----------------------------


async def test_guardar_envia_el_access_token_cifrado(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    """El test central de RF-18: el token en claro nunca llega a PostgreSQL."""
    supabase_falso.respuestas[TABLA] = [_fila(cipher)]

    await _repo(supabase_falso, cipher).guardar(_token())

    payload = supabase_falso.ultima_llamada().payload
    serializado = json.dumps(payload)

    assert TOKEN_PLANO not in serializado
    assert payload["access_token_cifrado"].startswith("gA")
    assert cipher.descifrar(payload["access_token_cifrado"]) == TOKEN_PLANO


async def test_guardar_envia_el_refresh_token_cifrado(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.respuestas[TABLA] = [_fila(cipher)]

    await _repo(supabase_falso, cipher).guardar(_token(refresh_token=REFRESH_PLANO))

    payload = supabase_falso.ultima_llamada().payload
    assert REFRESH_PLANO not in json.dumps(payload)
    assert cipher.descifrar(payload["refresh_token_cifrado"]) == REFRESH_PLANO


async def test_nunca_se_loguea_el_valor_del_token(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    """Guardia permanente contra un `logger.debug(..., fila=fila)` futuro."""
    supabase_falso.respuestas[TABLA] = [_fila(cipher)]

    with structlog.testing.capture_logs() as eventos:
        await _repo(supabase_falso, cipher).guardar(_token(refresh_token=REFRESH_PLANO))

    registrado = json.dumps(eventos, default=str)
    assert TOKEN_PLANO not in registrado
    assert REFRESH_PLANO not in registrado


# --- Contrato del upsert ----------------------------------------------------


async def test_guardar_hace_upsert_por_usuario_y_proveedor(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.respuestas[TABLA] = [_fila(cipher)]

    await _repo(supabase_falso, cipher).guardar(_token())

    llamada = supabase_falso.ultima_llamada()
    assert llamada.operacion == "upsert"
    assert llamada.on_conflict == "usuario_id,proveedor"


async def test_guardar_no_envia_el_id(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    """Mandar el id haría que el upsert reescriba la PK de la fila existente."""
    supabase_falso.respuestas[TABLA] = [_fila(cipher)]

    await _repo(supabase_falso, cipher).guardar(_token(id=uuid4()))

    assert "id" not in supabase_falso.ultima_llamada().payload


async def test_el_payload_solo_lleva_tipos_serializables(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    """httpx serializa con json.dumps: un UUID o un datetime crudos explotarían."""
    supabase_falso.respuestas[TABLA] = [_fila(cipher)]

    await _repo(supabase_falso, cipher).guardar(
        _token(expira_en=datetime(2026, 8, 16, 15, 0, tzinfo=UTC), scopes=("a", "b"))
    )

    json.dumps(supabase_falso.ultima_llamada().payload)  # no debe lanzar


async def test_guardar_sin_fila_devuelta_es_un_error(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.respuestas[TABLA] = []

    with pytest.raises(RepositoryError):
        await _repo(supabase_falso, cipher).guardar(_token())


# --- Lectura: se descifra al volver ----------------------------------------


async def test_obtener_descifra_el_access_token(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.respuestas[TABLA] = [
        _fila(cipher, refresh_token_cifrado=cipher.cifrar(REFRESH_PLANO))
    ]

    token = await _repo(supabase_falso, cipher).obtener(USUARIO_ID, ProveedorOAuth.GOOGLE)

    assert token is not None
    assert token.access_token == TOKEN_PLANO
    assert token.refresh_token == REFRESH_PLANO
    assert token.id == TOKEN_ID
    assert token.scopes == ("calendar", "gmail")
    assert token.expira_en == datetime(2026, 8, 16, 15, 0, tzinfo=UTC)


async def test_obtener_devuelve_none_si_no_hay_fila(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.respuestas[TABLA] = []

    assert await _repo(supabase_falso, cipher).obtener(USUARIO_ID, ProveedorOAuth.GOOGLE) is None


async def test_obtener_filtra_por_usuario_y_proveedor(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.respuestas[TABLA] = []

    await _repo(supabase_falso, cipher).obtener(USUARIO_ID, ProveedorOAuth.NOTION)

    assert supabase_falso.ultima_llamada().filtros == {
        "usuario_id": str(USUARIO_ID),
        "proveedor": "notion",
    }


async def test_listar_por_usuario_mapea_todas_las_filas(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.respuestas[TABLA] = [
        _fila(cipher),
        _fila(cipher, id=str(uuid4()), proveedor="notion"),
    ]

    tokens = await _repo(supabase_falso, cipher).listar_por_usuario(USUARIO_ID)

    assert [token.proveedor for token in tokens] == [
        ProveedorOAuth.GOOGLE,
        ProveedorOAuth.NOTION,
    ]


async def test_una_fila_corrupta_lanza_error_de_cifrado(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.respuestas[TABLA] = [_fila(cipher, access_token_cifrado="gAAAAA-corrupto")]

    with pytest.raises(EncryptionError):
        await _repo(supabase_falso, cipher).obtener(USUARIO_ID, ProveedorOAuth.GOOGLE)


async def test_eliminar_filtra_por_usuario_y_proveedor(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    await _repo(supabase_falso, cipher).eliminar(USUARIO_ID, ProveedorOAuth.GOOGLE)

    llamada = supabase_falso.ultima_llamada()
    assert llamada.operacion == "delete"
    assert llamada.filtros == {"usuario_id": str(USUARIO_ID), "proveedor": "google"}


# --- Traducción de errores --------------------------------------------------


def _api_error(codigo: str) -> APIError:
    return APIError({"code": codigo, "message": "error de prueba", "hint": None, "details": None})


@pytest.mark.parametrize(
    ("codigo", "esperado"),
    [
        ("23505", InvalidValueError),
        ("23503", EntityNotFoundError),
        ("23514", InvalidValueError),
        ("42501", RepositoryError),
        ("", RepositoryError),
    ],
    ids=["unique", "foreign_key", "check", "permisos", "sin_codigo"],
)
async def test_los_errores_de_postgres_se_traducen(
    supabase_falso: FakeSupabaseClient,
    cipher: TokenCipher,
    codigo: str,
    esperado: type[Exception],
) -> None:
    supabase_falso.errores[TABLA] = _api_error(codigo)

    with pytest.raises(esperado):
        await _repo(supabase_falso, cipher).guardar(_token())


async def test_un_error_de_red_se_traduce_a_servicio_no_disponible(
    supabase_falso: FakeSupabaseClient, cipher: TokenCipher
) -> None:
    supabase_falso.errores[TABLA] = httpx.ConnectError("sin conexión")

    with pytest.raises(ServiceUnavailableError):
        await _repo(supabase_falso, cipher).obtener(USUARIO_ID, ProveedorOAuth.GOOGLE)
