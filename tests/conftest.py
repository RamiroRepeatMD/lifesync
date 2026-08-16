"""Fixtures compartidas de la suite de tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.persistence.encryption import TokenCipher
from src.interfaces.api.app import create_app
from tests.dobles import FakeSupabaseClient


@pytest.fixture
def settings() -> Settings:
    """Configuración determinística de test: sin leer el `.env` del desarrollador."""
    return Settings(
        _env_file=None,
        environment=Environment.TESTING,
        log_level="WARNING",
        log_json=False,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """Cliente HTTP contra una instancia aislada de la aplicación."""
    with TestClient(create_app(settings)) as cliente:
        yield cliente


@pytest.fixture
def clave_fernet() -> str:
    """Clave de cifrado nueva para cada test."""
    return Fernet.generate_key().decode("utf-8")


@pytest.fixture
def cipher(clave_fernet: str) -> TokenCipher:
    """Cifrador listo para usar."""
    return TokenCipher(clave_fernet)


@pytest.fixture
def supabase_falso() -> FakeSupabaseClient:
    """Cliente de Supabase de mentira, sin red."""
    return FakeSupabaseClient()


@pytest.fixture
def client_con_supabase(
    settings: Settings,
    supabase_falso: FakeSupabaseClient,
    cipher: TokenCipher,
) -> Iterator[TestClient]:
    """App con la persistencia simulada, como si el lifespan la hubiera abierto.

    El estado se pisa *después* de entrar al `TestClient`, porque el lifespan
    corre en `__enter__` y dejaría `supabase = None` en modo degradado.
    """
    app = create_app(settings)
    with TestClient(app) as cliente:
        app.state.supabase = supabase_falso
        app.state.token_cipher = cipher
        yield cliente
