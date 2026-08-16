"""Fixtures compartidas de la suite de tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from src.application.use_cases.procesar_mensaje_entrante import ProcesarMensajeEntrante
from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.persistence.encryption import TokenCipher
from src.interfaces.api.app import create_app
from src.interfaces.api.dependencies import get_procesador_de_mensajes
from tests.dobles import FakeSupabaseClient, MensajeroFalso, RepositorioUsuarioEnMemoria
from tests.payloads_meta import PHONE_NUMBER_ID

# Credenciales de mentira, compartidas por los tests que firman peticiones.
APP_SECRET = "app-secret-de-prueba"
VERIFY_TOKEN = "verify-token-de-prueba"


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
def settings_whatsapp() -> Settings:
    """Configuración de test con las credenciales de WhatsApp completas."""
    return Settings(
        _env_file=None,
        environment=Environment.TESTING,
        log_level="WARNING",
        log_json=False,
        whatsapp_token="token-de-prueba",
        whatsapp_phone_number_id=PHONE_NUMBER_ID,
        whatsapp_verify_token=VERIFY_TOKEN,
        whatsapp_app_secret=APP_SECRET,
    )


@pytest.fixture
def mensajero_falso() -> MensajeroFalso:
    """Doble del puerto de envío por WhatsApp."""
    return MensajeroFalso()


@pytest.fixture
def repositorio_usuarios() -> RepositorioUsuarioEnMemoria:
    """Doble en memoria del repositorio de usuarios."""
    return RepositorioUsuarioEnMemoria()


@pytest.fixture
def client_con_whatsapp(
    settings_whatsapp: Settings,
    repositorio_usuarios: RepositorioUsuarioEnMemoria,
    mensajero_falso: MensajeroFalso,
) -> Iterator[TestClient]:
    """App con el webhook operativo y la persistencia simulada.

    Se sobreescriben las dependencias en vez de pisar `app.state` porque el
    procesador se arma dentro del provider: interceptarlo ahí es lo que permite
    inyectar los dobles sin abrir un cliente HTTP ni una conexión a Supabase.
    """
    app = create_app(settings_whatsapp)
    app.dependency_overrides[get_procesador_de_mensajes] = lambda: ProcesarMensajeEntrante(
        repositorio_usuarios, mensajero_falso
    )
    with TestClient(app) as cliente:
        yield cliente
    app.dependency_overrides.clear()


@pytest.fixture
def client_con_whatsapp_y_supabase(
    settings_whatsapp: Settings,
    supabase_falso: FakeSupabaseClient,
    cipher: TokenCipher,
) -> Iterator[TestClient]:
    """App con ambas dependencias operativas, para el readiness completo."""
    app = create_app(settings_whatsapp)
    with TestClient(app) as cliente:
        app.state.supabase = supabase_falso
        app.state.token_cipher = cipher
        yield cliente


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
