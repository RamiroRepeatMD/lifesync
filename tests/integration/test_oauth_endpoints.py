"""Tests de integración de los endpoints OAuth2 con Google (PB-009).

Estos endpoints los abre una persona en el navegador, así que devuelven HTML.
Las aserciones van sobre lo que se guardó y sobre lo que se le muestra, nunca
sólo sobre el status code.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.exceptions import AutorizacionFallidaError, InvalidValueError
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.config.settings import Environment, Settings
from src.interfaces.api.app import create_app
from src.interfaces.api.dependencies import get_conectar_google
from tests.dobles import AutorizadorFalso, RepositorioOAuthTokenEnMemoria

INICIAR = "/oauth/google/iniciar"
CALLBACK = "/oauth/google/callback"
USUARIO = uuid4()


@pytest.fixture
def tokens() -> RepositorioOAuthTokenEnMemoria:
    return RepositorioOAuthTokenEnMemoria()


@pytest.fixture
def autorizador() -> AutorizadorFalso:
    return AutorizadorFalso(usuario_fijo=USUARIO)


@pytest.fixture
def client_oauth(
    tokens: RepositorioOAuthTokenEnMemoria, autorizador: AutorizadorFalso
) -> Iterator[TestClient]:
    """App con el flujo de OAuth operativo y sin red."""
    app = create_app(Settings(_env_file=None, environment=Environment.TESTING, log_level="WARNING"))
    app.dependency_overrides[get_conectar_google] = lambda: ConectarGoogle(tokens, autorizador)
    with TestClient(app) as cliente:
        yield cliente
    app.dependency_overrides.clear()


@pytest.fixture
def client_sin_oauth() -> Iterator[TestClient]:
    """App sin configuración de OAuth: el provider real devuelve None."""
    app = create_app(Settings(_env_file=None, environment=Environment.TESTING, log_level="WARNING"))
    with TestClient(app) as cliente:
        yield cliente


# --- /iniciar ----------------------------------------------------------------


def test_iniciar_redirige_a_google(client_oauth: TestClient) -> None:
    respuesta = client_oauth.get(INICIAR, params={"state": "firmado"}, follow_redirects=False)

    assert respuesta.status_code == 307
    assert respuesta.headers["location"].startswith("https://accounts.google.com")


def test_iniciar_sin_state_no_redirige(client_oauth: TestClient) -> None:
    respuesta = client_oauth.get(INICIAR, follow_redirects=False)

    assert respuesta.status_code == 400
    assert "location" not in respuesta.headers


def test_iniciar_con_state_invalido_no_redirige(
    client_oauth: TestClient, autorizador: AutorizadorFalso
) -> None:
    autorizador.estado_invalido = InvalidValueError("falsificado")

    respuesta = client_oauth.get(INICIAR, params={"state": "falso"}, follow_redirects=False)

    assert respuesta.status_code == 400
    assert "location" not in respuesta.headers


# --- /callback: el camino feliz ----------------------------------------------


async def test_el_callback_guarda_las_credenciales(
    client_oauth: TestClient, tokens: RepositorioOAuthTokenEnMemoria
) -> None:
    respuesta = client_oauth.get(CALLBACK, params={"code": "4/codigo", "state": "firmado"})

    assert respuesta.status_code == 200
    assert await tokens.obtener(USUARIO, ProveedorOAuth.GOOGLE) is not None


def test_el_callback_devuelve_html_y_no_json(client_oauth: TestClient) -> None:
    """Del otro lado hay alguien mirando el teléfono, no un cliente de API."""
    respuesta = client_oauth.get(CALLBACK, params={"code": "4/codigo", "state": "firmado"})

    assert respuesta.headers["content-type"].startswith("text/html")
    assert "WhatsApp" in respuesta.text


# --- /callback: lo que puede salir mal ---------------------------------------


async def test_un_state_invalido_no_guarda_nada(
    client_oauth: TestClient,
    tokens: RepositorioOAuthTokenEnMemoria,
    autorizador: AutorizadorFalso,
) -> None:
    """Es el ataque que el `state` firmado previene."""
    autorizador.estado_invalido = InvalidValueError("falsificado")

    respuesta = client_oauth.get(CALLBACK, params={"code": "4/codigo", "state": "falso"})

    assert respuesta.status_code == 400
    assert await tokens.obtener(USUARIO, ProveedorOAuth.GOOGLE) is None


def test_si_la_persona_cancela_se_le_explica(client_oauth: TestClient) -> None:
    """access_denied no es una falla del sistema: se responde 200."""
    respuesta = client_oauth.get(CALLBACK, params={"error": "access_denied"})

    assert respuesta.status_code == 200
    assert "/conectar" in respuesta.text


def test_un_callback_sin_codigo_no_rompe(client_oauth: TestClient) -> None:
    respuesta = client_oauth.get(CALLBACK, params={"state": "firmado"})

    assert respuesta.status_code == 400
    assert "text/html" in respuesta.headers["content-type"]


def test_un_callback_vacio_no_rompe(client_oauth: TestClient) -> None:
    """Sin parámetros opcionales daría un 422 con el detalle de qué falta."""
    respuesta = client_oauth.get(CALLBACK)

    assert respuesta.status_code == 400


def test_si_google_rechaza_el_canje_se_avisa(
    client_oauth: TestClient, autorizador: AutorizadorFalso
) -> None:
    autorizador.fallar_canje = AutorizacionFallidaError()

    respuesta = client_oauth.get(CALLBACK, params={"code": "4/codigo", "state": "firmado"})

    assert respuesta.status_code == 503
    assert "/conectar" in respuesta.text


# --- Sin OAuth configurado ---------------------------------------------------


def test_sin_configuracion_los_endpoints_existen_y_explican(
    client_sin_oauth: TestClient,
) -> None:
    """No es 404: la ruta existe, lo que falta es la configuración."""
    for ruta in (INICIAR, CALLBACK):
        respuesta = client_sin_oauth.get(ruta, params={"state": "x"})
        assert respuesta.status_code == 503
        assert "Google" in respuesta.text


# --- Privacidad (RF-18) ------------------------------------------------------


def test_el_codigo_de_autorizacion_no_queda_en_los_logs(
    client_oauth: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """El `code` es una credencial y viaja en la query string.

    Se usa `capsys` y no `caplog` por la trampa ya documentada en
    `test_logging.py`: `configure_logging` limpia los handlers de la raíz y se
    lleva puesto el de caplog, así que la aserción negativa pasaría sin probar
    nada.
    """
    capsys.readouterr()  # descarta lo emitido durante el arranque

    client_oauth.get(CALLBACK, params={"code": "4/codigo-super-secreto", "state": "firmado"})

    salida = capsys.readouterr().out
    assert "4/codigo-super-secreto" not in salida
