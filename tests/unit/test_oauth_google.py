"""Tests del adaptador OAuth2 contra Google (PB-009).

Se usa `httpx.MockTransport` en vez de un doble a mano, igual que en el cliente
de WhatsApp: deja ejercitar el `AsyncClient` real —con su serialización de
formulario— e inspeccionar la petición que habría salido.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
import structlog

from src.domain.exceptions import AutorizacionFallidaError, ServiceUnavailableError
from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.external.google.oauth import (
    SCOPES,
    URL_CONSENTIMIENTO,
    OAuthGoogle,
    create_autorizador_google,
)

CLIENT_ID = "123-abc.apps.googleusercontent.com"
CLIENT_SECRET = "secreto-de-google"
REDIRECT = "https://lifesync.up.railway.app/oauth/google/callback"
USUARIO = uuid4()

Manejador = Callable[[httpx.Request], httpx.Response]


def _settings(**extra: Any) -> Settings:
    return Settings(
        _env_file=None,
        environment=Environment.TESTING,
        token_encryption_key="clave-fernet-de-prueba",
        google_client_id=CLIENT_ID,
        google_client_secret=CLIENT_SECRET,
        google_redirect_uri=REDIRECT,
        **extra,
    )


def _autorizador(manejador: Manejador) -> tuple[OAuthGoogle, list[httpx.Request]]:
    pedidos: list[httpx.Request] = []

    def interceptar(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        return manejador(pedido)

    http = httpx.AsyncClient(transport=httpx.MockTransport(interceptar))
    return create_autorizador_google(http, _settings()), pedidos


def _respuesta_ok(**extra: Any) -> dict[str, Any]:
    return {
        "access_token": "ya29.access-de-google",
        "refresh_token": "1//refresh-de-google",
        "expires_in": 3599,
        "scope": " ".join(SCOPES),
        "token_type": "Bearer",
        **extra,
    }


def _ahora() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)


# --- La URL de consentimiento -----------------------------------------------


def _parametros_de_la_url() -> dict[str, list[str]]:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json={}))
    url = autorizador.url_de_autorizacion(USUARIO, _ahora())
    assert url.startswith(URL_CONSENTIMIENTO)
    return parse_qs(urlparse(url).query)


def test_la_url_pide_un_refresh_token() -> None:
    """Sin access_type=offline, Google no emite refresh y la conexión dura una hora."""
    parametros = _parametros_de_la_url()

    assert parametros["access_type"] == ["offline"]


def test_la_url_fuerza_el_consentimiento() -> None:
    """Sin prompt=consent, una reconexión nos deja sin refresh token y sin aviso."""
    assert _parametros_de_la_url()["prompt"] == ["consent"]


def test_la_url_lleva_las_credenciales_y_el_destino() -> None:
    parametros = _parametros_de_la_url()

    assert parametros["client_id"] == [CLIENT_ID]
    assert parametros["redirect_uri"] == [REDIRECT]
    assert parametros["response_type"] == ["code"]


def test_se_piden_exactamente_los_permisos_que_se_usan() -> None:
    """Privilegio mínimo, versión PB-032: calendario + tareas + leer y enviar correo.

    El readonly se conserva porque `calendarList` lo necesita; `calendar.events`
    es lo que usan crear/modificar/eliminar; `tasks` es la lista de tareas;
    `gmail.readonly` es buscar y leer; `gmail.send`, sólo enviar (ni modificar
    ni borrar).
    Si alguien suma un scope acá sin sumar la función que lo usa, este test lo
    delata (RF-18).
    """
    scopes = _parametros_de_la_url()["scope"][0].split()

    assert scopes == list(SCOPES)
    assert set(scopes) == {
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/tasks",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    }


def test_la_url_lleva_un_state_verificable() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json={}))
    ahora = _ahora()

    url = autorizador.url_de_autorizacion(USUARIO, ahora)
    state = parse_qs(urlparse(url).query)["state"][0]

    assert autorizador.usuario_del_estado(state, ahora) == USUARIO


def test_el_secreto_no_viaja_en_la_url() -> None:
    """El client_secret va en el POST de atrás, nunca en la URL del navegador."""
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json={}))

    assert CLIENT_SECRET not in autorizador.url_de_autorizacion(USUARIO, _ahora())


# --- El canje del código ----------------------------------------------------


async def test_el_canje_manda_lo_que_google_espera() -> None:
    autorizador, pedidos = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    await autorizador.canjear_codigo("4/codigo-de-un-solo-uso")

    cuerpo = parse_qs(pedidos[0].content.decode())
    assert cuerpo["grant_type"] == ["authorization_code"]
    assert cuerpo["code"] == ["4/codigo-de-un-solo-uso"]
    assert cuerpo["client_secret"] == [CLIENT_SECRET]
    assert cuerpo["redirect_uri"] == [REDIRECT]


async def test_el_canje_devuelve_las_credenciales() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    credenciales = await autorizador.canjear_codigo("4/codigo")

    assert credenciales.access_token == "ya29.access-de-google"
    assert credenciales.refresh_token == "1//refresh-de-google"
    assert credenciales.scopes == SCOPES
    assert credenciales.expira_en is not None


async def test_el_vencimiento_lleva_zona_horaria() -> None:
    """La entidad OAuthToken rechaza fechas naive, y con razón."""
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    credenciales = await autorizador.canjear_codigo("4/codigo")

    assert credenciales.expira_en is not None
    assert credenciales.expira_en.tzinfo is not None


async def test_el_vencimiento_se_adelanta_un_margen() -> None:
    """Un token que vence en dos segundos no debe darse por vigente."""
    autorizador, _ = _autorizador(
        lambda _: httpx.Response(200, json=_respuesta_ok(expires_in=3600))
    )

    credenciales = await autorizador.canjear_codigo("4/codigo")

    assert credenciales.expira_en is not None
    faltan = (credenciales.expira_en - _ahora()).total_seconds()
    assert faltan < 3600


async def test_una_respuesta_sin_refresh_token_no_rompe() -> None:
    """Google no lo repite al renovar: el caso de uso decide qué hacer."""
    sin_refresh = {k: v for k, v in _respuesta_ok().items() if k != "refresh_token"}
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json=sin_refresh))

    credenciales = await autorizador.canjear_codigo("4/codigo")

    assert credenciales.refresh_token is None
    assert credenciales.access_token


# --- El refresco ------------------------------------------------------------


async def test_el_refresco_manda_el_grant_correcto() -> None:
    autorizador, pedidos = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    await autorizador.refrescar("1//refresh-viejo")

    cuerpo = parse_qs(pedidos[0].content.decode())
    assert cuerpo["grant_type"] == ["refresh_token"]
    assert cuerpo["refresh_token"] == ["1//refresh-viejo"]


# --- Errores ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("codigo_http", "error"),
    [(400, "invalid_grant"), (401, "invalid_client"), (500, None)],
    ids=["codigo_usado", "credenciales_malas", "google_caido"],
)
async def test_un_rechazo_de_google_es_autorizacion_fallida(
    codigo_http: int, error: str | None
) -> None:
    cuerpo = {"error": error} if error else {}
    autorizador, _ = _autorizador(lambda _: httpx.Response(codigo_http, json=cuerpo))

    with pytest.raises(AutorizacionFallidaError):
        await autorizador.canjear_codigo("4/codigo")


async def test_un_error_de_red_es_autorizacion_fallida() -> None:
    def caerse(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin conexión", request=pedido)

    autorizador, _ = _autorizador(caerse)

    with pytest.raises(AutorizacionFallidaError):
        await autorizador.canjear_codigo("4/codigo")


async def test_una_respuesta_sin_access_token_es_autorizacion_fallida() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json={"token_type": "Bearer"}))

    with pytest.raises(AutorizacionFallidaError):
        await autorizador.canjear_codigo("4/codigo")


async def test_una_respuesta_sin_json_no_rompe() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(500, text="<html>error</html>"))

    with pytest.raises(AutorizacionFallidaError):
        await autorizador.canjear_codigo("4/codigo")


# --- Privacidad (RF-18) ------------------------------------------------------


async def test_no_se_loguea_ninguna_credencial() -> None:
    """Ni el código, ni el access_token, ni el refresh_token."""
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    with structlog.testing.capture_logs() as eventos:
        await autorizador.canjear_codigo("4/codigo-secreto")

    # Sin esto la aserción negativa de abajo pasaría aunque `capture_logs`
    # no hubiera capturado nada. Ver `test_logging.py`, sección de la trampa.
    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "4/codigo-secreto" not in registrado
    assert "ya29.access-de-google" not in registrado
    assert "1//refresh-de-google" not in registrado


async def test_se_avisa_si_google_no_mando_refresh_token() -> None:
    """Es el sensor de que faltó access_type=offline: hay que poder verlo."""
    sin_refresh = {k: v for k, v in _respuesta_ok().items() if k != "refresh_token"}
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json=sin_refresh))

    with structlog.testing.capture_logs() as eventos:
        await autorizador.canjear_codigo("4/codigo")

    evento = next(e for e in eventos if e["event"] == "google.oauth.credenciales_obtenidas")
    assert evento["trajo_refresh"] is False


async def test_el_error_de_google_si_se_loguea() -> None:
    """Son diagnósticos de la API, no datos de la persona: sin esto se depura a ciegas."""
    autorizador, _ = _autorizador(
        lambda _: httpx.Response(400, json={"error": "invalid_grant", "error_description": "Bad"})
    )

    with structlog.testing.capture_logs() as eventos, pytest.raises(AutorizacionFallidaError):
        await autorizador.canjear_codigo("4/codigo")

    rechazo = next(e for e in eventos if e["event"] == "google.oauth.rechazado")
    assert rechazo["error"] == "invalid_grant"


# --- Construcción ------------------------------------------------------------


def test_sin_configuracion_no_se_puede_construir() -> None:
    http = httpx.AsyncClient()
    incompleto = Settings(
        _env_file=None,
        environment=Environment.TESTING,
        token_encryption_key="clave-fernet-de-prueba",
        google_client_id=CLIENT_ID,
    )

    with pytest.raises(ServiceUnavailableError):
        create_autorizador_google(http, incompleto)
