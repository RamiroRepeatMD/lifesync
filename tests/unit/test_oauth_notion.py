"""Tests del adaptador OAuth2 contra Notion (PB-014). Sin red: MockTransport.

Los pedidos respetan la referencia de Notion (verificada el 27/10/2026): el
canje y el refresco van en JSON con HTTP Basic, y todo pedido lleva
`Notion-Version`, también los de OAuth.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
import structlog

from src.domain.exceptions import (
    AutorizacionFallidaError,
    InvalidValueError,
    ServiceUnavailableError,
)
from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.external.google.oauth import create_autorizador_google
from src.infrastructure.external.notion.oauth import (
    URL_CONSENTIMIENTO,
    URL_REVOCACION,
    URL_TOKENS,
    OAuthNotion,
    create_autorizador_notion,
)
from src.infrastructure.external.notion.transporte import VERSION_DE_NOTION

CLIENT_ID = "1a2b3c4d-cliente-de-notion"
CLIENT_SECRET = "secret_de_notion"
REDIRECT = "https://lifesync.up.railway.app/oauth/notion/callback"
BASIC = "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
USUARIO = uuid4()

Manejador = Callable[[httpx.Request], httpx.Response]


def _settings(**cambios: Any) -> Settings:
    datos: dict[str, Any] = {
        "token_encryption_key": "clave-fernet-de-prueba",
        "notion_client_id": CLIENT_ID,
        "notion_client_secret": CLIENT_SECRET,
        "notion_redirect_uri": REDIRECT,
        "google_client_id": "123-abc.apps.googleusercontent.com",
        "google_client_secret": "secreto-de-google",
        "google_redirect_uri": "https://lifesync.up.railway.app/oauth/google/callback",
    }
    datos.update(cambios)
    return Settings(_env_file=None, environment=Environment.TESTING, **datos)


def _autorizador(manejador: Manejador) -> tuple[OAuthNotion, list[httpx.Request]]:
    pedidos: list[httpx.Request] = []

    def interceptar(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        return manejador(pedido)

    http = httpx.AsyncClient(transport=httpx.MockTransport(interceptar))
    return create_autorizador_notion(http, _settings()), pedidos


def _respuesta_ok(**cambios: Any) -> dict[str, Any]:
    """La forma de la referencia, con un `owner` que trae el correo de la persona."""
    datos: dict[str, Any] = {
        "access_token": "ntn_access",
        "refresh_token": "ntn_refresh",
        "token_type": "bearer",
        "bot_id": "bot-1",
        "workspace_id": "ws-1",
        "workspace_name": "Espacio de Ramiro",
        "owner": {"type": "user", "user": {"id": "u1", "person": {"email": "ramiro@ejemplo.com"}}},
    }
    datos.update(cambios)
    return datos


def _ahora() -> datetime:
    return datetime.now(UTC)


# --- La URL de consentimiento -----------------------------------------------------


def test_la_url_lleva_lo_que_pide_notion() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json={}))

    url = autorizador.url_de_autorizacion(USUARIO, _ahora())

    assert url.startswith(f"{URL_CONSENTIMIENTO}?")
    parametros = parse_qs(urlparse(url).query)
    assert parametros["client_id"] == [CLIENT_ID]
    assert parametros["redirect_uri"] == [REDIRECT]
    assert parametros["response_type"] == ["code"]
    assert parametros["owner"] == ["user"]  # "Always use user"
    assert "state" in parametros


def test_el_secreto_no_viaja_en_la_url() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json={}))

    assert CLIENT_SECRET not in autorizador.url_de_autorizacion(USUARIO, _ahora())


def test_la_url_lleva_un_state_verificable() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json={}))
    url = autorizador.url_de_autorizacion(USUARIO, _ahora())
    [estado] = parse_qs(urlparse(url).query)["state"]

    assert autorizador.usuario_del_estado(estado, _ahora()) == USUARIO


def test_un_state_de_google_no_vale_en_notion_ni_al_reves() -> None:
    """Misma clave maestra, distinta etiqueta: un enlace no sirve para el otro callback."""
    notion, _ = _autorizador(lambda _: httpx.Response(200, json={}))
    google = create_autorizador_google(httpx.AsyncClient(), _settings())
    [de_google] = parse_qs(urlparse(google.url_de_autorizacion(USUARIO, _ahora())).query)["state"]
    [de_notion] = parse_qs(urlparse(notion.url_de_autorizacion(USUARIO, _ahora())).query)["state"]

    with pytest.raises(InvalidValueError):
        notion.usuario_del_estado(de_google, _ahora())
    with pytest.raises(InvalidValueError):
        google.usuario_del_estado(de_notion, _ahora())


# --- Canje y refresco ---------------------------------------------------------------


async def test_el_canje_manda_lo_que_notion_espera() -> None:
    autorizador, pedidos = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    await autorizador.canjear_codigo("codigo-de-notion")

    [pedido] = pedidos
    assert pedido.method == "POST"
    assert str(pedido.url) == URL_TOKENS
    assert pedido.headers["Authorization"] == BASIC
    assert pedido.headers["Notion-Version"] == VERSION_DE_NOTION
    assert json.loads(pedido.content) == {
        "grant_type": "authorization_code",
        "code": "codigo-de-notion",
        "redirect_uri": REDIRECT,
    }


async def test_el_canje_devuelve_el_par() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    credenciales = await autorizador.canjear_codigo("codigo")

    assert (credenciales.access_token, credenciales.refresh_token) == ("ntn_access", "ntn_refresh")


@pytest.mark.parametrize("refresh", [None, "", "ausente"])
async def test_una_respuesta_sin_refresh_token_no_rompe(refresh: str | None) -> None:
    """La referencia lo tipa "string or null"."""
    datos = _respuesta_ok(refresh_token=refresh)
    if refresh == "ausente":
        del datos["refresh_token"]
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json=datos))

    credenciales = await autorizador.canjear_codigo("codigo")

    assert credenciales.refresh_token is None


async def test_el_refresco_manda_el_grant_correcto() -> None:
    autorizador, pedidos = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    await autorizador.refrescar("ntn_refresh_viejo")

    [pedido] = pedidos
    assert str(pedido.url) == URL_TOKENS
    assert pedido.headers["Authorization"] == BASIC
    assert pedido.headers["Notion-Version"] == VERSION_DE_NOTION
    assert json.loads(pedido.content) == {
        "grant_type": "refresh_token",
        "refresh_token": "ntn_refresh_viejo",
    }


@pytest.mark.parametrize(
    ("status", "cuerpo", "codigo"),
    [
        (400, {"error": "invalid_grant"}, "invalid_grant"),
        (401, {"object": "error", "status": 401, "code": "unauthorized"}, "unauthorized"),
        (500, {"object": "error", "status": 500, "code": "internal_server_error"}, None),
    ],
)
async def test_un_rechazo_es_autorizacion_fallida_y_se_loguea_su_codigo(
    status: int, cuerpo: dict[str, Any], codigo: str | None
) -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(status, json=cuerpo))

    with structlog.testing.capture_logs() as eventos, pytest.raises(AutorizacionFallidaError):
        await autorizador.canjear_codigo("codigo")

    rechazo = next(e for e in eventos if e["event"] == "notion.oauth.rechazado")
    assert rechazo["status_code"] == status
    assert rechazo["codigo"] == (codigo or cuerpo.get("code"))


async def test_un_error_de_red_es_autorizacion_fallida() -> None:
    def caido(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=pedido)

    autorizador, _ = _autorizador(caido)

    with pytest.raises(AutorizacionFallidaError):
        await autorizador.refrescar("ntn_refresh")


@pytest.mark.parametrize(
    "respuesta",
    [
        httpx.Response(200, json={"token_type": "bearer"}),
        httpx.Response(200, text="no soy JSON"),
        httpx.Response(200, json=["una", "lista"]),
    ],
    ids=["sin_access_token", "sin_json", "json_que_no_es_objeto"],
)
async def test_una_respuesta_inutilizable_es_autorizacion_fallida(
    respuesta: httpx.Response,
) -> None:
    autorizador, _ = _autorizador(lambda _: respuesta)

    with pytest.raises(AutorizacionFallidaError):
        await autorizador.canjear_codigo("codigo")


async def test_no_se_loguea_ninguna_credencial_ni_el_correo_del_owner() -> None:
    autorizador, _ = _autorizador(lambda _: httpx.Response(200, json=_respuesta_ok()))

    with structlog.testing.capture_logs() as eventos:
        await autorizador.canjear_codigo("codigo-secreto")
        await autorizador.refrescar("ntn_refresh")

    assert eventos
    registrado = json.dumps(eventos, default=str)
    for secreto in ("codigo-secreto", "ntn_access", "ntn_refresh", CLIENT_SECRET):
        assert secreto not in registrado
    assert "ramiro@ejemplo.com" not in registrado
    assert "Espacio de Ramiro" not in registrado


# --- Revocar ------------------------------------------------------------------------


async def test_revocar_manda_el_token_con_las_credenciales_de_la_app() -> None:
    autorizador, pedidos = _autorizador(lambda _: httpx.Response(200, json={"request_id": "r"}))

    assert await autorizador.revocar("ntn_access") is True

    [pedido] = pedidos
    assert str(pedido.url) == URL_REVOCACION
    assert pedido.headers["Authorization"] == BASIC
    assert pedido.headers["Notion-Version"] == VERSION_DE_NOTION
    assert json.loads(pedido.content) == {"token": "ntn_access"}


async def test_un_token_que_ya_no_valia_cuenta_como_revocado() -> None:
    """El estado final es el mismo: la credencial ya no sirve."""
    autorizador, _ = _autorizador(lambda _: httpx.Response(400, json={"error": "invalid_grant"}))

    assert await autorizador.revocar("ntn_viejo") is True


@pytest.mark.parametrize(
    "respuesta",
    [
        httpx.Response(500, json={"code": "internal_server_error"}),
        httpx.Response(400, json={"code": "validation_error"}),
    ],
    ids=["caido", "otro_rechazo"],
)
async def test_una_revocacion_no_confirmada_es_false(respuesta: httpx.Response) -> None:
    autorizador, _ = _autorizador(lambda _: respuesta)

    assert await autorizador.revocar("ntn_access") is False


async def test_revocar_sin_red_no_lanza() -> None:
    def caido(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=pedido)

    autorizador, _ = _autorizador(caido)

    assert await autorizador.revocar("ntn_access") is False


# --- Construcción ------------------------------------------------------------------


def test_sin_configuracion_no_se_puede_construir() -> None:
    with pytest.raises(ServiceUnavailableError):
        create_autorizador_notion(httpx.AsyncClient(), _settings(notion_client_secret=None))
