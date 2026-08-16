"""Tests del cliente de envío contra la Graph API (PB-004).

Se usa `httpx.MockTransport` en vez de un doble a mano: deja ejercitar el
`httpx.AsyncClient` real —con sus headers y su serialización— e inspeccionar la
petición que habría salido, que es justamente lo que hay que verificar.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import structlog

from src.domain.exceptions import RepositoryError, ServiceUnavailableError
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp
from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.external.whatsapp.cliente import (
    VERSION_GRAPH,
    ClienteWhatsApp,
    create_whatsapp_client,
)
from tests.payloads_meta import PHONE_NUMBER_ID, TELEFONO_E164, WA_ID

DESTINO = NumeroWhatsApp.desde_wa_id(WA_ID)
TOKEN = "token-de-graph-de-prueba"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        environment=Environment.TESTING,
        whatsapp_token=TOKEN,
        whatsapp_phone_number_id=PHONE_NUMBER_ID,
    )


def _respuesta_ok(wa_id: str = WA_ID) -> dict[str, Any]:
    return {
        "messaging_product": "whatsapp",
        "contacts": [{"input": f"+{wa_id}", "wa_id": wa_id}],
        "messages": [{"id": "wamid.RESPUESTA"}],
    }


Manejador = Callable[[httpx.Request], httpx.Response]


def _cliente_con(manejador: Manejador) -> tuple[ClienteWhatsApp, list[httpx.Request]]:
    """Arma el cliente con un transporte simulado que registra las peticiones."""
    pedidos: list[httpx.Request] = []

    def interceptar(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        return manejador(pedido)

    settings = _settings()
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(interceptar),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    return ClienteWhatsApp(http, settings), pedidos


# --- La petición que sale ---------------------------------------------------


async def test_envia_el_cuerpo_que_espera_meta() -> None:
    cliente, pedidos = _cliente_con(lambda _: httpx.Response(200, json=_respuesta_ok()))

    await cliente.enviar_texto(DESTINO, "hola")

    cuerpo = json.loads(pedidos[0].content)
    assert cuerpo == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": TELEFONO_E164,
        "type": "text",
        "text": {"preview_url": False, "body": "hola"},
    }


@pytest.mark.parametrize("wa_id", ["5491141234567", "14155552671"], ids=["argentino", "usa"])
async def test_el_destino_siempre_lleva_el_mas(wa_id: str) -> None:
    """Sin el `+`, Meta antepone el código de país de nuestro número."""
    cliente, pedidos = _cliente_con(lambda _: httpx.Response(200, json=_respuesta_ok(wa_id)))

    await cliente.enviar_texto(NumeroWhatsApp.desde_wa_id(wa_id), "hola")

    assert json.loads(pedidos[0].content)["to"].startswith("+")


async def test_la_url_lleva_la_version_y_nuestro_numero() -> None:
    cliente, pedidos = _cliente_con(lambda _: httpx.Response(200, json=_respuesta_ok()))

    await cliente.enviar_texto(DESTINO, "hola")

    url = str(pedidos[0].url)
    assert f"/{VERSION_GRAPH}/" in url
    assert PHONE_NUMBER_ID in url
    assert url.startswith("https://graph.facebook.com/")


async def test_manda_el_token_en_el_header() -> None:
    cliente, pedidos = _cliente_con(lambda _: httpx.Response(200, json=_respuesta_ok()))

    await cliente.enviar_texto(DESTINO, "hola")

    assert pedidos[0].headers["authorization"] == f"Bearer {TOKEN}"


# --- Errores ----------------------------------------------------------------


async def test_un_error_de_red_es_servicio_no_disponible() -> None:
    def caerse(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin conexión", request=pedido)

    cliente, _ = _cliente_con(caerse)

    with pytest.raises(ServiceUnavailableError):
        await cliente.enviar_texto(DESTINO, "hola")


@pytest.mark.parametrize(
    ("codigo_http", "codigo_meta"),
    [(400, 131047), (400, 131030), (429, 130429), (500, None)],
    ids=["ventana_24h", "fuera_de_allowed_list", "rate_limit", "error_de_meta"],
)
async def test_un_rechazo_de_meta_es_error_de_repositorio(
    codigo_http: int, codigo_meta: int | None
) -> None:
    cuerpo = {"error": {"code": codigo_meta, "message": "falló"}} if codigo_meta else {}
    cliente, _ = _cliente_con(lambda _: httpx.Response(codigo_http, json=cuerpo))

    with pytest.raises(RepositoryError):
        await cliente.enviar_texto(DESTINO, "hola")


async def test_una_respuesta_sin_json_no_rompe() -> None:
    cliente, _ = _cliente_con(lambda _: httpx.Response(500, text="<html>error</html>"))

    with pytest.raises(RepositoryError):
        await cliente.enviar_texto(DESTINO, "hola")


# --- El sensor del 9 argentino ---------------------------------------------


async def test_avisa_si_meta_normalizo_el_numero() -> None:
    """Es el detector del problema del 9: queremos enterarnos con datos propios."""
    cliente, _ = _cliente_con(
        lambda _: httpx.Response(200, json=_respuesta_ok(wa_id="541141234567"))
    )

    with structlog.testing.capture_logs() as eventos:
        await cliente.enviar_texto(DESTINO, "hola")

    avisos = [e for e in eventos if e["event"] == "whatsapp.numero_normalizado_por_meta"]
    assert len(avisos) == 1
    assert avisos[0]["largo_enviado"] == len(WA_ID)
    assert avisos[0]["largo_devuelto"] == len("541141234567")


async def test_el_aviso_no_incluye_el_numero() -> None:
    """Se loguean los largos, nunca los números: son dato personal (RF-18)."""
    cliente, _ = _cliente_con(
        lambda _: httpx.Response(200, json=_respuesta_ok(wa_id="541141234567"))
    )

    with structlog.testing.capture_logs() as eventos:
        await cliente.enviar_texto(DESTINO, "hola")

    registrado = json.dumps(eventos, default=str)
    assert WA_ID not in registrado
    assert "541141234567" not in registrado


async def test_no_avisa_cuando_meta_devuelve_lo_mismo() -> None:
    cliente, _ = _cliente_con(lambda _: httpx.Response(200, json=_respuesta_ok()))

    with structlog.testing.capture_logs() as eventos:
        await cliente.enviar_texto(DESTINO, "hola")

    assert not [e for e in eventos if e["event"] == "whatsapp.numero_normalizado_por_meta"]


async def test_nunca_se_loguea_el_texto_del_mensaje() -> None:
    cliente, _ = _cliente_con(lambda _: httpx.Response(200, json=_respuesta_ok()))

    with structlog.testing.capture_logs() as eventos:
        await cliente.enviar_texto(DESTINO, "informe medico confidencial")

    assert "informe medico confidencial" not in json.dumps(eventos, default=str)


# --- Construcción -----------------------------------------------------------


def test_sin_credenciales_no_se_puede_construir() -> None:
    with pytest.raises(ServiceUnavailableError):
        create_whatsapp_client(Settings(_env_file=None, environment=Environment.TESTING))


async def test_el_cliente_construido_trae_el_token_y_timeout() -> None:
    http = create_whatsapp_client(_settings())
    try:
        assert http.headers["authorization"] == f"Bearer {TOKEN}"
        assert http.timeout.read is not None
    finally:
        await http.aclose()
