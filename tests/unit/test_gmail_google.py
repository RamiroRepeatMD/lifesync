"""Tests del adaptador de Gmail (PB-033). Sin red: MockTransport.

Los JSON respetan la forma real de la API: `messages.list` devuelve sólo ids,
`messages.get` trae el árbol MIME con los cuerpos en base64url (a veces sin
relleno) y `internalDate` en milisegundos.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest

from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import (
    EntityNotFoundError,
    PermisoInsuficienteError,
    ServiceUnavailableError,
)
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.external.google.gmail import (
    MAX_CARACTERES_DE_CUERPO,
    GmailGoogle,
)
from tests.dobles import AutorizadorFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
Manejador = Callable[[httpx.Request], httpx.Response]
# 30/09/2026 14:32 UTC, en milisegundos, como lo manda Gmail.
INTERNA = str(int(datetime(2026, 9, 30, 14, 32, tzinfo=UTC).timestamp() * 1000))


def _b64(texto: str, charset: str = "utf-8") -> str:
    # Gmail usa base64url y muchas veces sin el relleno "=".
    return base64.urlsafe_b64encode(texto.encode(charset)).decode().rstrip("=")


def _parte(tipo: str, texto: str, charset: str = "utf-8") -> dict[str, Any]:
    return {
        "mimeType": tipo,
        "headers": [{"name": "Content-Type", "value": f'{tipo}; charset="{charset}"'}],
        "body": {"data": _b64(texto, charset)},
    }


def _mensaje(payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
    cabeceras = [
        {"name": "From", "value": "Juan Pérez <juan@ejemplo.com>"},
        {"name": "Subject", "value": "Factura de septiembre"},
    ]
    return {
        "id": "m1",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Te adjunto la factura &amp; el detalle",
        "internalDate": INTERNA,
        "payload": {"headers": cabeceras, **payload},
        **extra,
    }


async def _gmail(pedidos: list[httpx.Request], manejador: Manejador) -> GmailGoogle:
    tokens = RepositorioOAuthTokenEnMemoria()
    await tokens.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=ProveedorOAuth.GOOGLE,
            access_token="access-vigente",
            refresh_token="refresh",
            expira_en=datetime.now(UTC) + timedelta(hours=1),
        )
    )

    def interceptar(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        return manejador(pedido)

    http = httpx.AsyncClient(transport=httpx.MockTransport(interceptar))
    return GmailGoogle(http, ConectarGoogle(tokens, AutorizadorFalso(usuario_fijo=USUARIO)))


def _bandeja(*mensajes: dict[str, Any]) -> Manejador:
    """Contesta la lista de ids y cada mensaje por su id."""
    por_id = {m["id"]: m for m in mensajes}

    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": i} for i in por_id]})
        return httpx.Response(200, json=por_id[pedido.url.path.rsplit("/", 1)[1]])

    return responder


def _abierto(payload: dict[str, Any]) -> Manejador:
    return lambda _: httpx.Response(200, json=_mensaje(payload))


# --- Buscar --------------------------------------------------------------------


async def test_buscar_manda_la_consulta_y_pide_los_metadatos_de_cada_uno() -> None:
    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, _bandeja(_mensaje({}), _mensaje({}, id="m2")))

    correos = await gmail.buscar(USUARIO, "from:juan is:unread", 5)

    lista = parse_qs(urlparse(str(pedidos[0].url)).query)
    assert lista["q"] == ["from:juan is:unread"]
    assert lista["maxResults"] == ["5"]
    detalles = [parse_qs(urlparse(str(p.url)).query) for p in pedidos[1:]]
    assert all(d["format"] == ["metadata"] for d in detalles)
    assert [c.id for c in correos] == ["m1", "m2"]


async def test_un_correo_del_listado_trae_lo_que_hay_que_mostrar() -> None:
    gmail = await _gmail([], _bandeja(_mensaje({})))

    (correo,) = await gmail.buscar(USUARIO, "", 5)

    assert correo.remitente == "Juan Pérez <juan@ejemplo.com>"
    assert correo.asunto == "Factura de septiembre"
    assert correo.fecha == datetime(2026, 9, 30, 14, 32, tzinfo=UTC)
    assert correo.no_leido is True
    assert correo.fragmento == "Te adjunto la factura & el detalle"  # entidades resueltas
    assert correo.cuerpo is None  # el listado no trae cuerpo


async def test_sin_consulta_busca_en_la_bandeja_de_entrada_y_acota_la_cantidad() -> None:
    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, lambda _: httpx.Response(200, json={}))

    assert await gmail.buscar(USUARIO, "  ", 50) == ()

    lista = parse_qs(urlparse(str(pedidos[0].url)).query)
    assert lista["q"] == ["in:inbox"]
    assert lista["maxResults"] == ["10"]
    assert len(pedidos) == 1  # sin ids, no hay nada más que pedir


async def test_un_asunto_codificado_en_rfc_2047_se_decodifica() -> None:
    codificado = "=?UTF-8?B?" + base64.b64encode("Reunión mañana".encode()).decode() + "?="
    mensaje = _mensaje({})
    mensaje["payload"]["headers"][1]["value"] = codificado
    gmail = await _gmail([], _bandeja(mensaje))

    (correo,) = await gmail.buscar(USUARIO, "", 5)

    assert correo.asunto == "Reunión mañana"


# --- Leer: el árbol MIME ---------------------------------------------------------


async def test_con_texto_y_html_gana_el_texto_plano() -> None:
    alternativa = {
        "mimeType": "multipart/alternative",
        "parts": [_parte("text/plain", "Hola José, qué tal"), _parte("text/html", "<b>HTML</b>")],
    }
    gmail = await _gmail([], _abierto(alternativa))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo == "Hola José, qué tal"  # tildes intactas tras base64url


async def test_si_solo_hay_html_se_extrae_el_texto() -> None:
    html = (
        "<html><head><style>p{color:red}</style></head><body>"
        "<p>Hola &amp; chau</p><script>alert(1)</script><div>Segunda línea</div>"
        "</body></html>"
    )
    gmail = await _gmail([], _abierto(_parte("text/html", html)))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo is not None
    assert "Hola & chau" in correo.cuerpo
    assert "Segunda línea" in correo.cuerpo
    assert "<" not in correo.cuerpo
    assert "alert" not in correo.cuerpo and "color" not in correo.cuerpo


async def test_un_mixto_anidado_da_el_cuerpo_y_los_nombres_de_los_adjuntos() -> None:
    mixto = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [_parte("text/plain", "Va la factura"), _parte("text/html", "<p>x</p>")],
            },
            {
                "mimeType": "application/pdf",
                "filename": "factura.pdf",
                "body": {"attachmentId": "a1", "size": 1234},
            },
        ],
    }
    gmail = await _gmail([], _abierto(mixto))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo == "Va la factura"
    assert correo.adjuntos == ("factura.pdf",)


async def test_respeta_el_charset_de_la_parte() -> None:
    gmail = await _gmail([], _abierto(_parte("text/plain", "Año y señal", charset="iso-8859-1")))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo == "Año y señal"


async def test_saca_lo_citado_de_hilos_viejos() -> None:
    texto = "Dale, nos vemos.\n\nEl lunes, Ana escribió:\n> ¿Nos vemos el martes?\n> Avisame"
    gmail = await _gmail([], _abierto(_parte("text/plain", texto)))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo is not None
    assert "Dale, nos vemos." in correo.cuerpo
    assert "martes" not in correo.cuerpo


async def test_un_cuerpo_larguisimo_se_recorta() -> None:
    gmail = await _gmail([], _abierto(_parte("text/plain", "x" * 10_000)))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo is not None
    assert correo.cuerpo.count("x") == MAX_CARACTERES_DE_CUERPO
    assert correo.cuerpo.endswith("[El correo sigue: se recortó.]")


# --- Errores ---------------------------------------------------------------------


async def test_un_correo_que_no_existe_es_error_con_nombre() -> None:
    gmail = await _gmail([], lambda _: httpx.Response(404, json={}))

    with pytest.raises(EntityNotFoundError):
        await gmail.leer(USUARIO, "fantasma")


async def test_api_deshabilitada_no_manda_a_reconectar() -> None:
    """La lección de Tasks: el scope solo no alcanza, hay que habilitar la API."""
    cuerpo = {
        "error": {
            "code": 403,
            "message": "Gmail API has not been used in project X before or it is disabled.",
            "details": [{"reason": "SERVICE_DISABLED"}],
        }
    }
    gmail = await _gmail([], lambda _: httpx.Response(403, json=cuerpo))

    with pytest.raises(ServiceUnavailableError):
        await gmail.buscar(USUARIO, "", 5)


async def test_sin_el_scope_de_correo_es_permiso_insuficiente() -> None:
    """Una cuenta conectada antes de PB-033 no tiene gmail.readonly."""
    gmail = await _gmail([], lambda _: httpx.Response(403, json={}))

    with pytest.raises(PermisoInsuficienteError):
        await gmail.buscar(USUARIO, "", 5)
