"""Tests del adaptador de lectura de Notion (PB-014). Sin red: MockTransport.

Los JSON respetan la referencia (versión 2026-03-11, verificada el 27/10/2026):
el título de una página es la propiedad de tipo `title` y se arma con sus
partes; el de un data source, su `title`; un data source puede no traer `url`, y
las personas vienen sin nombre si la conexión no pide datos de usuarios.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import pytest
import structlog

from src.application.use_cases.conectar_notion import ConectarNotion
from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import (
    AutorizacionFallidaError,
    EntityNotFoundError,
    NotionNoConectadoError,
    ServiceUnavailableError,
)
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.domain.value_objects.tipo_de_pagina import TipoDePagina
from src.infrastructure.external.notion.paginas import (
    MAX_CARACTERES_DE_TEXTO,
    PROPIEDADES_POR_FILA,
    NotionApi,
)
from src.infrastructure.external.notion.transporte import VERSION_DE_NOTION
from tests.dobles import AutorizadorNotionFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
PAGINA_ID = "1f2e3d4c-5b6a-4789-8abc-def012345678"
BASE_ID = "9a8b7c6d-5e4f-4321-9fed-cba987654321"
EDITADA = "2026-10-12T21:30:00.000Z"
Manejador = Callable[[httpx.Request], httpx.Response]


def _pagina(**cambios: Any) -> dict[str, Any]:
    datos: dict[str, Any] = {
        "object": "page",
        "id": PAGINA_ID,
        "url": f"https://www.notion.so/Recetas-{PAGINA_ID.replace('-', '')}",
        "last_edited_time": EDITADA,
        "parent": {"type": "workspace", "workspace": True},
        "properties": {
            # El nombre lo elige quien la creó: se busca por el tipo.
            "Nombre": {"id": "title", "type": "title", "title": _texto("Rece", "tas")},
        },
    }
    datos.update(cambios)
    return datos


def _base(**cambios: Any) -> dict[str, Any]:
    datos: dict[str, Any] = {
        "object": "data_source",
        "id": BASE_ID,
        "title": _texto("Gastos"),
        "last_edited_time": EDITADA,
        "parent": {"type": "database_id", "database_id": "5c4b3a29-1807-4f6e-9d5c-4b3a29180716"},
    }
    datos.update(cambios)
    return datos


def _fila(titulo: str = "Luz", **propiedades: Any) -> dict[str, Any]:
    return {
        "object": "page",
        "id": str(uuid4()),
        "url": "https://www.notion.so/Luz-0a1b2c3d4e5f60718293a4b5c6d7e8f9",
        "last_edited_time": EDITADA,
        "properties": {
            "Concepto": {"id": "title", "type": "title", "title": _texto(titulo)},
            **propiedades,
        },
    }


def _texto(*partes: str) -> list[dict[str, Any]]:
    return [{"type": "text", "plain_text": parte} for parte in partes]


def _lista(*resultados: Any, **extra: Any) -> dict[str, Any]:
    return {"object": "list", "results": list(resultados), "has_more": False, **extra}


async def _api(
    pedidos: list[httpx.Request], manejador: Manejador, *, conectado: bool = True
) -> tuple[NotionApi, AutorizadorNotionFalso]:
    tokens = RepositorioOAuthTokenEnMemoria()
    if conectado:
        await tokens.guardar(
            OAuthToken(
                usuario_id=USUARIO,
                proveedor=ProveedorOAuth.NOTION,
                access_token="access-vigente",
                refresh_token="refresh-vigente",
            )
        )
    autorizador = AutorizadorNotionFalso(usuario_fijo=USUARIO)

    def interceptar(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        return manejador(pedido)

    http = httpx.AsyncClient(transport=httpx.MockTransport(interceptar))
    return NotionApi(http, ConectarNotion(tokens, autorizador)), autorizador


def _cuerpo(pedido: httpx.Request) -> Any:
    return json.loads(pedido.content)


# --- Buscar ------------------------------------------------------------------------


async def test_buscar_por_titulo_manda_la_consulta_y_las_credenciales() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=_lista()))

    await api.buscar(USUARIO, "recetas", 5)

    [pedido] = pedidos
    assert (pedido.method, pedido.url.path) == ("POST", "/v1/search")
    assert pedido.headers["Authorization"] == "Bearer access-vigente"
    assert pedido.headers["Notion-Version"] == VERSION_DE_NOTION
    assert _cuerpo(pedido) == {"query": "recetas", "page_size": 5}


async def test_sin_texto_lista_las_ultimas_editadas() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=_lista()))

    await api.buscar(USUARIO, " \n ", 5)

    assert _cuerpo(pedidos[0]) == {
        "page_size": 5,
        "sort": {"timestamp": "last_edited_time", "direction": "descending"},
    }


@pytest.mark.parametrize(("pedida", "enviada"), [(50, 10), (0, 1), (-3, 1)])
async def test_la_cantidad_se_acota(pedida: int, enviada: int) -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=_lista()))

    await api.buscar(USUARIO, "x", pedida)

    assert _cuerpo(pedidos[0])["page_size"] == enviada


async def test_el_texto_va_en_una_linea_y_acotado() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=_lista()))

    await api.buscar(USUARIO, "  re\ncetas\t de  mamá ", 5)
    await api.buscar(USUARIO, "a" * 500, 5)

    assert _cuerpo(pedidos[0])["query"] == "re cetas de mamá"
    assert _cuerpo(pedidos[1])["query"] == "a" * MAX_CARACTERES_DE_TEXTO


async def test_trae_paginas_y_bases_con_lo_que_hay_que_mostrar() -> None:
    api, _ = await _api([], lambda _: httpx.Response(200, json=_lista(_pagina(), _base())))

    pagina, base = await api.buscar(USUARIO, "", 5)

    assert (pagina.id, pagina.titulo, pagina.tipo) == (PAGINA_ID, "Recetas", TipoDePagina.PAGINA)
    assert pagina.editada == datetime(2026, 10, 12, 21, 30, tzinfo=UTC)
    assert pagina.enlace == f"https://www.notion.so/Recetas-{PAGINA_ID.replace('-', '')}"
    assert (base.titulo, base.tipo) == ("Gastos", TipoDePagina.BASE)
    assert base.enlace is None  # un data source puede venir sin url


async def test_lo_que_no_se_entiende_se_descarta_sin_lanzar() -> None:
    """Una excepción en una tool tumba el turno entero: el parser no lanza."""
    raros = _lista(
        _pagina(id="bueno"),
        "no soy un objeto",
        {"object": "page"},
        {"object": "user", "id": "u1", "last_edited_time": EDITADA},
        _pagina(id="fecha-rota", last_edited_time="ayer"),
        _pagina(id="sin-zona", last_edited_time="2026-10-12T21:30:00"),
        _pagina(id="propiedades-raras", properties=["no", "es", "un", "dict"]),
        _base(id="titulo-raro", title="no es una lista"),
    )
    api, _ = await _api([], lambda _: httpx.Response(200, json=raros))

    encontradas = await api.buscar(USUARIO, "", 10)

    assert [p.id for p in encontradas] == ["bueno", "propiedades-raras", "titulo-raro"]
    assert encontradas[1].titulo == encontradas[2].titulo == "(sin título)"


async def test_un_link_que_no_es_https_no_se_usa() -> None:
    respuesta = _lista(_pagina(url="javascript:alert(1)"), _pagina(id="otra", url=None))
    api, _ = await _api([], lambda _: httpx.Response(200, json=respuesta))

    encontradas = await api.buscar(USUARIO, "", 5)

    assert [p.enlace for p in encontradas] == [None, None]


# --- Leer una página -----------------------------------------------------------------


def _pagina_y_markdown(markdown: str, *, truncada: bool = False) -> Manejador:
    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.endswith("/markdown"):
            return httpx.Response(
                200,
                json={
                    "object": "page_markdown",
                    "id": PAGINA_ID,
                    "markdown": markdown,
                    "truncated": truncada,
                    "unknown_block_ids": [],
                },
            )
        return httpx.Response(200, json=_pagina())

    return responder


async def test_leer_pide_la_pagina_y_despues_su_markdown() -> None:
    """El Markdown no trae el título: sale del pedido de la página."""
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _pagina_y_markdown("# Milanesas\nCon puré."))

    leida = await api.leer(USUARIO, PAGINA_ID)

    pagina, markdown = pedidos
    assert (pagina.method, pagina.url.path) == ("GET", f"/v1/pages/{PAGINA_ID}")
    assert (markdown.method, markdown.url.path) == ("GET", f"/v1/pages/{PAGINA_ID}/markdown")
    assert leida.pagina.titulo == "Recetas"
    assert leida.markdown == "# Milanesas\nCon puré."
    assert leida.recortada is False


async def test_leer_avisa_si_notion_la_recorto() -> None:
    api, _ = await _api([], _pagina_y_markdown("…", truncada=True))

    assert (await api.leer(USUARIO, PAGINA_ID)).recortada is True


async def test_el_id_se_lleva_a_su_forma_canonica() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _pagina_y_markdown(""))

    await api.leer(USUARIO, PAGINA_ID.replace("-", "").upper())

    assert pedidos[0].url.path == f"/v1/pages/{PAGINA_ID}"


@pytest.mark.parametrize(
    "identificador",
    [
        "../users",
        f"{PAGINA_ID}/markdown",
        f"{{{PAGINA_ID}}}",
        f"urn:uuid:{PAGINA_ID}",
        "0x" + "a" * 32,
        f"{PAGINA_ID}\n",
        "1f2e3d4c5b6a-4789-8abc-def0-12345678",  # guiones en otro lado
        "",
    ],
    ids=["path", "subrecurso", "llaves", "urn", "0x", "salto", "guiones", "vacio"],
)
async def test_un_id_que_no_es_de_notion_no_genera_pedido(identificador: str) -> None:
    """`uuid.UUID` aceptaría casi todos: por eso la validación es una regex estricta."""
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _pagina_y_markdown(""))

    with pytest.raises(EntityNotFoundError):
        await api.leer(USUARIO, identificador)
    with pytest.raises(EntityNotFoundError):
        await api.consultar(USUARIO, identificador, 10)

    assert pedidos == []


async def test_leer_una_base_como_pagina_no_sirve() -> None:
    api, _ = await _api([], lambda _: httpx.Response(200, json=_base(id=PAGINA_ID)))

    with pytest.raises(EntityNotFoundError):
        await api.leer(USUARIO, PAGINA_ID)


# --- Consultar una base --------------------------------------------------------------


def _base_y_filas(*filas: dict[str, Any], hay_mas: bool = False) -> Manejador:
    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.endswith("/query"):
            return httpx.Response(200, json=_lista(*filas, has_more=hay_mas))
        return httpx.Response(200, json=_base())

    return responder


async def test_consultar_pide_la_base_y_sus_filas_mas_recientes() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _base_y_filas(_fila()))

    consultada = await api.consultar(USUARIO, BASE_ID, 10)

    base, consulta = pedidos
    assert (base.method, base.url.path) == ("GET", f"/v1/data_sources/{BASE_ID}")
    assert (consulta.method, consulta.url.path) == ("POST", f"/v1/data_sources/{BASE_ID}/query")
    assert _cuerpo(consulta) == {
        "page_size": 10,
        "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}],
    }
    assert consultada.base.titulo == "Gastos"
    assert [f.titulo for f in consultada.filas] == ["Luz"]


async def test_las_filas_se_acotan_a_veinte() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _base_y_filas())

    await api.consultar(USUARIO, BASE_ID, 500)

    assert _cuerpo(pedidos[1])["page_size"] == 20


async def test_las_propiedades_pasan_a_texto() -> None:
    fila = _fila(
        Monto={"type": "number", "number": 12345.0},
        Estado={"type": "status", "status": {"name": "Pagado"}},
        Etiquetas={"type": "multi_select", "multi_select": [{"name": "casa"}, {"name": "fijo"}]},
        Vence={"type": "date", "date": {"start": "2026-10-20", "end": "2026-10-25"}},
        Pagado={"type": "checkbox", "checkbox": True},
        Nota={"type": "rich_text", "rich_text": _texto("Débito ", "automático")},
    )
    api, _ = await _api([], _base_y_filas(fila))

    [leida] = (await api.consultar(USUARIO, BASE_ID, 10)).filas

    assert leida.propiedades == (
        ("Monto", "12345"),
        ("Estado", "Pagado"),
        ("Etiquetas", "casa, fijo"),
        ("Vence", "2026-10-20 → 2026-10-25"),
        ("Pagado", "sí"),
        ("Nota", "Débito automático"),
    )


@pytest.mark.parametrize(
    ("propiedad", "esperado"),
    [
        ({"type": "select", "select": {"name": "Alta"}}, "Alta"),
        ({"type": "select", "select": None}, ""),
        ({"type": "number", "number": 3.5}, "3.5"),
        ({"type": "number", "number": None}, ""),
        ({"type": "url", "url": "https://ejemplo.com"}, "https://ejemplo.com"),
        ({"type": "email", "email": "ana@ejemplo.com"}, "ana@ejemplo.com"),
        ({"type": "phone_number", "phone_number": "+54 11 5555-5555"}, "+54 11 5555-5555"),
        ({"type": "date", "date": {"start": "2026-10-20", "end": None}}, "2026-10-20"),
        ({"type": "checkbox", "checkbox": False}, "no"),
        ({"type": "people", "people": [{"object": "user", "id": "u1"}]}, ""),
        ({"type": "people", "people": [{"object": "user", "name": "Ana"}]}, "Ana"),
        ({"type": "formula", "formula": {"type": "number", "number": 16}}, "16"),
        ({"type": "formula", "formula": {"type": "string", "string": "ok"}}, "ok"),
        ({"type": "formula", "formula": {"type": "boolean", "boolean": True}}, "sí"),
        ({"type": "formula", "formula": {"type": "formula", "formula": {}}}, ""),
        ({"type": "relation", "relation": [{"id": "x"}], "has_more": False}, ""),
        ({"type": "files", "files": [{"name": "factura.pdf"}]}, ""),
        ({"type": "desconocido"}, ""),
    ],
    ids=[
        "select",
        "select_vacio",
        "numero_decimal",
        "numero_vacio",
        "url",
        "email",
        "telefono",
        "fecha_sin_fin",
        "casilla_no",
        "persona_sin_nombre",
        "persona_con_nombre",
        "formula_numero",
        "formula_texto",
        "formula_booleana",
        "formula_anidada",
        "relacion",
        "archivos",
        "desconocido",
    ],
)
async def test_cada_tipo_de_propiedad(propiedad: dict[str, Any], esperado: str) -> None:
    """Lo que no se sabe mostrar (o viene vacío) no aparece."""
    api, _ = await _api([], _base_y_filas(_fila(Dato=propiedad)))

    [fila] = (await api.consultar(USUARIO, BASE_ID, 10)).filas

    assert fila.propiedades == ((("Dato", esperado),) if esperado else ())


async def test_como_mucho_seis_propiedades_por_fila() -> None:
    muchas: dict[str, Any] = {f"P{n}": {"type": "number", "number": n} for n in range(10)}
    api, _ = await _api([], _base_y_filas(_fila(**muchas)))

    [fila] = (await api.consultar(USUARIO, BASE_ID, 10)).filas

    assert len(fila.propiedades) == PROPIEDADES_POR_FILA


async def test_dice_si_quedaron_filas_sin_traer() -> None:
    api, _ = await _api([], _base_y_filas(_fila(), hay_mas=True))

    assert (await api.consultar(USUARIO, BASE_ID, 1)).hay_mas is True


async def test_consultar_una_pagina_como_base_no_sirve() -> None:
    api, _ = await _api([], lambda _: httpx.Response(200, json=_pagina(id=BASE_ID)))

    with pytest.raises(EntityNotFoundError):
        await api.consultar(USUARIO, BASE_ID, 10)


# --- El token y los rechazos ------------------------------------------------------


async def test_sin_conexion_no_le_pide_nada_a_notion() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=_lista()), conectado=False)

    with pytest.raises(NotionNoConectadoError):
        await api.buscar(USUARIO, "", 5)

    assert pedidos == []


async def test_un_401_renueva_una_vez_y_reintenta_con_el_token_nuevo() -> None:
    """Notion no informa vencimiento: el 401 es la única señal."""
    pedidos: list[httpx.Request] = []

    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.headers["Authorization"] == "Bearer access-vigente":
            return httpx.Response(401, json={"object": "error", "code": "unauthorized"})
        return httpx.Response(200, json=_lista(_pagina()))

    api, autorizador = await _api(pedidos, responder)

    [pagina] = await api.buscar(USUARIO, "", 5)

    assert pagina.titulo == "Recetas"
    assert autorizador.refresh_usados == ["refresh-vigente"]
    assert [p.headers["Authorization"] for p in pedidos] == [
        "Bearer access-vigente",
        "Bearer notion-access-2",
    ]


async def test_un_401_despues_de_renovar_es_autorizacion_fallida() -> None:
    pedidos: list[httpx.Request] = []
    api, autorizador = await _api(
        pedidos, lambda _: httpx.Response(401, json={"code": "unauthorized"})
    )

    with pytest.raises(AutorizacionFallidaError):
        await api.buscar(USUARIO, "", 5)

    assert len(pedidos) == 2  # el original y un solo reintento
    assert autorizador.refresh_usados == ["refresh-vigente"]


@pytest.mark.parametrize(
    ("status", "codigo", "esperado"),
    [
        (404, "object_not_found", EntityNotFoundError),
        (403, "restricted_resource", ServiceUnavailableError),  # nuestra configuración
        (429, "rate_limited", ServiceUnavailableError),
        (400, "validation_error", ServiceUnavailableError),
        (500, "internal_server_error", ServiceUnavailableError),
    ],
)
async def test_cada_rechazo_tiene_su_remedio(
    status: int, codigo: str, esperado: type[Exception]
) -> None:
    """Un 403 no manda a /conectar: reconectar no le suma capacidades a la conexión."""
    cuerpo = {"object": "error", "status": status, "code": codigo, "message": "x"}
    api, _ = await _api([], lambda _: httpx.Response(status, json=cuerpo))

    with pytest.raises(esperado):
        await api.leer(USUARIO, PAGINA_ID)


async def test_sin_red_es_servicio_no_disponible() -> None:
    def caido(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=pedido)

    api, _ = await _api([], caido)

    with pytest.raises(ServiceUnavailableError):
        await api.buscar(USUARIO, "", 5)


# --- Privacidad (RF-18) -------------------------------------------------------------


async def test_ni_los_titulos_ni_el_contenido_se_loguean() -> None:
    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.endswith("/markdown"):
            return httpx.Response(200, json={"markdown": "diagnóstico reservado"})
        if pedido.url.path == "/v1/search":
            return httpx.Response(200, json=_lista(_pagina()))
        titulo = _texto("Resultados de biopsia")
        return httpx.Response(
            200, json=_pagina(properties={"N": {"type": "title", "title": titulo}})
        )

    api, _ = await _api([], responder)

    with structlog.testing.capture_logs() as eventos:
        await api.buscar(USUARIO, "biopsia", 5)
        await api.leer(USUARIO, PAGINA_ID)

    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "biopsia" not in registrado
    assert "diagnóstico" not in registrado


async def test_el_mensaje_de_error_de_notion_no_se_loguea() -> None:
    """Puede repetir lo que se pidió; alcanza con el código."""
    cuerpo = {"object": "error", "status": 400, "code": "validation_error", "message": "biopsia"}
    api, _ = await _api([], lambda _: httpx.Response(400, json=cuerpo))

    with structlog.testing.capture_logs() as eventos, pytest.raises(ServiceUnavailableError):
        await api.buscar(USUARIO, "biopsia", 5)

    rechazo = next(e for e in eventos if e["event"] == "notion.rechazado")
    assert rechazo["codigo"] == "validation_error"
    assert "biopsia" not in json.dumps(eventos, default=str)
