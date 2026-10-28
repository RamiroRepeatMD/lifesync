"""Tests del adaptador de Notion: leer (PB-014) y escribir (RF-07). Sin red: MockTransport.

Los JSON respetan la referencia (versión 2026-03-11, verificada el 27/10/2026):
el título de una página es la propiedad de tipo `title` y se arma con sus
partes; el de un data source, su `title`; un data source puede no traer `url`, y
las personas vienen sin nombre si la conexión no pide datos de usuarios. Al
escribir: el padre es `page_id` o `data_source_id`, las propiedades van por id,
y un archivo se sube en tres pasos (crear, mandar y adjuntar como bloque).
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

import httpx
import pytest
import structlog

from src.application.ports.paginas import Bloque, Propiedad, TipoDeBloque, TipoDePropiedad
from src.application.use_cases.conectar_notion import ConectarNotion
from src.domain.entities.oauth_token import OAuthToken
from src.domain.entities.pagina import Pagina
from src.domain.exceptions import (
    AutorizacionFallidaError,
    EntityNotFoundError,
    EscrituraEnDudaError,
    NotionNoConectadoError,
    NotionRechazoError,
    NotionSinPermisoError,
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


# --- Lo que el bot no ve (RF-07) -------------------------------------------------------


async def test_lo_que_esta_en_la_papelera_no_aparece_al_buscar() -> None:
    api, _ = await _api(
        [], lambda _: httpx.Response(200, json=_lista(_pagina(in_trash=True), _base()))
    )

    [encontrada] = await api.buscar(USUARIO, "", 5)

    assert encontrada.tipo is TipoDePagina.BASE


async def test_cada_fila_trae_su_id() -> None:
    """Sin el id, el modelo no tiene cómo cambiarle un dato: cada mes tiene su «Luz»."""
    fila = _fila()
    api, _ = await _api([], _base_y_filas(fila))

    [leida] = (await api.consultar(USUARIO, BASE_ID, 5)).filas

    assert leida.id == fila["id"]


# --- Obtener y leer los datos -------------------------------------------------------


async def test_obtener_trae_la_pagina_por_su_id_canonico() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=_pagina()))

    pagina = await api.obtener(USUARIO, PAGINA_ID.replace("-", "").upper())

    assert (pagina.id, pagina.tipo, pagina.en_base) == (PAGINA_ID, TipoDePagina.PAGINA, False)
    [pedido] = pedidos
    assert (pedido.method, pedido.url.path) == ("GET", f"/v1/pages/{PAGINA_ID}")


@pytest.mark.parametrize("status", [404, 400])
async def test_obtener_una_base_la_busca_como_data_source(status: int) -> None:
    """La doc no dice si pedir un data source como página da 404 o 400: valen los dos."""
    pedidos: list[httpx.Request] = []

    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.startswith("/v1/pages/"):
            return httpx.Response(status, json={"object": "error", "code": "x"})
        return httpx.Response(200, json=_base())

    api, _ = await _api(pedidos, responder)

    base = await api.obtener(USUARIO, BASE_ID)

    assert base.tipo is TipoDePagina.BASE
    assert [p.url.path for p in pedidos] == [
        f"/v1/pages/{BASE_ID}",
        f"/v1/data_sources/{BASE_ID}",
    ]


async def test_obtener_lo_que_no_existe() -> None:
    api, _ = await _api([], lambda _: httpx.Response(404, json={"code": "object_not_found"}))

    with pytest.raises(EntityNotFoundError):
        await api.obtener(USUARIO, PAGINA_ID)


@pytest.mark.parametrize("clave", ["in_trash", "archived", "is_archived"])
async def test_lo_que_esta_en_la_papelera_no_esta(clave: str) -> None:
    api, _ = await _api([], lambda _: httpx.Response(200, json=_pagina(**{clave: True})))

    with pytest.raises(EntityNotFoundError):
        await api.obtener(USUARIO, PAGINA_ID)


async def test_una_fila_sabe_que_esta_en_una_base() -> None:
    fila = _pagina(parent={"type": "data_source_id", "data_source_id": BASE_ID})
    api, _ = await _api([], lambda _: httpx.Response(200, json=fila))

    assert (await api.obtener(USUARIO, PAGINA_ID)).en_base


def _fila_con_datos() -> dict[str, Any]:
    return _pagina(
        parent={"type": "data_source_id", "data_source_id": BASE_ID},
        properties={
            "Concepto": {"id": "title", "type": "title", "title": _texto("Luz")},
            "Monto": {"id": "a%3Fb", "type": "number", "number": 1500},
            "Pagado": {"id": "c1", "type": "checkbox", "checkbox": False},
            "Vence": {"id": "d1", "type": "date", "date": {"start": "2026-11-10"}},
            "Estado": {"id": "e1", "type": "status", "status": {"name": "Pendiente"}},
            "Rubro": {"id": "f1", "type": "select", "select": None},
            "Notas": {"id": "g1", "type": "rich_text", "rich_text": _texto("débito")},
            # Las que no se cambian: etiquetas, automáticas y sin id.
            "Etiquetas": {"id": "h1", "type": "multi_select", "multi_select": []},
            "Creada": {"id": "i1", "type": "created_time", "created_time": EDITADA},
            "Rara": {"type": "number", "number": 1},
        },
    )


def _esquema() -> dict[str, Any]:
    opciones = [{"name": "Pendiente"}, {"name": "Pagado"}]
    return _base(
        properties={
            "Estado": {"id": "e1", "type": "status", "status": {"options": opciones}},
            "Rubro": {
                "id": "f1",
                "type": "select",
                "select": {"options": [{"name": "Servicios"}, {"id": "sin-nombre"}]},
            },
            "Monto": {"id": "a%3Fb", "type": "number", "number": {"format": "number"}},
        }
    )


async def test_los_datos_de_una_fila_traen_las_opciones_de_su_base() -> None:
    pedidos: list[httpx.Request] = []

    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.startswith("/v1/data_sources/"):
            return httpx.Response(200, json=_esquema())
        return httpx.Response(200, json=_fila_con_datos())

    api, _ = await _api(pedidos, responder)

    pagina, propiedades = await api.propiedades(USUARIO, PAGINA_ID)

    assert pagina.en_base
    assert [p.url.path for p in pedidos] == [
        f"/v1/pages/{PAGINA_ID}",
        f"/v1/data_sources/{BASE_ID}",
    ]
    datos = {p.nombre: p for p in propiedades}
    assert list(datos) == ["Concepto", "Monto", "Pagado", "Vence", "Estado", "Rubro", "Notas"]
    assert (datos["Concepto"].tipo, datos["Concepto"].valor) == (TipoDePropiedad.TITULO, "Luz")
    assert (datos["Monto"].id, datos["Monto"].valor) == ("a%3Fb", "1500")
    assert datos["Pagado"].valor == "no"
    assert datos["Vence"].valor == "2026-11-10"
    assert datos["Estado"].opciones == ("Pendiente", "Pagado")
    assert (datos["Rubro"].valor, datos["Rubro"].opciones) == ("", ("Servicios",))
    assert datos["Notas"].tipo is TipoDePropiedad.TEXTO


async def test_una_pagina_suelta_solo_tiene_el_titulo() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=_pagina()))

    _, propiedades = await api.propiedades(USUARIO, PAGINA_ID)

    assert [(p.id, p.tipo, p.valor) for p in propiedades] == [
        ("title", TipoDePropiedad.TITULO, "Recetas")
    ]
    assert len(pedidos) == 1  # sin base, no hay esquema que pedir


async def test_los_datos_de_una_base_no_sirven() -> None:
    api, _ = await _api([], lambda _: httpx.Response(200, json=_base()))

    with pytest.raises(EntityNotFoundError):
        await api.propiedades(USUARIO, BASE_ID)


# --- Escribir -------------------------------------------------------------------------

DESTINO_ID = "0a1b2c3d-4e5f-4607-8899-aabbccddeeff"
SUBIDA_ID = "7e6d5c4b-3a29-4180-9f6e-5d4c3b2a1908"


def _rico(texto: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": texto}}]


def _padre(tipo: TipoDePagina = TipoDePagina.PAGINA) -> Pagina:
    identificador = BASE_ID if tipo is TipoDePagina.BASE else PAGINA_ID
    return Pagina(id=identificador, titulo="Padre", tipo=tipo, editada=datetime.now(UTC))


def _acepta(pedido: httpx.Request) -> httpx.Response:
    """Un Notion que acepta todo: las subidas con su id, lo demás con una página."""
    if pedido.url.path.startswith("/v1/file_uploads"):
        return httpx.Response(200, json={"object": "file_upload", "id": SUBIDA_ID})
    return httpx.Response(200, json=_pagina())


async def test_crear_una_pagina_adentro_de_otra() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)
    bloques = (
        Bloque(TipoDeBloque.TITULO_2, "Compras"),
        Bloque(TipoDeBloque.TAREA, "Harina", hecho=True),
        Bloque(TipoDeBloque.PARRAFO, "Para el domingo"),
    )

    creada = await api.crear(USUARIO, _padre(), "Torta", bloques)

    assert creada.id == PAGINA_ID
    [pedido] = pedidos
    assert (pedido.method, pedido.url.path) == ("POST", "/v1/pages")
    assert _cuerpo(pedido) == {
        "parent": {"type": "page_id", "page_id": PAGINA_ID},
        "properties": {"title": {"title": _rico("Torta")}},
        "children": [
            {"object": "block", "type": "heading_2", "heading_2": {"rich_text": _rico("Compras")}},
            {
                "object": "block",
                "type": "to_do",
                "to_do": {"rich_text": _rico("Harina"), "checked": True},
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": _rico("Para el domingo")},
            },
        ],
    }


async def test_crear_en_una_base_es_una_fila_nueva() -> None:
    pedidos: list[httpx.Request] = []
    fila = _pagina(parent={"type": "data_source_id", "data_source_id": BASE_ID})
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=fila))

    creada = await api.crear(USUARIO, _padre(TipoDePagina.BASE), "Gas", ())

    assert creada.en_base
    [pedido] = pedidos
    assert _cuerpo(pedido) == {
        "parent": {"type": "data_source_id", "data_source_id": BASE_ID},
        "properties": {"title": {"title": _rico("Gas")}},
    }


async def test_si_notion_no_devuelve_la_creada_queda_en_duda() -> None:
    api, _ = await _api([], lambda _: httpx.Response(200, json={"object": "page"}))

    with pytest.raises(EscrituraEnDudaError):
        await api.crear(USUARIO, _padre(), "Torta", ())


async def test_agregar_va_al_final_de_la_pagina() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)
    bloques = tuple(
        Bloque(tipo, "x")
        for tipo in (
            TipoDeBloque.TITULO_1,
            TipoDeBloque.TITULO_3,
            TipoDeBloque.VINETA,
            TipoDeBloque.NUMERADO,
            TipoDeBloque.TAREA,
        )
    )

    await api.agregar(USUARIO, PAGINA_ID, bloques)

    [pedido] = pedidos
    assert (pedido.method, pedido.url.path) == ("PATCH", f"/v1/blocks/{PAGINA_ID}/children")
    cuerpo = _cuerpo(pedido)
    assert list(cuerpo) == ["children"]  # sin `position`: siempre al final
    assert [b["type"] for b in cuerpo["children"]] == [
        "heading_1",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
    ]
    assert cuerpo["children"][-1]["to_do"]["checked"] is False


async def test_un_texto_largo_va_en_trozos() -> None:
    """Notion corta cada parte en 2000 caracteres: se mandan de a 1900."""
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)

    await api.agregar(USUARIO, PAGINA_ID, (Bloque(TipoDeBloque.PARRAFO, "a" * 4000),))

    [bloque] = _cuerpo(pedidos[0])["children"]
    partes = bloque["paragraph"]["rich_text"]
    assert [len(p["text"]["content"]) for p in partes] == [1900, 1900, 200]


@pytest.mark.parametrize(
    ("tipo", "valor", "esperado"),
    [
        (TipoDePropiedad.TITULO, "Gas", {"title": _rico("Gas")}),
        (TipoDePropiedad.TEXTO, "Pagar online", {"rich_text": _rico("Pagar online")}),
        (TipoDePropiedad.NUMERO, 1500.0, {"number": 1500}),
        (TipoDePropiedad.NUMERO, 12.5, {"number": 12.5}),
        (TipoDePropiedad.CASILLA, True, {"checkbox": True}),
        (TipoDePropiedad.FECHA, date(2026, 11, 10), {"date": {"start": "2026-11-10"}}),
        (TipoDePropiedad.SELECCION, "Servicios", {"select": {"name": "Servicios"}}),
        (TipoDePropiedad.ESTADO, "Pagado", {"status": {"name": "Pagado"}}),
    ],
)
async def test_cambiar_cada_tipo_de_dato_por_el_id(
    tipo: TipoDePropiedad, valor: str | float | bool | date, esperado: dict[str, Any]
) -> None:
    """Por el id: no cambia si alguien renombra la columna."""
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)

    await api.cambiar(USUARIO, PAGINA_ID, Propiedad(id="a%3Fb", nombre="Dato", tipo=tipo), valor)

    [pedido] = pedidos
    assert (pedido.method, pedido.url.path) == ("PATCH", f"/v1/pages/{PAGINA_ID}")
    assert _cuerpo(pedido) == {"properties": {"a%3Fb": esperado}}


@pytest.mark.parametrize(
    ("tipo", "valor"),
    [
        (TipoDePropiedad.NUMERO, math.nan),
        (TipoDePropiedad.NUMERO, math.inf),
        (TipoDePropiedad.NUMERO, True),
        (TipoDePropiedad.CASILLA, "sí"),
        (TipoDePropiedad.FECHA, "2026-11-10"),
        (TipoDePropiedad.TITULO, 3.0),
    ],
)
async def test_un_valor_que_no_corresponde_no_se_manda(
    tipo: TipoDePropiedad, valor: str | float | bool
) -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)

    with pytest.raises(NotionRechazoError):
        await api.cambiar(USUARIO, PAGINA_ID, Propiedad(id="x", nombre="x", tipo=tipo), valor)

    assert pedidos == []


async def test_mover_adentro_de_otra_pagina() -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)

    await api.mover(USUARIO, PAGINA_ID, DESTINO_ID.replace("-", ""))

    [pedido] = pedidos
    assert (pedido.method, pedido.url.path) == ("POST", f"/v1/pages/{PAGINA_ID}/move")
    assert _cuerpo(pedido) == {"parent": {"type": "page_id", "page_id": DESTINO_ID}}


async def test_a_la_papelera_y_nada_mas() -> None:
    """A la papelera, de donde se recupera: nunca un borrado definitivo."""
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)

    await api.a_la_papelera(USUARIO, PAGINA_ID)

    [pedido] = pedidos
    assert (pedido.method, pedido.url.path) == ("PATCH", f"/v1/pages/{PAGINA_ID}")
    assert _cuerpo(pedido) == {"in_trash": True}


_INVALIDO = "../users"
_TITULO = Propiedad(id="title", nombre="Nombre", tipo=TipoDePropiedad.TITULO)


@pytest.mark.parametrize(
    "operacion",
    [
        lambda api: api.obtener(USUARIO, _INVALIDO),
        lambda api: api.propiedades(USUARIO, _INVALIDO),
        lambda api: api.agregar(USUARIO, _INVALIDO, (Bloque(TipoDeBloque.PARRAFO, "x"),)),
        lambda api: api.cambiar(USUARIO, _INVALIDO, _TITULO, "x"),
        lambda api: api.mover(USUARIO, _INVALIDO, DESTINO_ID),
        lambda api: api.mover(USUARIO, PAGINA_ID, _INVALIDO),
        lambda api: api.a_la_papelera(USUARIO, _INVALIDO),
        lambda api: api.subir(USUARIO, _INVALIDO, "a.pdf", "application/pdf", b"x"),
    ],
)
async def test_un_id_invalido_no_genera_ningun_pedido(
    operacion: Callable[[NotionApi], Awaitable[object]],
) -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)

    with pytest.raises(EntityNotFoundError):
        await operacion(api)

    assert pedidos == []


# --- Subir un archivo ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("tipo_mime", "enviado", "bloque"),
    [
        ("application/pdf", "application/pdf", "pdf"),
        ("image/jpeg", "image/jpeg", "image"),
        ("IMAGE/PNG", "image/png", "image"),
        ("text/plain; charset=utf-8", "text/plain", "file"),
        ("application/vnd.ms-excel", "application/vnd.ms-excel", "file"),
    ],
)
async def test_subir_crea_manda_y_adjunta_al_final(
    tipo_mime: str, enviado: str, bloque: str
) -> None:
    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, _acepta)

    await api.subir(USUARIO, PAGINA_ID, "recibo.pdf", tipo_mime, b"%PDF-1.7 contenido")

    crear, mandar, adjuntar = pedidos
    assert (crear.method, crear.url.path) == ("POST", "/v1/file_uploads")
    assert _cuerpo(crear) == {
        "mode": "single_part",
        "filename": "recibo.pdf",
        "content_type": enviado,
    }
    assert (mandar.method, mandar.url.path) == ("POST", f"/v1/file_uploads/{SUBIDA_ID}/send")
    assert mandar.headers["Content-Type"].startswith("multipart/form-data")
    assert b'name="file"; filename="recibo.pdf"' in mandar.content
    assert b"%PDF-1.7 contenido" in mandar.content
    assert mandar.extensions["timeout"]["read"] == 60.0  # más que una consulta
    assert (adjuntar.method, adjuntar.url.path) == ("PATCH", f"/v1/blocks/{PAGINA_ID}/children")
    assert _cuerpo(adjuntar) == {
        "children": [
            {"type": bloque, bloque: {"type": "file_upload", "file_upload": {"id": SUBIDA_ID}}}
        ]
    }


@pytest.mark.parametrize("id_de_subida", ["../../pages/x", None, 7])
async def test_una_subida_sin_un_id_de_notion_no_sigue(id_de_subida: object) -> None:
    """La URL del segundo paso se arma con el id: nunca con uno que no sea de Notion."""
    pedidos: list[httpx.Request] = []
    respuesta = {"object": "file_upload", "id": id_de_subida}
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=respuesta))

    with pytest.raises(ServiceUnavailableError) as error:
        await api.subir(USUARIO, PAGINA_ID, "recibo.pdf", "application/pdf", b"x")

    assert type(error.value) is ServiceUnavailableError
    assert len(pedidos) == 1


@pytest.mark.parametrize("falla", ["crear", "mandar"])
async def test_un_corte_antes_de_adjuntar_no_deja_nada_en_duda(falla: str) -> None:
    """Una subida que no se adjunta vence sola: la página no cambió."""

    def responder(pedido: httpx.Request) -> httpx.Response:
        paso = "mandar" if pedido.url.path.endswith("/send") else "crear"
        if paso == falla:
            raise httpx.ReadTimeout("sin respuesta", request=pedido)
        return _acepta(pedido)

    api, _ = await _api([], responder)

    with pytest.raises(ServiceUnavailableError) as error:
        await api.subir(USUARIO, PAGINA_ID, "recibo.pdf", "application/pdf", b"x")

    assert type(error.value) is ServiceUnavailableError


async def test_un_corte_al_adjuntar_queda_en_duda() -> None:
    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.startswith("/v1/blocks/"):
            return httpx.Response(502, json={"code": "bad_gateway"})
        return _acepta(pedido)

    api, _ = await _api([], responder)

    with pytest.raises(EscrituraEnDudaError):
        await api.subir(USUARIO, PAGINA_ID, "recibo.pdf", "application/pdf", b"x")


async def test_si_notion_no_acepta_el_archivo_es_un_rechazo() -> None:
    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.endswith("/send"):
            return httpx.Response(400, json={"code": "validation_error"})
        return _acepta(pedido)

    pedidos: list[httpx.Request] = []
    api, _ = await _api(pedidos, responder)

    with pytest.raises(NotionRechazoError) as error:
        await api.subir(USUARIO, PAGINA_ID, "recibo.pdf", "application/pdf", b"x")

    assert type(error.value) is NotionRechazoError
    assert len(pedidos) == 2  # no se adjuntó nada


def _yo(limites: Any) -> dict[str, Any]:
    dueno = {"type": "user", "user": {"name": "Ana", "person": {"email": "ana@ejemplo.com"}}}
    bot: dict[str, Any] = {"owner": dueno, "workspace_name": "Espacio de Ana"}
    if limites is not None:
        bot["workspace_limits"] = limites
    return {"object": "user", "type": "bot", "bot": bot}


async def test_el_limite_de_subida_es_el_del_espacio_y_el_dueno_no_se_loguea() -> None:
    pedidos: list[httpx.Request] = []
    yo = _yo({"max_file_upload_size_in_bytes": 5_242_880})
    api, _ = await _api(pedidos, lambda _: httpx.Response(200, json=yo))

    with structlog.testing.capture_logs() as eventos:
        limite = await api.limite_de_subida(USUARIO)

    assert limite == 5_242_880
    [pedido] = pedidos
    assert (pedido.method, pedido.url.path) == ("GET", "/v1/users/me")
    assert "Ana" not in json.dumps(eventos, default=str)


@pytest.mark.parametrize(
    "limites",
    [
        None,
        {},
        {"max_file_upload_size_in_bytes": "5MB"},
        {"max_file_upload_size_in_bytes": 0},
        {"max_file_upload_size_in_bytes": True},
    ],
)
async def test_sin_un_limite_claro_no_hay_limite(limites: Any) -> None:
    api, _ = await _api([], lambda _: httpx.Response(200, json=_yo(limites)))

    assert await api.limite_de_subida(USUARIO) is None


# --- Los rechazos al escribir --------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "codigo", "esperado"),
    [
        (403, "restricted_resource", NotionSinPermisoError),  # token viejo o página ajena
        (400, "validation_error", NotionRechazoError),
        (409, "conflict_error", ServiceUnavailableError),  # se puede repetir
        (500, "internal_server_error", EscrituraEnDudaError),
        (503, "service_unavailable", EscrituraEnDudaError),
        (404, "object_not_found", EntityNotFoundError),
        (429, "rate_limited", ServiceUnavailableError),
    ],
)
async def test_cada_rechazo_al_escribir_dice_si_se_hizo_algo(
    status: int, codigo: str, esperado: type[Exception]
) -> None:
    cuerpo = {"object": "error", "status": status, "code": codigo, "message": "x"}
    api, _ = await _api([], lambda _: httpx.Response(status, json=cuerpo))

    with pytest.raises(esperado) as error:
        await api.agregar(USUARIO, PAGINA_ID, (Bloque(TipoDeBloque.PARRAFO, "x"),))

    assert type(error.value) is esperado  # la clase exacta: hay herencia entre ellas


async def test_una_respuesta_perdida_al_escribir_queda_en_duda() -> None:
    def cortado(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("sin respuesta", request=pedido)

    api, _ = await _api([], cortado)

    with pytest.raises(EscrituraEnDudaError):
        await api.crear(USUARIO, _padre(), "Torta", ())


async def test_si_no_se_pudo_conectar_seguro_que_no_se_escribio() -> None:
    def caido(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=pedido)

    api, _ = await _api([], caido)

    with pytest.raises(ServiceUnavailableError) as error:
        await api.crear(USUARIO, _padre(), "Torta", ())

    assert type(error.value) is ServiceUnavailableError


async def test_al_leer_una_respuesta_perdida_no_queda_en_duda() -> None:
    def cortado(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("sin respuesta", request=pedido)

    api, _ = await _api([], cortado)

    with pytest.raises(ServiceUnavailableError) as error:
        await api.buscar(USUARIO, "", 5)

    assert type(error.value) is ServiceUnavailableError


async def test_al_escribir_no_se_loguea_lo_escrito() -> None:
    api, _ = await _api([], _acepta)
    reservado = (Bloque(TipoDeBloque.PARRAFO, "diagnóstico reservado"),)

    with structlog.testing.capture_logs() as eventos:
        await api.crear(USUARIO, _padre(), "Biopsia", reservado)
        await api.agregar(USUARIO, PAGINA_ID, reservado)
        await api.cambiar(USUARIO, PAGINA_ID, _TITULO, "Biopsia")
        await api.subir(USUARIO, PAGINA_ID, "biopsia.pdf", "application/pdf", b"diagnostico")

    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "iopsia" not in registrado
    assert "diagn" not in registrado
