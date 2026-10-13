"""Integración de Tareas de punta a punta, sin red (PB-031).

Los tests del adaptador llaman a `TareasGoogle` suelto, y los de las tools usan
un puerto falso: la costura entre los dos —qué le llega a Google cuando el
agente decide algo— no la probaba nadie. Es donde PB-027 encontró cuatro bugs
en Calendar. Acá se arma el camino entero con piezas reales (`AgenteGemini`,
grafo, tools, `TareasGoogle`, `ConectarGoogle` con su refresco perezoso) y
sólo dos cosas son falsas: el modelo, que pide las acciones como las pide
Gemini, y Google Tasks, simulado con estado por HTTP. Se aserta el pedido
EXACTO que recibe Google.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.external.google.tareas import TareasGoogle
from src.infrastructure.llm.agente_gemini import CIERRE_DEL_LOTE, AgenteGemini
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import AutorizadorFalso, ModeloFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
HOY = datetime.now(ZONA_HORARIA).date()
RUTA = "/tasks/v1/lists/@default/tasks"


def _due(dia: date) -> str:
    return f"{dia.isoformat()}T00:00:00.000Z"


class GoogleTasksFalso:
    """Contesta como la API de Google Tasks, con estado, y registra cada pedido.

    Pagina como la real: devuelve como mucho `por_pagina` ítems (o el
    `maxResults` pedido, si es menor) y un `nextPageToken` si quedan más.
    """

    def __init__(self, *tareas: tuple[str, date | None], por_pagina: int = 100) -> None:
        self.tareas: dict[str, dict[str, Any]] = {}
        for titulo, vence in tareas:
            self._agregar({"title": titulo, **({"due": _due(vence)} if vence else {})})
        self.por_pagina = por_pagina
        self.pedidos: list[httpx.Request] = []
        self.rechazo: httpx.Response | None = None

    def _agregar(self, cuerpo: dict[str, Any]) -> dict[str, Any]:
        tarea_id = f"t{len(self.tareas) + 1}"
        item = {"id": tarea_id, "status": "needsAction", **cuerpo}
        self.tareas[tarea_id] = item
        return item

    def __call__(self, pedido: httpx.Request) -> httpx.Response:
        self.pedidos.append(pedido)
        if self.rechazo is not None:
            return self.rechazo
        tarea_id = pedido.url.path.removeprefix(RUTA).strip("/")
        if pedido.method == "GET":
            return self._listar(pedido)
        if pedido.method == "POST":
            return httpx.Response(200, json=self._agregar(json.loads(pedido.content)))
        if tarea_id not in self.tareas:
            return httpx.Response(404, json={"error": {"code": 404, "message": "Not Found"}})
        if pedido.method == "PATCH":
            self.tareas[tarea_id].update(json.loads(pedido.content))
            return httpx.Response(200, json=self.tareas[tarea_id])
        del self.tareas[tarea_id]  # DELETE
        return httpx.Response(204)

    def _listar(self, pedido: httpx.Request) -> httpx.Response:
        params = parse_qs(urlparse(str(pedido.url)).query)
        tamanio = min(int(params.get("maxResults", ["20"])[0]), self.por_pagina)
        desde = int(params.get("pageToken", ["0"])[0])
        pendientes = [t for t in self.tareas.values() if t["status"] != "completed"]
        cuerpo: dict[str, Any] = {"items": pendientes[desde : desde + tamanio]}
        if desde + tamanio < len(pendientes):
            cuerpo["nextPageToken"] = str(desde + tamanio)
        return httpx.Response(200, json=cuerpo)

    def con(self, metodo: str) -> list[httpx.Request]:
        return [p for p in self.pedidos if p.method == metodo]


class Bot:
    """El agente de verdad hasta el HTTP: sólo el modelo y Google son falsos."""

    def __init__(
        self, google: GoogleTasksFalso, *guion: AIMessage, token_vencido: bool = False
    ) -> None:
        self.google = google
        self.autorizador = AutorizadorFalso(usuario_fijo=USUARIO)
        self.tokens = RepositorioOAuthTokenEnMemoria()
        self._token_vencido = token_vencido
        self.modelo = ModeloFalso(guion=list(guion))
        http = httpx.AsyncClient(transport=httpx.MockTransport(google))
        tareas = TareasGoogle(http, ConectarGoogle(self.tokens, self.autorizador))
        self.grafo = construir_grafo(
            self.modelo, construir_herramientas(None, tareas), InMemorySaver()
        )
        self.agente = AgenteGemini(self.grafo)

    async def decir(self, texto: str) -> str:
        if not await self.tokens.listar_por_usuario(USUARIO):
            vence = timedelta(hours=-1) if self._token_vencido else timedelta(hours=1)
            await self.tokens.guardar(
                OAuthToken(
                    usuario_id=USUARIO,
                    proveedor=ProveedorOAuth.GOOGLE,
                    access_token="access-vigente",
                    refresh_token="refresh",
                    expira_en=datetime.now(UTC) + vence,
                )
            )
        return await self.agente.responder(
            ConsultaDelUsuario(conversacion_id=USUARIO, usuario_id=USUARIO, texto=texto)
        )

    async def respuestas_de_tools(self) -> list[str]:
        estado = await self.grafo.aget_state({"configurable": {"thread_id": str(USUARIO)}})
        return [str(m.content) for m in estado.values["messages"] if isinstance(m, ToolMessage)]


def _llamadas(*pedidos: tuple[str, dict[str, Any]]) -> AIMessage:
    """Un mensaje del modelo con todas las llamadas juntas, como hace Gemini."""
    return AIMessage(
        "",
        tool_calls=[
            {"name": nombre, "args": args, "id": f"c{i}"}
            for i, (nombre, args) in enumerate(pedidos)
        ],
    )


def _cuerpo(pedido: httpx.Request) -> dict[str, Any]:
    cuerpo: dict[str, Any] = json.loads(pedido.content)
    return cuerpo


# --- Crear ---------------------------------------------------------------------------


async def test_anotar_con_fecha_postea_titulo_y_medianoche_utc_sin_pausa() -> None:
    viernes = HOY + timedelta(days=3)
    google = GoogleTasksFalso()
    bot = Bot(
        google,
        _llamadas(("crear_tarea", {"titulo": "Comprar pan", "fecha_limite": viernes.isoformat()})),
        AIMessage("Listo, anotada."),
    )

    respuesta = await bot.decir("anotá comprar pan para el viernes")

    assert respuesta == "Listo, anotada."  # directo: no hubo pausa
    [post] = google.con("POST")
    assert post.url.path == RUTA
    assert _cuerpo(post) == {"title": "Comprar pan", "due": _due(viernes)}
    assert post.headers["Authorization"] == "Bearer access-vigente"


async def test_tres_tareas_juntas_una_pregunta_y_tres_posts_con_el_si() -> None:
    google = GoogleTasksFalso()
    bot = Bot(
        google,
        _llamadas(
            ("crear_tarea", {"titulo": "Comprar pan"}),
            ("crear_tarea", {"titulo": "Llamar a mamá"}),
            ("crear_tarea", {"titulo": "Pagar la luz"}),
        ),
        AIMessage("Listo, las tres."),
    )

    pregunta = await bot.decir("agendá estas tareas: comprar pan, llamar a mamá y pagar la luz")

    assert pregunta.endswith(CIERRE_DEL_LOTE)
    for titulo in ("Comprar pan", "Llamar a mamá", "Pagar la luz"):
        assert titulo in pregunta
    assert google.con("POST") == []  # nada llega a Google antes del sí

    await bot.decir("sí")

    titulos = sorted(_cuerpo(p)["title"] for p in google.con("POST"))
    assert titulos == ["Comprar pan", "Llamar a mamá", "Pagar la luz"]  # una vez cada una


async def test_tres_tareas_juntas_con_el_no_no_llega_nada_a_google() -> None:
    google = GoogleTasksFalso()
    bot = Bot(
        google,
        _llamadas(
            ("crear_tarea", {"titulo": "Comprar pan"}),
            ("crear_tarea", {"titulo": "Llamar a mamá"}),
        ),
        AIMessage("Ok, nada."),
    )

    await bot.decir("agendá comprar pan y llamar a mamá")
    await bot.decir("no")

    assert google.con("POST") == []


# --- Completar, posponer, eliminar ----------------------------------------------------


async def test_completar_lista_pendientes_y_patchea_el_id_correcto() -> None:
    google = GoogleTasksFalso(("Pagar la luz", None), ("Llamar al banco", None))
    bot = Bot(google, _llamadas(("completar_tarea", {"titulo": "banco"})), AIMessage("Hecho."))

    assert await bot.decir("ya llamé al banco") == "Hecho."  # directo

    [get] = google.con("GET")
    assert parse_qs(urlparse(str(get.url)).query)["showCompleted"] == ["false"]
    [patch] = google.con("PATCH")
    assert patch.url.path == f"{RUTA}/t2"
    assert _cuerpo(patch) == {"status": "completed"}


async def test_con_dos_coincidencias_no_toca_nada_y_pregunta_cual() -> None:
    google = GoogleTasksFalso(("Llamar a mamá", None), ("Llamar al banco", None))
    bot = Bot(google, _llamadas(("completar_tarea", {"titulo": "llamar"})), AIMessage("¿Cuál?"))

    await bot.decir("ya llamé")

    assert google.con("PATCH") == []
    [respuesta] = await bot.respuestas_de_tools()
    assert "Llamar a mamá" in respuesta and "Llamar al banco" in respuesta


async def test_posponer_patchea_solo_el_due() -> None:
    nueva = HOY + timedelta(days=4)
    google = GoogleTasksFalso(("Pagar la luz", HOY + timedelta(days=1)))
    bot = Bot(
        google,
        _llamadas(("posponer_tarea", {"titulo": "luz", "nueva_fecha": nueva.isoformat()})),
        AIMessage("Listo."),
    )

    await bot.decir("pasá lo de la luz")

    [patch] = google.con("PATCH")
    assert patch.url.path == f"{RUTA}/t1"
    assert _cuerpo(patch) == {"due": _due(nueva)}  # sólo eso: el resto queda como estaba


async def test_eliminar_pide_el_si_y_despues_borra_la_url_exacta() -> None:
    google = GoogleTasksFalso(("Pagar la luz", None), ("Llamar al banco", None))
    bot = Bot(google, _llamadas(("eliminar_tarea", {"titulo": "banco"})), AIMessage("Borrada."))

    pregunta = await bot.decir("borrá lo del banco")

    assert "¿Confirmás?" in pregunta
    assert google.con("DELETE") == []
    await bot.decir("sí")
    [delete] = google.con("DELETE")
    assert delete.url.path == f"{RUTA}/t2"
    assert "t2" not in google.tareas


# --- Credenciales y errores -------------------------------------------------------------


async def test_con_el_token_vencido_se_refresca_y_sale_con_el_nuevo() -> None:
    google = GoogleTasksFalso()
    bot = Bot(
        google,
        _llamadas(("crear_tarea", {"titulo": "Comprar pan"})),
        AIMessage("Listo."),
        token_vencido=True,
    )

    await bot.decir("anotá comprar pan")

    assert bot.autorizador.refresh_usados == ["refresh"]
    [post] = google.con("POST")
    assert post.headers["Authorization"] == "Bearer access-de-prueba"  # el renovado


def _error_403(**error: Any) -> httpx.Response:
    return httpx.Response(403, json={"error": {"code": 403, **error}})


async def _respuesta_ante(rechazo: httpx.Response) -> str:
    google = GoogleTasksFalso(("Pagar la luz", None))
    google.rechazo = rechazo
    bot = Bot(google, _llamadas(("tareas_pendientes", {})), AIMessage("ok"))
    await bot.decir("¿qué tengo pendiente?")
    [respuesta] = await bot.respuestas_de_tools()
    return respuesta


async def test_un_401_manda_a_renovar_el_permiso() -> None:
    respuesta = await _respuesta_ante(httpx.Response(401, json={"error": {"code": 401}}))

    assert "/conectar" in respuesta


async def test_un_403_por_scope_manda_a_conectar() -> None:
    respuesta = await _respuesta_ante(
        _error_403(message="Request had insufficient authentication scopes.")
    )

    assert "/conectar" in respuesta


async def test_la_api_deshabilitada_no_manda_a_conectar() -> None:
    detalle = {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "SERVICE_DISABLED"}
    respuesta = await _respuesta_ante(_error_403(message="API disabled", details=[detalle]))

    assert "/conectar" not in respuesta
    assert "Probá de nuevo" in respuesta


async def test_el_limite_de_uso_no_manda_a_conectar() -> None:
    """Reconectar no arregla un límite de pedidos: es esperar un minuto."""
    motivo = {"domain": "usageLimits", "reason": "userRateLimitExceeded"}
    respuesta = await _respuesta_ante(
        _error_403(message="User Rate Limit Exceeded", errors=[motivo])
    )

    assert "/conectar" not in respuesta
    assert "Probá de nuevo" in respuesta


# --- Más de una página --------------------------------------------------------------------


async def test_una_tarea_de_la_segunda_pagina_tambien_se_encuentra() -> None:
    """Con más de 100 pendientes, las de la página 2 existen igual para la persona."""
    relleno = [(f"Tarea {n:03d}", None) for n in range(149)]
    google = GoogleTasksFalso(*relleno, ("Renovar el pasaporte", None))
    bot = Bot(google, _llamadas(("completar_tarea", {"titulo": "pasaporte"})), AIMessage("Hecho."))

    await bot.decir("ya renové el pasaporte")

    assert len(google.con("GET")) == 2  # siguió el nextPageToken
    [patch] = google.con("PATCH")
    assert patch.url.path == f"{RUTA}/t150"
