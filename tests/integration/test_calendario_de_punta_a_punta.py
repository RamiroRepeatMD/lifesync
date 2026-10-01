"""Integración de Calendar de punta a punta, sin red (PB-027).

Los tests de las tools usan un calendario falso y los del adaptador llaman al
adaptador suelto: la costura entre los dos —qué le llega a Google cuando una
tool decide algo— no la probaba nadie. Acá se arma el camino entero con piezas
reales (grafo, tools, `CalendarioGoogle`) y sólo el HTTP es simulado, para
asertar **el cuerpo exacto que recibe Google**.

Así se encontró el bug de los eventos de día completo: la tool hacía bien el
merge, el adaptador serializaba bien lo que recibía, y entre los dos le
mandaban a Google un rango vacío.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.external.google.calendario import CalendarioGoogle
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas, fecha_en_palabras
from tests.dobles import AutorizadorFalso, ModeloFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
HOY = datetime.now(ZONA_HORARIA).date()


def _dia(dias_desde_hoy: int) -> date:
    return HOY + timedelta(days=dias_desde_hoy)


def _en_palabras(dia: date) -> str:
    return fecha_en_palabras(datetime(dia.year, dia.month, dia.day, tzinfo=UTC))


class GoogleFalso:
    """Contesta como la API de Calendar y registra cada pedido."""

    def __init__(self, eventos_del_principal: list[dict[str, Any]] | None = None) -> None:
        self.eventos = eventos_del_principal or []
        self.pedidos: list[httpx.Request] = []

    def __call__(self, pedido: httpx.Request) -> httpx.Response:
        self.pedidos.append(pedido)
        if "calendarList" in pedido.url.path:
            lista = [{"id": "yo@gmail.com", "summary": "Principal", "selected": True}]
            return httpx.Response(200, json={"items": lista})
        if pedido.method == "GET":
            return httpx.Response(200, json={"items": self.eventos})
        if pedido.method in ("POST", "PATCH"):
            return httpx.Response(200, json={"id": "id-google", **json.loads(pedido.content)})
        return httpx.Response(204)

    def cuerpo_de(self, metodo: str) -> dict[str, Any]:
        (pedido,) = [p for p in self.pedidos if p.method == metodo]
        cuerpo: dict[str, Any] = json.loads(pedido.content)
        return cuerpo


async def _calendario(google: GoogleFalso) -> CalendarioGoogle:
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
    http = httpx.AsyncClient(transport=httpx.MockTransport(google))
    return CalendarioGoogle(http, ConectarGoogle(tokens, AutorizadorFalso(usuario_fijo=USUARIO)))


async def _pedir_y_aprobar(google: GoogleFalso, nombre: str, **args: Any) -> dict[str, Any]:
    """Una tool de escritura, de punta a punta: pedido → interrupt → sí."""
    pedido = AIMessage("", tool_calls=[{"name": nombre, "args": args, "id": "c1"}])
    modelo = ModeloFalso(guion=[pedido, AIMessage("listo")])
    grafo = construir_grafo(
        modelo, construir_herramientas(await _calendario(google)), InMemorySaver()
    )
    configuracion = {"configurable": {"thread_id": "hilo"}}
    contexto = ContextoDeAgente(usuario_id=USUARIO)

    estado: dict[str, Any] = await grafo.ainvoke(
        {"messages": [HumanMessage("pedido")]}, config=configuracion, context=contexto
    )
    if "__interrupt__" in estado:
        await grafo.ainvoke(
            Command(resume={"aprobado": True}), config=configuracion, context=contexto
        )
    return estado


def _dia_completo(titulo: str, desde: date, dias: int) -> dict[str, Any]:
    return {
        "id": "id-dia-completo",
        "summary": titulo,
        "start": {"date": desde.isoformat()},
        # Exclusivo, como lo manda Google: un evento del 5 termina el 6.
        "end": {"date": (desde + timedelta(days=dias)).isoformat()},
    }


# --- Crear ---------------------------------------------------------------------


async def test_crear_a_las_2330_manda_el_huso_argentino_y_termina_al_dia_siguiente() -> None:
    google = GoogleFalso()
    manana = _dia(1)

    await _pedir_y_aprobar(
        google,
        "crear_evento_en_calendario",
        titulo="Cena",
        fecha=manana.isoformat(),
        hora_inicio="23:30",
    )

    cuerpo = google.cuerpo_de("POST")
    assert cuerpo["start"] == {"dateTime": f"{manana.isoformat()}T23:30:00-03:00"}
    assert cuerpo["end"] == {"dateTime": f"{_dia(2).isoformat()}T00:30:00-03:00"}


# --- Día completo: el bug de PB-027 ----------------------------------------------


async def test_renombrar_un_dia_completo_manda_el_fin_exclusivo() -> None:
    """Hasta PB-027 se mandaba `end = start`: para Google, un rango vacío."""
    cumple = _dia(5)
    google = GoogleFalso([_dia_completo("Cumple de mamá", cumple, dias=1)])

    await _pedir_y_aprobar(
        google,
        "modificar_evento_del_calendario",
        fecha=cumple.isoformat(),
        titulo="cumple",
        nuevo_titulo="Cumple de mamá (llevar torta)",
    )

    cuerpo = google.cuerpo_de("PATCH")
    assert cuerpo["start"] == {"date": cumple.isoformat()}
    assert cuerpo["end"] == {"date": (cumple + timedelta(days=1)).isoformat()}


async def test_mover_uno_de_tres_dias_corre_las_dos_puntas() -> None:
    inicio = _dia(5)
    google = GoogleFalso([_dia_completo("Vacaciones", inicio, dias=3)])
    nuevo_inicio = _dia(12)

    await _pedir_y_aprobar(
        google,
        "modificar_evento_del_calendario",
        fecha=inicio.isoformat(),
        titulo="vacaciones",
        nueva_fecha=nuevo_inicio.isoformat(),
    )

    cuerpo = google.cuerpo_de("PATCH")
    assert cuerpo["start"] == {"date": nuevo_inicio.isoformat()}
    assert cuerpo["end"] == {"date": (nuevo_inicio + timedelta(days=3)).isoformat()}


# --- Leer ----------------------------------------------------------------------------


async def test_listar_hoy_pide_las_medianoches_locales() -> None:
    google = GoogleFalso()
    hoy = HOY.isoformat()

    await _pedir_y_aprobar(google, "eventos_del_calendario", desde=hoy, hasta=hoy)

    (pedido,) = [p for p in google.pedidos if "/events" in p.url.path]
    parametros = parse_qs(urlparse(str(pedido.url)).query)
    assert parametros["timeMin"] == [f"{hoy}T00:00:00-03:00"]
    assert parametros["timeMax"] == [f"{_dia(1).isoformat()}T00:00:00-03:00"]
    assert unquote(pedido.url.path).endswith("/calendars/yo@gmail.com/events")


# --- Eliminar: el día del resumen ------------------------------------------------------


async def test_el_resumen_de_eliminar_nombra_el_dia_local_aunque_google_mande_utc() -> None:
    """Un calendario en otro huso manda el evento de las 22:30 locales como 01:30 UTC.

    La hora ya se mostraba en Argentina; el día salía en el huso del
    calendario, y quedaban inconsistentes: "22:30 … del <día siguiente>".
    """
    dia = _dia(3)
    siguiente = dia + timedelta(days=1)
    google = GoogleFalso(
        [
            {
                "id": "id-cena",
                "summary": "Cena",
                "start": {"dateTime": f"{siguiente.isoformat()}T01:30:00Z"},
                "end": {"dateTime": f"{siguiente.isoformat()}T02:30:00Z"},
            }
        ]
    )

    estado = await _pedir_y_aprobar(
        google, "eliminar_evento_del_calendario", fecha=dia.isoformat(), titulo="cena"
    )

    resumen = estado["__interrupt__"][0].value["resumen"]
    assert "22:30" in resumen
    assert _en_palabras(dia) in resumen
    assert _en_palabras(siguiente) not in resumen


# --- Eventos recurrentes (PB-025) ---------------------------------------------------


def _proximo(dia_de_semana: int) -> date:
    """El próximo día de la semana pedido, nunca hoy (0 = lunes)."""
    return HOY + timedelta(days=(dia_de_semana - HOY.weekday()) % 7 or 7)


async def test_una_serie_manda_la_regla_y_la_zona_que_google_exige() -> None:
    """Sin `timeZone` en start y end, Google rechaza un recurrente con 400."""
    google = GoogleFalso()
    lunes = _proximo(0)

    await _pedir_y_aprobar(
        google,
        "crear_evento_en_calendario",
        titulo="Gimnasio",
        fecha=lunes.isoformat(),
        hora_inicio="19:00",
        repetir="semanal",
        dias="lunes, miércoles",
        veces=4,
    )

    cuerpo = google.cuerpo_de("POST")
    assert cuerpo["recurrence"] == ["RRULE:FREQ=WEEKLY;BYDAY=MO,WE;COUNT=4"]
    assert cuerpo["start"] == {
        "dateTime": f"{lunes.isoformat()}T19:00:00-03:00",
        "timeZone": "America/Argentina/Buenos_Aires",
    }
    assert cuerpo["end"]["timeZone"] == "America/Argentina/Buenos_Aires"


async def test_el_fin_de_la_serie_viaja_en_utc() -> None:
    """RFC 5545: con `dateTime`, UNTIL va en UTC. 23:59:59 locales = 02:59:59Z del día siguiente."""
    google = GoogleFalso()
    manana = _dia(1)
    ultimo = _dia(10)

    await _pedir_y_aprobar(
        google,
        "crear_evento_en_calendario",
        titulo="Pastilla",
        fecha=manana.isoformat(),
        hora_inicio="09:00",
        repetir="diaria",
        hasta=ultimo.isoformat(),
    )

    siguiente = ultimo + timedelta(days=1)
    assert google.cuerpo_de("POST")["recurrence"] == [
        f"RRULE:FREQ=DAILY;UNTIL={siguiente:%Y%m%d}T025959Z"
    ]


async def test_borrar_toda_la_serie_borra_el_id_de_la_serie() -> None:
    """Google entrega cada repetición con `recurringEventId`: ése es el que se borra."""
    lunes = _proximo(0)
    google = GoogleFalso(
        [
            {
                "id": f"gym_{lunes:%Y%m%d}T220000Z",
                "recurringEventId": "gym",
                "summary": "Gimnasio",
                "start": {"dateTime": f"{lunes.isoformat()}T19:00:00-03:00"},
                "end": {"dateTime": f"{lunes.isoformat()}T20:00:00-03:00"},
            }
        ]
    )

    await _pedir_y_aprobar(
        google,
        "eliminar_evento_del_calendario",
        fecha=lunes.isoformat(),
        titulo="gimnasio",
        toda_la_serie=True,
    )

    (borrado,) = [p for p in google.pedidos if p.method == "DELETE"]
    assert borrado.url.path.endswith("/events/gym")
