"""Tests de las tools con eventos recurrentes (PB-025): crear, listar, borrar.

Las fechas se calculan desde el próximo miércoles: así el caso "todos los
lunes pedido un miércoles" existe siempre, y ninguna fecha se vuelve pasado.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.domain.entities.evento import Evento
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import _en_hora_local, _linea, construir_herramientas
from tests.dobles import CalendarioFalso, ModeloFalso

USUARIO = uuid4()
_HOY = datetime.now(ZONA_HORARIA).date()
MIERCOLES = _HOY + timedelta(days=(2 - _HOY.weekday()) % 7 or 7)  # el próximo, nunca hoy
LUNES_SIGUIENTE = MIERCOLES + timedelta(days=5)


async def _pedir(calendario: CalendarioFalso, herramienta: str, **args: Any) -> tuple[Any, Any]:
    pedido = AIMessage("", tool_calls=[{"name": herramienta, "args": args, "id": "c1"}])
    modelo = ModeloFalso(guion=[pedido, AIMessage("listo")])
    grafo = construir_grafo(modelo, construir_herramientas(calendario), InMemorySaver())
    estado = await grafo.ainvoke(
        {"messages": [HumanMessage("pedido")]},
        config={"configurable": {"thread_id": "hilo"}},
        context=ContextoDeAgente(usuario_id=USUARIO),
    )
    return estado, grafo


async def _aprobar(grafo: Any) -> None:
    await grafo.ainvoke(
        Command(resume={"aprobado": True}),
        config={"configurable": {"thread_id": "hilo"}},
        context=ContextoDeAgente(usuario_id=USUARIO),
    )


def _resultado(estado: Any) -> str:
    return str(next(m for m in estado["messages"] if isinstance(m, ToolMessage)).content)


def _crear(**extra: Any) -> dict[str, Any]:
    return {"titulo": "Gimnasio", "fecha": MIERCOLES.isoformat(), "hora_inicio": "19:00", **extra}


# --- Crear una serie -------------------------------------------------------------


async def test_todos_los_lunes_pedido_un_miercoles_arranca_el_lunes() -> None:
    """RFC 5545: la fecha de inicio es la primera repetición aunque no cumpla la regla."""
    calendario = CalendarioFalso()
    _, grafo = await _pedir(
        calendario, "crear_evento_en_calendario", **_crear(repetir="semanal", dias="lunes")
    )
    await _aprobar(grafo)

    _, evento = calendario.creados[0]
    assert evento.inicio.date() == LUNES_SIGUIENTE  # no ese miércoles suelto
    assert evento.recurrencia is not None
    assert evento.recurrencia.dias == (0,)


async def test_los_dias_se_entienden_con_y_sin_tilde() -> None:
    calendario = CalendarioFalso()
    _, grafo = await _pedir(
        calendario,
        "crear_evento_en_calendario",
        **_crear(repetir="semanal", dias="Sabado y miércoles"),
    )
    await _aprobar(grafo)

    _, evento = calendario.creados[0]
    assert evento.recurrencia is not None
    assert evento.recurrencia.dias == (2, 5)


async def test_semanal_sin_dias_repite_el_dia_de_la_fecha() -> None:
    calendario = CalendarioFalso()
    _, grafo = await _pedir(calendario, "crear_evento_en_calendario", **_crear(repetir="semanal"))
    await _aprobar(grafo)

    _, evento = calendario.creados[0]
    assert evento.recurrencia is not None
    assert evento.recurrencia.dias == (MIERCOLES.weekday(),)
    assert evento.inicio.date() == MIERCOLES


async def test_el_resumen_describe_la_serie_completa() -> None:
    estado, _ = await _pedir(
        CalendarioFalso(),
        "crear_evento_en_calendario",
        **_crear(repetir="semanal", dias="lunes, miércoles", veces=4),
    )

    resumen = estado["__interrupt__"][0].value["resumen"]
    assert 'Crear "Gimnasio" todos los lunes y miércoles de 19:00 a 20:00' in resumen
    assert "desde el miércoles" in resumen  # el miércoles cumple la regla: arranca ese día
    assert resumen.endswith("(4 veces)")


async def test_sin_fin_lo_dice() -> None:
    estado, _ = await _pedir(
        CalendarioFalso(), "crear_evento_en_calendario", **_crear(repetir="diaria")
    )

    assert "todos los días" in estado["__interrupt__"][0].value["resumen"]
    assert "(sin fecha de fin)" in estado["__interrupt__"][0].value["resumen"]


async def test_una_sola_vez_no_lleva_recurrencia() -> None:
    calendario = CalendarioFalso()
    _, grafo = await _pedir(calendario, "crear_evento_en_calendario", **_crear())
    await _aprobar(grafo)

    _, evento = calendario.creados[0]
    assert evento.recurrencia is None


async def test_un_dia_inventado_no_llega_a_la_pausa() -> None:
    calendario = CalendarioFalso()

    estado, _ = await _pedir(
        calendario, "crear_evento_en_calendario", **_crear(repetir="semanal", dias="lunes, feriado")
    )

    assert "__interrupt__" not in estado
    assert "«feriado»" in _resultado(estado)
    assert calendario.creados == []


async def test_hasta_y_veces_juntos_no_llegan_a_la_pausa() -> None:
    estado, _ = await _pedir(
        CalendarioFalso(),
        "crear_evento_en_calendario",
        **_crear(repetir="diaria", hasta=(MIERCOLES + timedelta(days=30)).isoformat(), veces=3),
    )

    assert "__interrupt__" not in estado
    assert "no las dos cosas" in _resultado(estado)


async def test_una_frecuencia_desconocida_se_explica() -> None:
    estado, _ = await _pedir(
        CalendarioFalso(), "crear_evento_en_calendario", **_crear(repetir="cada tanto")
    )

    assert "__interrupt__" not in estado
    assert '"semanal"' in _resultado(estado)


# --- Listar y borrar ----------------------------------------------------------------


def _repeticion(dia: date) -> Evento:
    """Una repetición de la serie "gym", tal como la entrega Google con singleEvents."""
    inicio = datetime(dia.year, dia.month, dia.day, 22, 0, tzinfo=UTC)  # 19:00 locales
    return Evento(
        titulo="Gimnasio",
        inicio=inicio,
        fin=inicio + timedelta(hours=1),
        id=f"gym_{dia:%Y%m%d}",
        serie_id="gym",
    )


def test_una_repeticion_se_lista_como_tal() -> None:
    evento = _repeticion(LUNES_SIGUIENTE)

    assert "(se repite)" in _linea(evento, _en_hora_local(evento))


async def test_borrar_una_repeticion_dice_que_las_demas_quedan() -> None:
    calendario = CalendarioFalso(eventos=(_repeticion(LUNES_SIGUIENTE),))
    estado, grafo = await _pedir(
        calendario,
        "eliminar_evento_del_calendario",
        fecha=LUNES_SIGUIENTE.isoformat(),
        titulo="gimnasio",
    )

    resumen = estado["__interrupt__"][0].value["resumen"]
    await _aprobar(grafo)

    assert resumen.startswith("Eliminar sólo esta repetición")
    assert "las demás repeticiones quedan" in resumen
    assert calendario.eliminados == [(USUARIO, f"gym_{LUNES_SIGUIENTE:%Y%m%d}")]


async def test_toda_la_serie_borra_la_serie_y_no_la_repeticion() -> None:
    calendario = CalendarioFalso(eventos=(_repeticion(LUNES_SIGUIENTE),))
    estado, grafo = await _pedir(
        calendario,
        "eliminar_evento_del_calendario",
        fecha=LUNES_SIGUIENTE.isoformat(),
        titulo="gimnasio",
        toda_la_serie=True,
    )

    resumen = estado["__interrupt__"][0].value["resumen"]
    await _aprobar(grafo)

    assert "toda la serie de «Gimnasio»" in resumen
    assert calendario.eliminados == [(USUARIO, "gym")]  # el id de la serie


async def test_toda_la_serie_sobre_un_evento_suelto_lo_explica() -> None:
    suelto = Evento(
        titulo="Dentista",
        inicio=datetime(
            LUNES_SIGUIENTE.year, LUNES_SIGUIENTE.month, LUNES_SIGUIENTE.day, 13, tzinfo=UTC
        ),
        id="id-dentista",
    )
    calendario = CalendarioFalso(eventos=(suelto,))

    estado, _ = await _pedir(
        calendario,
        "eliminar_evento_del_calendario",
        fecha=LUNES_SIGUIENTE.isoformat(),
        titulo="dentista",
        toda_la_serie=True,
    )

    assert "__interrupt__" not in estado
    assert "no se repite" in _resultado(estado)
    assert calendario.eliminados == []


async def test_modificar_una_repeticion_lo_aclara() -> None:
    calendario = CalendarioFalso(eventos=(_repeticion(LUNES_SIGUIENTE),))

    estado, _ = await _pedir(
        calendario,
        "modificar_evento_del_calendario",
        fecha=LUNES_SIGUIENTE.isoformat(),
        titulo="gimnasio",
        nueva_hora_inicio="20:00",
    )

    assert "(sólo esta repetición)" in estado["__interrupt__"][0].value["resumen"]
