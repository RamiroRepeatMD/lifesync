"""Tests de las herramientas de tareas (PB-028): el ciclo RF-08 y el modelo.

Mismo esqueleto que la escritura del calendario: se atraviesa el grafo real
con un modelo falso, y las aserciones van sobre el doble del puerto.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from uuid import uuid4

import structlog
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from src.domain.entities.tarea import Tarea
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import ModeloFalso, TareasFalsas

USUARIO = uuid4()


def _grafo_con(tareas: TareasFalsas, *guion: AIMessage) -> Any:
    modelo = ModeloFalso(guion=list(guion))
    return construir_grafo(modelo, construir_herramientas(None, tareas), InMemorySaver())


async def _preguntar(grafo: Any, texto: str, hilo: str = "hilo-1") -> dict[str, Any]:
    resultado: dict[str, Any] = await grafo.ainvoke(
        {"messages": [HumanMessage(texto)]},
        config={"configurable": {"thread_id": hilo}},
        context=ContextoDeAgente(usuario_id=USUARIO),
    )
    return resultado


async def _reanudar(grafo: Any, aprobado: bool, hilo: str = "hilo-1") -> dict[str, Any]:
    resultado: dict[str, Any] = await grafo.ainvoke(
        Command(resume={"aprobado": aprobado}),
        config={"configurable": {"thread_id": hilo}},
        context=ContextoDeAgente(usuario_id=USUARIO),
    )
    return resultado


def _pedido(nombre: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": nombre, "args": args, "id": "t1"}])


def _pendientes() -> tuple[Tarea, ...]:
    return (
        Tarea(titulo="Comprar regalo", vencimiento=date(2026, 9, 19), id="t-regalo"),
        Tarea(titulo="Llamar al banco", id="t-banco"),
    )


# --- Listar ------------------------------------------------------------------


async def test_listar_redacta_con_vencimiento_en_palabras() -> None:
    tareas = TareasFalsas(pendientes=_pendientes())
    grafo = _grafo_con(tareas, _pedido("tareas_pendientes"), AIMessage("listo"))

    await _preguntar(grafo, "¿qué tengo pendiente?")

    assert tareas.consultados == [USUARIO]


async def test_sin_tareas_lo_dice_sin_inventar() -> None:
    tareas = TareasFalsas()
    grafo = _grafo_con(tareas, _pedido("tareas_pendientes"), AIMessage("nada"))

    estado = await _preguntar(grafo, "¿qué tengo?")

    contenidos = [str(m.content) for m in estado["messages"]]
    assert any("No hay tareas pendientes" in c for c in contenidos)


# --- Crear: el ciclo RF-08 ---------------------------------------------------


async def test_crear_sin_confirmacion_no_escribe() -> None:
    tareas = TareasFalsas()
    grafo = _grafo_con(
        tareas, _pedido("crear_tarea", titulo="Comprar regalo", fecha_limite="2026-09-19")
    )

    estado = await _preguntar(grafo, "anotá comprar regalo")

    assert "__interrupt__" in estado
    assert tareas.creadas == []


async def test_crear_aprobado_escribe_una_sola_vez() -> None:
    tareas = TareasFalsas()
    grafo = _grafo_con(
        tareas,
        _pedido("crear_tarea", titulo="Comprar regalo", fecha_limite="2026-09-19"),
        AIMessage("anotada"),
    )

    await _preguntar(grafo, "anotá comprar regalo")
    await _reanudar(grafo, aprobado=True)

    assert len(tareas.creadas) == 1
    _, creada = tareas.creadas[0]
    assert creada.titulo == "Comprar regalo"
    assert creada.vencimiento == date(2026, 9, 19)


async def test_crear_rechazado_no_escribe_nada() -> None:
    tareas = TareasFalsas()
    grafo = _grafo_con(tareas, _pedido("crear_tarea", titulo="X"), AIMessage("ok"))

    await _preguntar(grafo, "anotá x")
    await _reanudar(grafo, aprobado=False)

    assert tareas.creadas == []


async def test_crear_con_fecha_invalida_pide_el_formato() -> None:
    tareas = TareasFalsas()
    grafo = _grafo_con(
        tareas, _pedido("crear_tarea", titulo="X", fecha_limite="mañana"), AIMessage("ok")
    )

    estado = await _preguntar(grafo, "anotá x para mañana")

    assert "__interrupt__" not in estado  # ni siquiera propuso
    assert tareas.creadas == []


# --- Completar ---------------------------------------------------------------


async def test_completar_desambigua_antes_de_confirmar() -> None:
    dos = (
        Tarea(titulo="Llamar al banco", id="a"),
        Tarea(titulo="Llamar al médico", id="b"),
    )
    tareas = TareasFalsas(pendientes=dos)
    grafo = _grafo_con(tareas, _pedido("completar_tarea", titulo="llamar"), AIMessage("ok"))

    estado = await _preguntar(grafo, "ya llamé")

    assert "__interrupt__" not in estado
    assert tareas.completadas == []


async def test_completar_lo_que_no_existe_avisa() -> None:
    tareas = TareasFalsas(pendientes=_pendientes())
    grafo = _grafo_con(tareas, _pedido("completar_tarea", titulo="inexistente"), AIMessage("ok"))

    estado = await _preguntar(grafo, "ya hice eso")

    assert "__interrupt__" not in estado
    assert tareas.completadas == []


async def test_completar_aprobado_patchea_la_correcta() -> None:
    tareas = TareasFalsas(pendientes=_pendientes())
    grafo = _grafo_con(tareas, _pedido("completar_tarea", titulo="banco"), AIMessage("hecho"))

    estado = await _preguntar(grafo, "ya llamé al banco")
    assert estado["__interrupt__"][0].value["resumen"] == "Marcar como hecha: Llamar al banco"
    await _reanudar(grafo, aprobado=True)

    assert tareas.completadas == [(USUARIO, "t-banco")]


async def test_completar_rechazado_deja_la_tarea_pendiente() -> None:
    tareas = TareasFalsas(pendientes=_pendientes())
    grafo = _grafo_con(tareas, _pedido("completar_tarea", titulo="banco"), AIMessage("ok"))

    await _preguntar(grafo, "ya llamé")
    await _reanudar(grafo, aprobado=False)

    assert tareas.completadas == []


# --- Seguridad y privacidad --------------------------------------------------


def test_el_modelo_no_ve_el_usuario_en_ninguna_tool_de_tareas() -> None:
    herramientas = {h.name: h for h in construir_herramientas(None, TareasFalsas())}

    esperados = {
        "tareas_pendientes": set(),
        "crear_tarea": {"titulo", "fecha_limite", "notas"},
        "completar_tarea": {"titulo"},
    }
    for nombre, campos in esperados.items():
        esquema = herramientas[nombre].tool_call_schema
        assert isinstance(esquema, type) and issubclass(esquema, BaseModel)
        assert set(esquema.model_json_schema().get("properties", {})) == campos, nombre


async def test_los_titulos_de_tareas_no_se_loguean() -> None:
    tareas = TareasFalsas()
    grafo = _grafo_con(
        tareas,
        _pedido("crear_tarea", titulo="Turno con la psicóloga", notas="tema privado"),
        AIMessage("listo"),
    )

    with structlog.testing.capture_logs() as eventos:
        await _preguntar(grafo, "anotá el turno")
        await _reanudar(grafo, aprobado=True)

    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "psicóloga" not in registrado
    assert "tema privado" not in registrado


def test_sin_puerto_de_tareas_no_se_ofrecen_las_tools() -> None:
    from tests.dobles import CalendarioFalso

    nombres = [h.name for h in construir_herramientas(CalendarioFalso(), None)]

    assert "tareas_pendientes" not in nombres
    assert "crear_tarea" not in nombres
    assert len(nombres) == 5  # fecha + las 4 de calendario


def test_solo_tareas_sin_calendario_tambien_funciona() -> None:
    nombres = [h.name for h in construir_herramientas(None, TareasFalsas())]

    assert nombres == ["fecha_y_hora_actual", "tareas_pendientes", "crear_tarea", "completar_tarea"]
