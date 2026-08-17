"""Tests del grafo del agente (PB-005).

Corren sin red y sin API key: el modelo entra por parámetro, así que se puede
ejercitar el ciclo completo —incluido el paso por las herramientas— con un
doble. Ése es el motivo de que `construir_grafo` no arme el modelo adentro.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.infrastructure.llm.grafo import MAX_MENSAJES_DE_HISTORIAL, construir_grafo
from src.infrastructure.llm.herramientas import HERRAMIENTAS
from src.infrastructure.llm.prompt import INSTRUCCIONES
from tests.dobles import ModeloFalso

PEDIDO_DE_HERRAMIENTA = AIMessage(
    "", tool_calls=[{"name": "fecha_y_hora_actual", "args": {}, "id": "llamada-1"}]
)


def _grafo(*guion: AIMessage, herramientas: Any = HERRAMIENTAS) -> tuple[Any, ModeloFalso]:
    modelo = ModeloFalso(guion=list(guion))
    return construir_grafo(modelo, herramientas, InMemorySaver()), modelo


async def _preguntar(grafo: Any, texto: str, hilo: str = "hilo-1") -> dict[str, Any]:
    resultado: dict[str, Any] = await grafo.ainvoke(
        {"messages": [HumanMessage(texto)]},
        config={"configurable": {"thread_id": hilo}},
    )
    return resultado


# --- Camino simple ----------------------------------------------------------


async def test_contesta_lo_que_dijo_el_modelo() -> None:
    grafo, _ = _grafo(AIMessage("Hola, ¿en qué te ayudo?"))

    estado = await _preguntar(grafo, "hola")

    assert estado["messages"][-1].content == "Hola, ¿en qué te ayudo?"


async def test_el_system_prompt_viaja_en_cada_invocacion() -> None:
    grafo, modelo = _grafo(AIMessage("listo"))

    await _preguntar(grafo, "hola")

    primero = modelo.recibidos[0][0]
    assert isinstance(primero, SystemMessage)
    assert primero.content == INSTRUCCIONES


async def test_las_herramientas_se_le_declaran_al_modelo() -> None:
    _, modelo = _grafo(AIMessage("listo"))

    assert [h.name for h in modelo.herramientas_asociadas] == ["fecha_y_hora_actual"]


# --- Ciclo de tool-calling --------------------------------------------------


async def test_ejecuta_la_herramienta_y_vuelve_al_modelo() -> None:
    """El ciclo completo: el modelo pide, la herramienta corre, el modelo cierra."""
    grafo, modelo = _grafo(PEDIDO_DE_HERRAMIENTA, AIMessage("Hoy es lunes."))

    estado = await _preguntar(grafo, "¿qué día es hoy?")

    tipos = [type(m).__name__ for m in estado["messages"]]
    assert tipos == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert estado["messages"][-1].content == "Hoy es lunes."
    # Dos vueltas por el nodo del agente: antes y después de la herramienta.
    assert len(modelo.recibidos) == 2


async def test_el_resultado_de_la_herramienta_llega_al_modelo() -> None:
    """Quien le habla a la persona es el modelo, nunca la salida cruda."""
    grafo, modelo = _grafo(PEDIDO_DE_HERRAMIENTA, AIMessage("Hoy es lunes."))

    await _preguntar(grafo, "¿qué día es hoy?")

    segunda_vuelta = modelo.recibidos[1]
    assert any(isinstance(m, ToolMessage) for m in segunda_vuelta)


async def test_sin_herramientas_el_grafo_sigue_siendo_valido() -> None:
    """La lista vacía no puede romper el armado: es el modo degradado del grafo."""
    grafo, modelo = _grafo(AIMessage("hola"), herramientas=())

    estado = await _preguntar(grafo, "hola")

    assert estado["messages"][-1].content == "hola"
    assert modelo.herramientas_asociadas == []


# --- Memoria conversacional (RF-09) -----------------------------------------


async def test_recuerda_los_turnos_anteriores_del_mismo_hilo() -> None:
    grafo, modelo = _grafo(AIMessage("primera"), AIMessage("segunda"))

    await _preguntar(grafo, "me llamo Ramiro")
    await _preguntar(grafo, "¿cómo me llamo?")

    segunda_invocacion = modelo.recibidos[1]
    humanos = [m.content for m in segunda_invocacion if isinstance(m, HumanMessage)]
    assert humanos == ["me llamo Ramiro", "¿cómo me llamo?"]


async def test_los_hilos_distintos_no_se_mezclan() -> None:
    """Dos personas no pueden verse la conversación (RF-18)."""
    grafo, modelo = _grafo(AIMessage("respuesta"))

    await _preguntar(grafo, "secreto de Ana", hilo="ana")
    await _preguntar(grafo, "hola", hilo="beto")

    de_beto = [m.content for m in modelo.recibidos[1] if isinstance(m, HumanMessage)]
    assert de_beto == ["hola"]


async def test_el_historial_se_recorta() -> None:
    """Sin tope, una charla larga agota la ventana de contexto y encarece el turno."""
    grafo, modelo = _grafo(AIMessage("ok"))

    for i in range(MAX_MENSAJES_DE_HISTORIAL):
        await _preguntar(grafo, f"mensaje {i}")

    # El system prompt se antepone aparte del historial recortado.
    ultima = modelo.recibidos[-1]
    assert len(ultima) <= MAX_MENSAJES_DE_HISTORIAL + 1


async def test_el_recorte_deja_el_historial_empezando_en_la_persona() -> None:
    """Un ToolMessage huérfano de su AIMessage hace que Gemini devuelva 400."""
    grafo, modelo = _grafo(AIMessage("ok"))

    for i in range(MAX_MENSAJES_DE_HISTORIAL + 5):
        await _preguntar(grafo, f"mensaje {i}")

    sin_system = [m for m in modelo.recibidos[-1] if not isinstance(m, SystemMessage)]
    assert isinstance(sin_system[0], HumanMessage)
