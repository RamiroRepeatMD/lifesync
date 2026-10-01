"""Tests de la respuesta post-acción (PB-026).

El caso que importa: el modelo falla DESPUÉS de que una escritura se ejecutó.
Hasta PB-026 la persona recibía "el servicio no está disponible" aunque el
evento se había creado, y lo natural —reintentar— lo duplicaba.

Se ejercita de punta a punta con piezas reales: el grafo, el ToolNode, el
checkpointer en memoria y el `AgenteGemini`. El único falso es el modelo, que
es lo único que tiene que fallar.

Desde la segunda versión de RF-08, crear y completar salen directo: la falla
del modelo llega en el mismo turno del pedido, sin un "sí" de por medio.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
import structlog
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.domain.entities.tarea import Tarea
from src.domain.exceptions import AgenteNoDisponibleError
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.llm.agente_gemini import AgenteGemini
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import CalendarioFalso, ModeloFalso, TareasFalsas

USUARIO = uuid4()
CONFIGURACION = {"configurable": {"thread_id": str(USUARIO)}}
MANANA = (datetime.now(ZONA_HORARIA) + timedelta(days=1)).date().isoformat()


def _pedido(nombre: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": nombre, "args": args, "id": f"llamada-{nombre}"}])


def _crear_dentista(titulo: str = "Dentista") -> AIMessage:
    return _pedido("crear_evento_en_calendario", titulo=titulo, fecha=MANANA, hora_inicio="15:00")


def _armar(
    *guion: AIMessage,
    fallar_en: frozenset[int] = frozenset(),
    error: str = "504 DEADLINE_EXCEEDED (simulado)",
    tareas: TareasFalsas | None = None,
) -> tuple[AgenteGemini, Any, ModeloFalso, CalendarioFalso, TareasFalsas]:
    calendario = CalendarioFalso()
    tareas = tareas or TareasFalsas()
    modelo = ModeloFalso(guion=list(guion), fallar_en_llamadas=fallar_en, error=error)
    grafo = construir_grafo(modelo, construir_herramientas(calendario, tareas), InMemorySaver())
    return AgenteGemini(grafo), grafo, modelo, calendario, tareas


async def _decir(agente: AgenteGemini, texto: str) -> str:
    return await agente.responder(
        ConsultaDelUsuario(conversacion_id=USUARIO, usuario_id=USUARIO, texto=texto)
    )


# --- El caso central ---------------------------------------------------------


async def test_si_el_modelo_falla_tras_crear_se_cuenta_lo_que_se_hizo() -> None:
    agente, _, _, calendario, _ = _armar(
        _crear_dentista(), AIMessage("(no se usa: esta llamada falla)"), fallar_en=frozenset({1})
    )

    respuesta = await _decir(agente, "agendame el dentista mañana a las 15")

    assert len(calendario.creados) == 1  # se escribió, y una sola vez
    assert respuesta.startswith('Listo — Evento creado: "Dentista"')
    assert "15:00 a 16:00" in respuesta  # los datos concretos, no un "listo" genérico
    assert "no hace falta repetirla" in respuesta


async def test_el_turno_queda_cerrado_y_el_siguiente_anda_normal() -> None:
    agente, grafo, modelo, _, _ = _armar(
        _crear_dentista(),
        AIMessage("(no se usa: esta llamada falla)"),
        AIMessage("¡De nada!"),
        fallar_en=frozenset({1}),
    )
    reemplazo = await _decir(agente, "agendame el dentista mañana a las 15")

    estado = await grafo.aget_state(CONFIGURACION)
    assert estado.next == ()  # no quedó el paso del modelo colgado
    assert estado.values["messages"][-1].content == reemplazo

    assert await _decir(agente, "gracias") == "¡De nada!"
    # El modelo recibió el historial bien formado: la acción, lo que se le
    # contestó a la persona, y recién después el mensaje nuevo.
    tipos = [type(m) for m in modelo.recibidos[-1][-3:]]
    assert tipos == [ToolMessage, AIMessage, HumanMessage]


async def test_la_cuota_agotada_tras_la_accion_tambien_se_cuenta() -> None:
    """Lo que la persona necesita saber es que se hizo, no que se acabó la cuota."""
    agente, _, _, calendario, _ = _armar(
        _crear_dentista(),
        AIMessage("(no se usa)"),
        fallar_en=frozenset({1}),
        error="429 RESOURCE_EXHAUSTED (simulado)",
    )

    respuesta = await _decir(agente, "agendame el dentista mañana a las 15")

    assert len(calendario.creados) == 1
    assert "no hace falta repetirla" in respuesta


async def test_el_mecanismo_cubre_tambien_las_tareas() -> None:
    tareas = TareasFalsas(pendientes=(Tarea(titulo="Pagar la luz", id="t-luz"),))
    agente, _, _, _, tareas = _armar(
        _pedido("completar_tarea", titulo="luz"),
        AIMessage("(no se usa)"),
        fallar_en=frozenset({1}),
        tareas=tareas,
    )

    respuesta = await _decir(agente, "ya pagué la luz")

    assert tareas.completadas == [(USUARIO, "t-luz")]
    assert respuesta.startswith("Listo — Tarea marcada como hecha: Pagar la luz")


# --- Sin acción ejecutada, el error sigue siendo error --------------------------


async def test_rechazar_y_que_falle_el_modelo_no_inventa_un_listo() -> None:
    """Un borrado sigue pidiendo el sí: rechazado, no hay nada que contar."""
    tareas = TareasFalsas(pendientes=(Tarea(titulo="Pagar la luz", id="t-luz"),))
    agente, _, _, _, tareas = _armar(
        _pedido("eliminar_tarea", titulo="luz"),
        AIMessage("(no se usa)"),
        fallar_en=frozenset({1}),
        tareas=tareas,
    )
    assert "¿Confirmás?" in await _decir(agente, "borrá lo de la luz")

    with pytest.raises(AgenteNoDisponibleError):
        await _decir(agente, "no")

    assert tareas.eliminadas == []


async def test_una_lectura_con_falla_posterior_sigue_siendo_error() -> None:
    lectura = _pedido("eventos_del_calendario", desde=MANANA, hasta=MANANA)
    agente, _, _, _, _ = _armar(lectura, AIMessage("(no se usa)"), fallar_en=frozenset({1}))

    with pytest.raises(AgenteNoDisponibleError):
        await _decir(agente, "¿qué tengo mañana?")


# --- Respuesta contextual ---------------------------------------------------------


async def test_el_modelo_recibe_los_datos_concretos_de_lo_que_se_hizo() -> None:
    """Con eso redacta una confirmación con datos, no un "listo" genérico."""
    agente, _, modelo, _, _ = _armar(_crear_dentista(), AIMessage("Listo, agendado."))
    await _decir(agente, "agendame el dentista mañana a las 15")

    resultado = [m for m in modelo.recibidos[-1] if isinstance(m, ToolMessage)][-1]
    assert str(resultado.content).startswith('Evento creado: "Dentista"')
    assert "15:00 a 16:00" in str(resultado.content)


async def test_si_una_accion_directa_precede_a_una_pregunta_la_pregunta_la_cuenta() -> None:
    """El modelo pidió de a una: lo directo ya se hizo y el borrado espera su sí."""
    tareas = TareasFalsas(pendientes=(Tarea(titulo="Pagar la luz", id="t-luz"),))
    agente, _, _, calendario, tareas = _armar(
        _crear_dentista(),
        _pedido("eliminar_tarea", titulo="luz"),
        AIMessage("Listo, las dos cosas."),
        tareas=tareas,
    )

    pregunta = await _decir(agente, "agendame el dentista mañana a las 15 y borrá lo de la luz")

    assert pregunta.startswith('Listo — Evento creado: "Dentista"')
    assert "Eliminar la tarea: Pagar la luz" in pregunta
    assert "¿Confirmás?" in pregunta
    assert len(calendario.creados) == 1
    assert tareas.eliminadas == []  # el borrado todavía espera su sí

    await _decir(agente, "sí")
    assert tareas.eliminadas == [(USUARIO, "t-luz")]


# --- Privacidad (RF-18) -------------------------------------------------------------


async def test_el_turno_fallido_no_loguea_los_datos_de_la_accion() -> None:
    agente, _, _, _, _ = _armar(
        _crear_dentista(titulo="Sesión con la psicóloga"),
        AIMessage("(no se usa)"),
        fallar_en=frozenset({1}),
    )
    with structlog.testing.capture_logs() as eventos:
        respuesta = await _decir(agente, "agendame la sesión mañana a las 15")

    assert "psicóloga" in respuesta  # a la persona sí se le cuenta
    assert any(e["event"] == "agente.redaccion_fallida_tras_accion" for e in eventos)
    assert "psicóloga" not in json.dumps(eventos, default=str)
