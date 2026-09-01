"""Tests de las herramientas de escritura del calendario (PB-016, RF-08).

El grupo que manda es el primero: **sin confirmación no hay escritura**. Se
prueba atravesando el grafo real —modelo falso, calendario falso, checkpointer
real— porque la garantía es estructural y sólo existe dentro del grafo.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import structlog
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from src.domain.entities.evento import Evento
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import CalendarioFalso, ModeloFalso

USUARIO = uuid4()

PEDIDO_CREAR = AIMessage(
    "",
    tool_calls=[
        {
            "name": "crear_evento_en_calendario",
            "args": {"titulo": "Dentista", "fecha": "2026-09-05", "hora_inicio": "10:00"},
            "id": "t1",
        }
    ],
)


def _grafo_con(calendario: CalendarioFalso, *guion: AIMessage) -> Any:
    modelo = ModeloFalso(guion=list(guion))
    return construir_grafo(modelo, construir_herramientas(calendario), InMemorySaver())


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


# --- RF-08: la garantía estructural ------------------------------------------


async def test_sin_confirmacion_no_hay_escritura() -> None:
    """El test más importante del PB: pedir crear NO crea. Pausa y pregunta."""
    calendario = CalendarioFalso()
    grafo = _grafo_con(calendario, PEDIDO_CREAR, AIMessage("cierro"))

    estado = await _preguntar(grafo, "agendame dentista el sábado a las 10")

    assert "__interrupt__" in estado
    assert calendario.creados == []  # ni una escritura antes del sí


async def test_el_resumen_del_interrupt_describe_exactamente_la_accion() -> None:
    """Lo que la persona confirma es lo que se ejecuta, sin reinterpretación."""
    calendario = CalendarioFalso()
    grafo = _grafo_con(calendario, PEDIDO_CREAR, AIMessage("cierro"))

    estado = await _preguntar(grafo, "agendame dentista")

    resumen = estado["__interrupt__"][0].value["resumen"]
    assert "Dentista" in resumen
    assert "10:00" in resumen and "11:00" in resumen  # duración default: 60 min


async def test_aprobar_escribe_exactamente_una_vez() -> None:
    calendario = CalendarioFalso()
    grafo = _grafo_con(calendario, PEDIDO_CREAR, AIMessage("Listo, agendado."))

    await _preguntar(grafo, "agendame dentista")
    estado = await _reanudar(grafo, aprobado=True)

    assert len(calendario.creados) == 1
    usuario_id, evento = calendario.creados[0]
    assert usuario_id == USUARIO  # la agenda de quien escribe, del contexto
    assert evento.titulo == "Dentista"
    assert estado["messages"][-1].content == "Listo, agendado."


async def test_rechazar_no_escribe_nada() -> None:
    calendario = CalendarioFalso()
    grafo = _grafo_con(calendario, PEDIDO_CREAR, AIMessage("Ok, no lo agendo."))

    await _preguntar(grafo, "agendame dentista")
    await _reanudar(grafo, aprobado=False)

    assert calendario.creados == []


async def test_un_resume_deforme_no_aprueba() -> None:
    """Sólo `{"aprobado": True}` ejecuta: cualquier otra forma cancela."""
    calendario = CalendarioFalso()
    grafo = _grafo_con(calendario, PEDIDO_CREAR, AIMessage("ok"))

    await _preguntar(grafo, "agendame dentista")
    await grafo.ainvoke(
        Command(resume="si"),  # un string suelto, no el dict esperado
        config={"configurable": {"thread_id": "hilo-1"}},
        context=ContextoDeAgente(usuario_id=USUARIO),
    )

    assert calendario.creados == []


# --- El esquema que ve el modelo ---------------------------------------------


def test_el_modelo_no_ve_el_usuario_ni_el_runtime() -> None:
    """Mismo control que en PB-015, ahora para las tools que escriben."""
    herramientas = {h.name: h for h in construir_herramientas(CalendarioFalso())}

    for nombre, esperados in [
        ("crear_evento_en_calendario", {"titulo", "fecha", "hora_inicio", "duracion_minutos"}),
        ("eliminar_evento_del_calendario", {"fecha", "titulo"}),
    ]:
        esquema = herramientas[nombre].tool_call_schema
        assert isinstance(esquema, type) and issubclass(esquema, BaseModel)
        json_schema = esquema.model_json_schema()
        assert set(json_schema["properties"]) == esperados, nombre
        assert "runtime" not in json.dumps(json_schema)


# --- Validación de entradas del modelo ---------------------------------------


async def test_una_fecha_invalida_no_llega_ni_al_interrupt() -> None:
    calendario = CalendarioFalso()
    pedido = AIMessage(
        "",
        tool_calls=[
            {
                "name": "crear_evento_en_calendario",
                "args": {"titulo": "X", "fecha": "mañana", "hora_inicio": "10:00"},
                "id": "t1",
            }
        ],
    )
    grafo = _grafo_con(calendario, pedido, AIMessage("cierro"))

    estado = await _preguntar(grafo, "agendame algo mañana")

    assert "__interrupt__" not in estado  # no hay nada válido que confirmar
    assert calendario.creados == []


async def test_una_duracion_absurda_se_rechaza() -> None:
    calendario = CalendarioFalso()
    pedido = AIMessage(
        "",
        tool_calls=[
            {
                "name": "crear_evento_en_calendario",
                "args": {
                    "titulo": "X",
                    "fecha": "2026-09-05",
                    "hora_inicio": "10:00",
                    "duracion_minutos": 5000,
                },
                "id": "t1",
            }
        ],
    )
    grafo = _grafo_con(calendario, pedido, AIMessage("cierro"))

    estado = await _preguntar(grafo, "agendame algo")

    assert "__interrupt__" not in estado
    assert calendario.creados == []


# --- Eliminar: la desambiguación es nuestra ----------------------------------


def _evento(titulo: str, hora: int, id_: str) -> Evento:
    inicio = datetime(2026, 9, 5, hora, 0, tzinfo=UTC)
    return Evento(titulo=titulo, inicio=inicio, fin=inicio + timedelta(hours=1), id=id_)


def _pedido_eliminar(titulo: str = "dentista") -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {
                "name": "eliminar_evento_del_calendario",
                "args": {"fecha": "2026-09-05", "titulo": titulo},
                "id": "t1",
            }
        ],
    )


async def test_eliminar_con_cero_coincidencias_no_interrumpe_ni_borra() -> None:
    calendario = CalendarioFalso(eventos=())
    grafo = _grafo_con(calendario, _pedido_eliminar(), AIMessage("no encontré"))

    estado = await _preguntar(grafo, "borrá el dentista")

    assert "__interrupt__" not in estado
    assert calendario.eliminados == []


async def test_eliminar_con_varias_coincidencias_pide_precision() -> None:
    """Nunca se borra por adivinanza: con 2+ candidatos no hay interrupt."""
    calendario = CalendarioFalso(
        eventos=(_evento("Dentista Norte", 10, "id-a"), _evento("Dentista Sur", 15, "id-b"))
    )
    grafo = _grafo_con(calendario, _pedido_eliminar(), AIMessage("¿cuál?"))

    estado = await _preguntar(grafo, "borrá el dentista")

    assert "__interrupt__" not in estado
    assert calendario.eliminados == []


async def test_eliminar_con_una_coincidencia_confirma_y_borra() -> None:
    calendario = CalendarioFalso(eventos=(_evento("Dentista", 10, "id-unico"),))
    grafo = _grafo_con(calendario, _pedido_eliminar(), AIMessage("Eliminado."))

    estado = await _preguntar(grafo, "borrá el dentista")
    assert "__interrupt__" in estado
    assert "Dentista" in estado["__interrupt__"][0].value["resumen"]
    assert calendario.eliminados == []  # todavía no

    await _reanudar(grafo, aprobado=True)
    assert calendario.eliminados == [(USUARIO, "id-unico")]


async def test_el_matching_de_titulos_ignora_tildes_y_mayusculas() -> None:
    calendario = CalendarioFalso(eventos=(_evento("Cumpleaños de Mamá", 12, "id-c"),))
    grafo = _grafo_con(calendario, _pedido_eliminar("cumpleanos"), AIMessage("ok"))

    estado = await _preguntar(grafo, "borrá el cumpleaños")

    assert "__interrupt__" in estado


# --- Privacidad (RF-18) ------------------------------------------------------


async def test_el_titulo_del_evento_no_se_loguea() -> None:
    calendario = CalendarioFalso()
    pedido = AIMessage(
        "",
        tool_calls=[
            {
                "name": "crear_evento_en_calendario",
                "args": {
                    "titulo": "Terapia con la Dra. Pérez",
                    "fecha": "2026-09-05",
                    "hora_inicio": "10:00",
                },
                "id": "t1",
            }
        ],
    )
    grafo = _grafo_con(calendario, pedido, AIMessage("listo"))

    with structlog.testing.capture_logs() as eventos:
        await _preguntar(grafo, "agendame terapia")
        await _reanudar(grafo, aprobado=True)

    # Sin esto la aserción negativa de abajo pasaría aunque `capture_logs`
    # no hubiera capturado nada. Ver `test_logging.py`, sección de la trampa.
    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "Terapia" not in registrado
    assert "Pérez" not in registrado
