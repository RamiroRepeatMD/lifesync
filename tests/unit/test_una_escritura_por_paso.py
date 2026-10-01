"""Una escritura por paso del grafo (bug de la prueba real del 30/09).

Gemini pidió "crear evento" y "crear tarea" **en un mismo mensaje**. Las dos
corrían en el mismo paso del grafo, y LangGraph re-ejecuta el paso entero al
reanudar cada pausa: al aprobar la tarea, la creación del evento volvía a
correr con su aprobación guardada. Resultado en producción: el evento, dos
veces. Con correos, el mismo mecanismo enviaría uno dos veces.

Los tests de PB-026 no lo vieron porque el modelo falso pedía las acciones de
a una por mensaje. Éstos modelan lo que Gemini hizo de verdad.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.domain.entities.tarea import Tarea
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.llm.agente_gemini import AgenteGemini
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import (
    CalendarioFalso,
    CorreosFalsos,
    ModeloFalso,
    RecordatoriosEnMemoria,
    TareasFalsas,
)

USUARIO = uuid4()
LUNES = (datetime.now(ZONA_HORARIA) + timedelta(days=5)).date().isoformat()


def _llamadas(*pedidos: tuple[str, str, dict[str, Any]]) -> AIMessage:
    """Un mensaje del modelo con varias llamadas a la vez, como hace Gemini."""
    return AIMessage(
        "", tool_calls=[{"name": nombre, "id": id_, "args": args} for nombre, id_, args in pedidos]
    )


EVENTO = (
    "crear_evento_en_calendario",
    "c1",
    {"titulo": "Dentista", "fecha": LUNES, "hora_inicio": "16:00"},
)
TAREA = ("crear_tarea", "c2", {"titulo": "comprar el regalo de mamá"})


class Bot:
    """El agente de verdad (grafo, tools, adaptador) con modelo y puertos falsos."""

    def __init__(self, *guion: AIMessage, tareas: TareasFalsas | None = None) -> None:
        self.calendario = CalendarioFalso()
        self.tareas = tareas or TareasFalsas()
        self.correos = CorreosFalsos()
        self.modelo = ModeloFalso(guion=list(guion))
        self.grafo = construir_grafo(
            self.modelo,
            construir_herramientas(self.calendario, self.tareas, self.correos),
            InMemorySaver(),
        )
        self._agente = AgenteGemini(self.grafo)

    async def decir(self, texto: str) -> str:
        return await self._agente.responder(
            ConsultaDelUsuario(conversacion_id=USUARIO, usuario_id=USUARIO, texto=texto)
        )

    async def mensajes(self) -> list[Any]:
        estado = await self.grafo.aget_state({"configurable": {"thread_id": str(USUARIO)}})
        mensajes: list[Any] = estado.values["messages"]
        return mensajes


async def test_lo_que_hizo_gemini_deja_un_evento_y_una_tarea() -> None:
    """El caso exacto de producción: dos escrituras en un mensaje, dos "sí"."""
    repide = _llamadas(("crear_tarea", "c3", {"titulo": "comprar el regalo de mamá"}))
    bot = Bot(_llamadas(EVENTO, TAREA), repide, AIMessage("Listo, las dos cosas."))

    assert (await bot.decir("agendame dentista el lunes y anotá el regalo")).startswith(
        'Crear "Dentista"'
    )
    segunda = await bot.decir("sí")
    await bot.decir("sí")

    assert len(bot.calendario.creados) == 1  # hasta este arreglo: 2
    assert len(bot.tareas.creadas) == 1
    assert segunda.startswith('Listo — Evento creado: "Dentista"')
    assert "Anotar la tarea" in segunda


async def test_dos_correos_en_un_mensaje_no_salen_juntos() -> None:
    """El peor caso del bug: un correo no se des-envía."""
    a_ana = ("enviar_correo", "c1", {"para": "ana@ejemplo.com", "asunto": "Hola", "texto": "A"})
    a_beto = ("enviar_correo", "c2", {"para": "beto@ejemplo.com", "asunto": "Hola", "texto": "B"})
    repide = _llamadas(("enviar_correo", "c3", a_beto[2]))
    bot = Bot(_llamadas(a_ana, a_beto), repide, AIMessage("listo"))

    primera = await bot.decir("mandales un hola a Ana y a Beto")
    segunda = await bot.decir("sí")

    assert "Para: ana@ejemplo.com" in primera
    assert "beto" not in primera  # la segunda no se propuso junto con la primera
    assert len(bot.correos.enviados) == 1  # sólo el que se aprobó
    assert "Para: beto@ejemplo.com" in segunda  # la otra pide su propia confirmación

    await bot.decir("no")
    assert len(bot.correos.enviados) == 1


async def test_las_lecturas_corren_y_la_escritura_espera_su_si() -> None:
    tareas = TareasFalsas(pendientes=(Tarea(titulo="Pagar la luz", id="t1"),))
    bot = Bot(
        _llamadas(("tareas_pendientes", "c1", {}), TAREA),
        AIMessage("listo"),
        tareas=tareas,
    )

    respuesta = await bot.decir("¿qué tengo pendiente? y anotá el regalo")

    assert tareas.consultados  # la lectura corrió
    assert "Anotar la tarea" in respuesta  # la escritura quedó esperando
    assert tareas.creadas == []
    await bot.decir("sí")
    assert len(tareas.creadas) == 1


async def test_toda_llamada_tiene_respuesta_y_el_mensaje_original_queda_intacto() -> None:
    """El mensaje de Gemini lleva firmas por llamada: no se recorta, se responde entero."""
    repide = _llamadas(("crear_tarea", "c3", {"titulo": "comprar el regalo de mamá"}))
    bot = Bot(_llamadas(EVENTO, TAREA), repide, AIMessage("listo"))
    await bot.decir("agendame dentista y anotá el regalo")
    await bot.decir("sí")
    await bot.decir("sí")

    mensajes = await bot.mensajes()
    pedidos = [m for m in mensajes if isinstance(m, AIMessage) and m.tool_calls]
    respondidas = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
    assert len(pedidos[0].tool_calls) == 2  # intacto: con sus dos llamadas
    assert all(c["id"] in respondidas for m in pedidos for c in m.tool_calls)


def test_toda_herramienta_esta_clasificada_una_sola_vez() -> None:
    """Una tool nueva sin clasificar rompe CI: si escribe, tiene que estar en ESCRITURA."""
    from src.infrastructure.llm.herramientas import (
        HERRAMIENTAS_DE_ESCRITURA,
        HERRAMIENTAS_DE_LECTURA,
    )

    todas = {
        h.name
        for h in construir_herramientas(
            CalendarioFalso(), TareasFalsas(), CorreosFalsos(), RecordatoriosEnMemoria()
        )
    }

    assert HERRAMIENTAS_DE_ESCRITURA.isdisjoint(HERRAMIENTAS_DE_LECTURA)
    assert todas == HERRAMIENTAS_DE_ESCRITURA | HERRAMIENTAS_DE_LECTURA
