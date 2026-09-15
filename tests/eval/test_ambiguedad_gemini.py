"""Evaluación del manejo de ambigüedad contra el modelo real (PB-014, RF-10).

RF-10 es comportamiento del modelo: no se puede afirmar con dobles. Esta suite
es la que lo hace **exigible** — corre contra Gemini de verdad, así que:

- Lleva el marker `gemini` y se saltea sin `GOOGLE_API_KEY`: CI sigue sin red.
- Gasta cuota real (~10 peticiones por corrida): no correrla en loop.
- Las aserciones son estructurales (¿preguntó? ¿propuso confirmar?), nunca de
  texto exacto: el modelo redacta distinto cada vez y eso está bien.

Los caminos determinísticos de RF-10 (rangos inválidos, 0/2+ coincidencias)
viven en los tests unitarios de las herramientas; acá se evalúa la parte que
decide el modelo: darse cuenta de que falta un dato ANTES de llamar la tool.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.domain.entities.evento import Evento
from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.llm.agente_gemini import AgenteGemini, crear_agente_gemini
from tests.dobles import CalendarioFalso, TareasFalsas

pytestmark = [
    pytest.mark.gemini,
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="evaluación contra Gemini real: necesita GOOGLE_API_KEY",
    ),
]

SENAL_DE_CONFIRMACION = "¿Confirmás?"


@pytest_asyncio.fixture
async def agente() -> AsyncIterator[tuple[AgenteGemini, CalendarioFalso, TareasFalsas]]:
    """Agente real (modelo de verdad) con calendario falso y memoria limpia."""
    manana = datetime.now(UTC) + timedelta(days=1)
    calendario = CalendarioFalso(
        eventos=(
            Evento(
                titulo="Dentista",
                inicio=manana.replace(hour=13, minute=0),
                fin=manana.replace(hour=14, minute=0),
                id="id-dentista",
            ),
        )
    )
    settings = Settings(
        _env_file=None,
        environment=Environment.TESTING,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    tareas = TareasFalsas()
    yield crear_agente_gemini(settings, calendario, tareas), calendario, tareas


async def _turno(agente: AgenteGemini, texto: str) -> str:
    identificador = uuid4()  # hilo nuevo por caso: sin contaminación entre tests
    return await agente.responder(
        ConsultaDelUsuario(conversacion_id=identificador, usuario_id=identificador, texto=texto)
    )


async def test_crear_sin_hora_pregunta_en_vez_de_inventar(
    agente: tuple[AgenteGemini, CalendarioFalso, TareasFalsas],
) -> None:
    modelo, calendario, _ = agente

    respuesta = await _turno(modelo, "agendame una reunión mañana")

    assert SENAL_DE_CONFIRMACION not in respuesta  # no propuso crear nada
    assert "?" in respuesta  # preguntó
    assert calendario.creados == []


async def test_eliminar_sin_dia_pregunta_cual(
    agente: tuple[AgenteGemini, CalendarioFalso, TareasFalsas],
) -> None:
    modelo, calendario, _ = agente

    respuesta = await _turno(modelo, "borrá la reunión")

    assert SENAL_DE_CONFIRMACION not in respuesta
    assert "?" in respuesta
    assert calendario.eliminados == []


async def test_modificar_sin_el_dato_nuevo_pregunta(
    agente: tuple[AgenteGemini, CalendarioFalso, TareasFalsas],
) -> None:
    modelo, calendario, _ = agente

    respuesta = await _turno(modelo, "cambiale la hora al dentista de mañana")

    assert SENAL_DE_CONFIRMACION not in respuesta
    assert "?" in respuesta
    assert calendario.modificados == []


async def test_un_pedido_completo_no_sobre_pregunta(
    agente: tuple[AgenteGemini, CalendarioFalso, TareasFalsas],
) -> None:
    """El control del otro lado: con todos los datos, va directo a confirmar."""
    modelo, _, _ = agente

    respuesta = await _turno(modelo, "agendame dentista mañana a las 15:00")

    assert SENAL_DE_CONFIRMACION in respuesta


async def test_tengo_que_sin_hora_es_tarea_y_no_evento(
    agente: tuple[AgenteGemini, CalendarioFalso, TareasFalsas],
) -> None:
    """El criterio nuevo de PB-028: sin hora, es una tarea."""
    modelo, calendario, _ = agente

    respuesta = await _turno(modelo, "acordate que tengo que llamar al banco")

    assert calendario.creados == []  # NO fue al calendario
    # Puede proponer la tarea (¿Confirmás?) o repreguntar; ambas son válidas.
    # Lo inválido es haber creado un evento o no haber hecho nada con sentido.
    assert SENAL_DE_CONFIRMACION in respuesta or "?" in respuesta
