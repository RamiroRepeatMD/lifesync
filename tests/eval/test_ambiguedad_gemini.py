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
from src.domain.entities.tarea import Tarea
from src.domain.exceptions import AgenteNoDisponibleError
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
    tareas = TareasFalsas(pendientes=(Tarea(titulo="Pagar la luz", id="t-luz"),))
    yield crear_agente_gemini(settings, calendario, tareas), calendario, tareas


INTENTOS_ANTE_TRANSITORIOS = 3


async def _turno(agente: AgenteGemini, texto: str) -> str:
    """Un turno en un hilo limpio, separando dos clases de fallo.

    Que el modelo conteste mal es un FALLO de esta eval. Que Gemini esté caído
    (504, ReadTimeout, cuota: frecuentes en el plan gratuito) no dice nada
    sobre el comportamiento: se reintenta, y si persiste el caso se marca
    SKIP. Sin esta distinción la eval fallaría los días que el proveedor anda
    mal, y ese fallo no significaría nada.
    """
    ultimo: Exception | None = None
    for _ in range(INTENTOS_ANTE_TRANSITORIOS):
        identificador = uuid4()  # hilo nuevo por intento: sin contaminación
        try:
            return await agente.responder(
                ConsultaDelUsuario(
                    conversacion_id=identificador, usuario_id=identificador, texto=texto
                )
            )
        except AgenteNoDisponibleError as exc:
            ultimo = exc
    pytest.skip(f"Gemini no respondió tras {INTENTOS_ANTE_TRANSITORIOS} intentos: {ultimo}")


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


async def test_si_ya_la_hizo_se_completa_no_se_elimina(
    agente: tuple[AgenteGemini, CalendarioFalso, TareasFalsas],
) -> None:
    """La frontera nueva de PB-029: hacerla es completar, no borrar."""
    modelo, _, _ = agente

    respuesta = await _turno(modelo, "ya pagué la luz")

    assert "Marcar como hecha" in respuesta
    assert "Eliminar" not in respuesta


async def test_si_ya_no_hace_falta_se_elimina_no_se_completa(
    agente: tuple[AgenteGemini, CalendarioFalso, TareasFalsas],
) -> None:
    modelo, _, _ = agente

    respuesta = await _turno(modelo, "borrá la tarea de la luz, ya no hace falta")

    assert "Eliminar la tarea" in respuesta
    assert "Marcar como hecha" not in respuesta
