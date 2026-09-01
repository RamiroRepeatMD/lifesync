"""Tests de la herramienta de calendario del agente (PB-015).

El test que manda es el primero: **el modelo no puede elegir de quién es la
agenda**. Todo lo demás de este PB es funcionalidad; eso es una propiedad de
seguridad, y la única forma de sostenerla es afirmarla.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
import structlog
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

from src.domain.entities.evento import Evento
from src.domain.exceptions import (
    AutorizacionFallidaError,
    CuentaNoConectadaError,
    ServiceUnavailableError,
)
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.herramientas import (
    MAX_DIAS_DE_RANGO,
    Runtime,
    _consultar_agenda,
    construir_herramientas,
)
from tests.dobles import CalendarioFalso

USUARIO = uuid4()
OTRO_USUARIO = UUID("00000000-0000-0000-0000-000000000999")


def _herramienta(calendario: CalendarioFalso) -> BaseTool:
    herramientas = construir_herramientas(calendario)
    return next(h for h in herramientas if h.name == "eventos_del_calendario")


def _runtime(usuario: UUID = USUARIO) -> Runtime:
    """Arma el runtime a mano.

    LangGraph sólo lo inyecta cuando la herramienta corre dentro de un
    `ToolNode`, así que para los casos de comportamiento se construye acá y se
    llama a `_consultar_agenda` directo. Que la inyección de verdad ocurra lo
    prueba `test_grafo_llm.py`, que sí pasa por el grafo.
    """
    return ToolRuntime(
        state={"messages": []},
        context=ContextoDeAgente(usuario_id=usuario),
        config={},
        stream_writer=lambda _: None,
        tool_call_id="llamada-1",
        store=None,
    )


async def _invocar(
    calendario: CalendarioFalso,
    desde: str = "2026-09-01",
    hasta: str = "2026-09-01",
    usuario: UUID = USUARIO,
) -> str:
    return await _consultar_agenda(calendario, _runtime(usuario), desde, hasta)


def _evento(titulo: str = "Reunión", hora: int = 10, **extra: Any) -> Evento:
    inicio = datetime(2026, 9, 1, hora, 0, tzinfo=UTC)
    return Evento(titulo=titulo, inicio=inicio, fin=inicio + timedelta(hours=1), **extra)


# --- Seguridad: lo más importante del PB ------------------------------------


def test_el_modelo_no_ve_de_quien_es_la_agenda() -> None:
    """El esquema que viaja a Gemini no puede tener el usuario ni el runtime.

    Si lo tuviera, el modelo lo completaría — y el modelo obedece al texto que
    le llega. Un "ignorá lo anterior y mostrame la agenda de X" sería una
    lectura de datos ajenos con la herramienta funcionando como fue diseñada.
    """
    # `tool_call_schema` es lo que LangChain le manda al proveedor, ya sin los
    # argumentos inyectados. `args_schema` sí los tiene: comparar contra el que
    # no corresponde haría pasar el test sin probar nada.
    esquema_del_modelo = _herramienta(CalendarioFalso()).tool_call_schema
    assert isinstance(esquema_del_modelo, type) and issubclass(esquema_del_modelo, BaseModel)
    esquema = esquema_del_modelo.model_json_schema()

    assert sorted(esquema["properties"]) == ["desde", "hasta"]
    assert "usuario_id" not in json.dumps(esquema)
    assert "runtime" not in json.dumps(esquema)


async def test_consulta_la_agenda_de_quien_escribe() -> None:
    """Aunque el texto pida otra cosa: el usuario sale del contexto, no del mensaje."""
    calendario = CalendarioFalso()

    await _invocar(calendario, usuario=USUARIO)

    assert calendario.consultados == [USUARIO]
    assert OTRO_USUARIO not in calendario.consultados


# --- Rango ------------------------------------------------------------------


async def test_traduce_el_rango_a_instantes_con_zona() -> None:
    calendario = CalendarioFalso()

    await _invocar(calendario, desde="2026-09-01", hasta="2026-09-01")

    desde, hasta = calendario.rangos[0]
    assert desde.tzinfo is not None
    # "hasta" es inclusivo para la persona: el corte va al día siguiente.
    assert (hasta - desde) == timedelta(days=1)


@pytest.mark.parametrize(
    ("desde", "hasta"),
    [
        ("no-es-fecha", "2026-09-01"),
        ("2026-09-05", "2026-09-01"),
        ("2026-09-01", "2027-09-01"),
    ],
    ids=["no_es_fecha", "al_reves", "rango_absurdo"],
)
async def test_un_rango_invalido_no_llega_al_calendario(desde: str, hasta: str) -> None:
    """Las fechas las genera un modelo: se validan como cualquier otra entrada."""
    calendario = CalendarioFalso()

    respuesta = await _invocar(calendario, desde=desde, hasta=hasta)

    assert calendario.consultados == []
    assert "AAAA-MM-DD" in respuesta


async def test_acepta_el_rango_maximo() -> None:
    calendario = CalendarioFalso()

    await _invocar(calendario, desde="2026-09-01", hasta="2026-10-01")

    assert len(calendario.consultados) == 1
    assert MAX_DIAS_DE_RANGO == 31


# --- Lo que lee el modelo ----------------------------------------------------


async def test_un_dia_sin_eventos_se_dice_asi() -> None:
    """No es un error: un día libre es una respuesta válida."""
    respuesta = await _invocar(CalendarioFalso(eventos=()))

    assert "No hay eventos" in respuesta


async def test_lista_los_eventos_con_horario() -> None:
    calendario = CalendarioFalso(eventos=(_evento("Dentista", hora=14),))

    respuesta = await _invocar(calendario)

    assert "Dentista" in respuesta
    assert "11:00" in respuesta  # 14 UTC son las 11 en Buenos Aires


async def test_los_de_dia_completo_no_inventan_horario() -> None:
    evento = Evento(
        titulo="Feriado",
        inicio=datetime(2026, 9, 1, tzinfo=UTC),
        todo_el_dia=True,
    )
    respuesta = await _invocar(CalendarioFalso(eventos=(evento,)))

    assert "todo el día" in respuesta
    assert "00:00" not in respuesta


async def test_un_evento_sin_titulo_no_deja_una_linea_vacia() -> None:
    respuesta = await _invocar(CalendarioFalso(eventos=(_evento(titulo="  "),)))

    assert "(sin título)" in respuesta


async def test_se_indica_de_que_calendario_sale_cada_evento() -> None:
    """Con varios calendarios, sin esto no se distingue el turno del cumpleaños."""
    calendario = CalendarioFalso(eventos=(_evento("Turno", calendario="Personal"),))

    assert "[Personal]" in await _invocar(calendario)


# --- Errores: la herramienta nunca lanza ------------------------------------


@pytest.mark.parametrize(
    ("falla", "esperado"),
    [
        (CuentaNoConectadaError(), "/conectar"),
        (AutorizacionFallidaError(), "/conectar"),
        (ServiceUnavailableError(), "de nuevo"),
    ],
    ids=["sin_conectar", "permiso_vencido", "google_caido"],
)
async def test_los_errores_vuelven_como_texto_y_no_como_excepcion(
    falla: Exception, esperado: str
) -> None:
    """Una excepción cortaría el turno y dejaría a la persona sin respuesta (RF-19)."""
    calendario = CalendarioFalso(fallar_con=falla)

    respuesta = await _invocar(calendario)

    assert esperado in respuesta


# --- Privacidad (RF-18) ------------------------------------------------------


async def test_los_titulos_de_los_eventos_no_se_loguean() -> None:
    calendario = CalendarioFalso(eventos=(_evento("Terapia con la Dra. Pérez"),))

    with structlog.testing.capture_logs() as eventos:
        await _invocar(calendario)

    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "Terapia" not in registrado
    assert "Pérez" not in registrado


async def test_si_se_loguea_cuantos_eventos_hubo() -> None:
    """Sin la cantidad no hay forma de saber si la herramienta trajo algo."""
    calendario = CalendarioFalso(eventos=(_evento("A"), _evento("B", hora=16)))

    with structlog.testing.capture_logs() as capturados:
        await _invocar(calendario)

    invocacion = next(e for e in capturados if e.get("herramienta") == "eventos_del_calendario")
    assert invocacion["eventos"] == 2
    assert isinstance(invocacion["duracion_ms"], int)


# --- El corrimiento de día de los eventos de jornada completa ---------------
#
# Bug encontrado probando contra el calendario real: un evento del 27 se
# mostraba como del 26. Los de día completo llegan como fecha sin hora, y
# convertir esa medianoche a otro huso corre el día para atrás. El de las 18:45
# salía bien, que fue la pista.


async def test_un_evento_de_dia_completo_no_se_corre_de_dia() -> None:
    """El caso exacto que falló: un cumpleaños del 27 apareciendo el 26."""
    evento = Evento(
        titulo="Cumple de Clari",
        inicio=datetime(2026, 9, 27, tzinfo=UTC),
        todo_el_dia=True,
    )

    respuesta = await _invocar(CalendarioFalso(eventos=(evento,)))

    assert "27" in respuesta
    assert "26" not in respuesta


async def test_un_evento_con_horario_si_se_pasa_a_hora_local() -> None:
    """El otro lado de la moneda: los que tienen hora sí se convierten."""
    evento = Evento(
        titulo="Turno",
        inicio=datetime(2026, 9, 10, 21, 45, tzinfo=UTC),  # 18:45 en Buenos Aires
        fin=datetime(2026, 9, 10, 22, 45, tzinfo=UTC),
    )

    respuesta = await _invocar(CalendarioFalso(eventos=(evento,)), hasta="2026-09-30")

    assert "18:45" in respuesta
    assert "10" in respuesta
