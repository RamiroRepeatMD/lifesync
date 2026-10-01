"""Casos límite de fechas (PB-027).

Fijan lo que la auditoría de PB-027 encontró correcto —para que no se rompa
sin aviso— y cubren los bugs que encontró, con el test escrito antes del
arreglo.

Las fechas van relativas a hoy en la zona local, salvo donde el caso ES la
fecha (bisiestos, días de la semana conocidos): con fechas fijas, un test de
"no se puede crear en el pasado" se rompería solo el día que el calendario la
pase.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.domain.entities.evento import Evento
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import (
    _aplicar_cambios,
    _en_hora_local,
    _linea,
    _momento_local,
    _rango,
    _redactar,
    construir_herramientas,
    fecha_en_palabras,
)
from src.infrastructure.llm.prompt import instrucciones
from tests.dobles import CalendarioFalso, ModeloFalso, TareasFalsas

USUARIO = uuid4()


def _local(anio: int, mes: int, dia: int) -> datetime:
    return datetime(anio, mes, dia, tzinfo=ZONA_HORARIA)


# --- Horas y fechas que manda el modelo ---------------------------------------


@pytest.mark.parametrize(
    ("hora", "esperada"),
    [("9:05", (9, 5)), ("09:05", (9, 5)), ("00:00", (0, 0)), ("23:59", (23, 59))],
)
def test_horas_validas(hora: str, esperada: tuple[int, int]) -> None:
    momento = _momento_local("2026-10-01", hora)

    assert momento is not None
    assert (momento.hour, momento.minute) == esperada


@pytest.mark.parametrize("hora", ["24:00", "23:60", "9", "10:30:00", "10.30", "-1:00", "", "diez"])
def test_horas_invalidas_no_explotan(hora: str) -> None:
    assert _momento_local("2026-10-01", hora) is None


@pytest.mark.parametrize("fecha", ["2026-02-30", "2027-02-29", "30/09/2026", "", "mañana"])
def test_fechas_invalidas_no_explotan(fecha: str) -> None:
    assert _momento_local(fecha, "10:00") is None


def test_el_29_de_febrero_existe_en_bisiestos() -> None:
    assert _momento_local("2028-02-29", "10:00") is not None


def test_el_momento_local_lleva_el_huso_argentino() -> None:
    momento = _momento_local("2026-10-01", "23:30")

    assert momento is not None
    assert momento.utcoffset() == timedelta(hours=-3)


# --- Rangos --------------------------------------------------------------------


def test_un_dia_va_de_medianoche_local_a_medianoche_local() -> None:
    rango = _rango("2026-10-01", "2026-10-01")

    assert rango == (_local(2026, 10, 1), _local(2026, 10, 2))


@pytest.mark.parametrize(
    ("desde", "hasta", "fin"),
    [
        ("2026-09-30", "2026-10-01", _local(2026, 10, 2)),  # borde de mes
        ("2026-12-31", "2027-01-01", _local(2027, 1, 2)),  # borde de año
    ],
)
def test_los_rangos_cruzan_meses_y_años(desde: str, hasta: str, fin: datetime) -> None:
    rango = _rango(desde, hasta)

    assert rango is not None
    assert rango[1] == fin


def test_el_rango_maximo_es_de_31_dias() -> None:
    assert _rango("2026-10-01", "2026-11-01") is not None  # 31 días
    assert _rango("2026-10-01", "2026-11-02") is None  # 32


def test_un_rango_al_reves_no_vale() -> None:
    assert _rango("2026-10-02", "2026-10-01") is None


# --- Días de la semana ------------------------------------------------------------


@pytest.mark.parametrize(
    ("dia", "palabras"),
    [
        (date(2026, 10, 1), "jueves 1 de octubre"),
        (date(2027, 1, 1), "viernes 1 de enero"),
        (date(2028, 2, 29), "martes 29 de febrero"),
    ],
)
def test_dias_de_la_semana_en_fechas_conocidas(dia: date, palabras: str) -> None:
    assert fecha_en_palabras(datetime(dia.year, dia.month, dia.day, tzinfo=UTC)) == palabras


# --- La frontera del día (UTC-3) -----------------------------------------------------


def test_a_las_2330_locales_el_prompt_sigue_diciendo_hoy() -> None:
    """Las 23:30 del 30/09 en Argentina ya son el 01/10 en UTC."""
    texto = instrucciones(datetime(2026, 10, 1, 2, 30, tzinfo=UTC))

    (hoy,) = [linea for linea in texto.splitlines() if linea.startswith("Hoy es")]
    assert "miércoles 30 de septiembre de 2026" in hoy
    assert "jueves" not in hoy  # el jueves 1 aparece en la tabla de próximos días, no como hoy
    assert "23:30" in hoy


def _cena_2330() -> Evento:
    """Jueves 1/10 a las 23:30 locales = viernes 2/10 a las 02:30 UTC."""
    inicio = datetime(2026, 10, 2, 2, 30, tzinfo=UTC)
    return Evento(titulo="Cena", inicio=inicio, fin=inicio + timedelta(hours=1), id="id-cena")


def test_un_evento_de_las_2330_se_lista_en_su_dia_local() -> None:
    texto = _redactar([_cena_2330()], _local(2026, 10, 1))

    assert texto.startswith("jueves 1 de octubre:")


def test_un_evento_que_cruza_la_medianoche_lo_dice() -> None:
    evento = _cena_2330()

    linea = _linea(evento, _en_hora_local(evento))

    assert "23:30 a 00:30 del viernes 2 de octubre" in linea


def test_un_evento_del_mismo_dia_no_agrega_la_fecha_al_fin() -> None:
    inicio = datetime(2026, 10, 1, 13, 0, tzinfo=UTC)  # 10:00 locales
    evento = Evento(titulo="Dentista", inicio=inicio, fin=inicio + timedelta(hours=1))

    assert "10:00 a 11:00 —" in _linea(evento, _en_hora_local(evento))


# --- Día completo: fechas, no instantes ------------------------------------------------


def _vacaciones(fin: bool = True) -> Evento:
    """Del 5 al 7 de octubre: Google marca el fin, exclusivo, el 8."""
    return Evento(
        titulo="Vacaciones",
        inicio=datetime(2026, 10, 5, tzinfo=UTC),
        fin=datetime(2026, 10, 8, tzinfo=UTC) if fin else None,
        todo_el_dia=True,
        id="id-vacaciones",
    )


def test_mover_un_dia_completo_corre_las_dos_puntas() -> None:
    movido = _aplicar_cambios(_vacaciones(), "", "2026-10-12", "", 0)

    assert movido is not None
    assert movido.todo_el_dia
    assert movido.inicio == datetime(2026, 10, 12, tzinfo=UTC)
    assert movido.fin == datetime(2026, 10, 15, tzinfo=UTC)  # siguen siendo 3 días


def test_renombrar_un_dia_completo_no_toca_las_fechas() -> None:
    original = _vacaciones()

    renombrado = _aplicar_cambios(original, "Vacaciones en Bariloche", "", "", 0)

    assert renombrado is not None
    assert (renombrado.inicio, renombrado.fin) == (original.inicio, original.fin)


def test_mover_un_dia_completo_sin_fin_conocido_asume_un_dia() -> None:
    movido = _aplicar_cambios(_vacaciones(fin=False), "", "2026-10-12", "", 0)

    assert movido is not None
    assert movido.fin == datetime(2026, 10, 13, tzinfo=UTC)


def test_darle_hora_a_un_dia_completo_lo_vuelve_de_una_hora() -> None:
    con_hora = _aplicar_cambios(_vacaciones(), "", "", "15:00", 0)

    assert con_hora is not None
    assert not con_hora.todo_el_dia
    assert con_hora.fin is not None
    assert con_hora.fin - con_hora.inicio == timedelta(hours=1)


# --- Crear en el pasado --------------------------------------------------------------------


async def _correr(tool: str, args: dict[str, Any], tareas: TareasFalsas | None = None) -> Any:
    calendario = CalendarioFalso()
    tareas = tareas or TareasFalsas()
    modelo = ModeloFalso(
        guion=[
            AIMessage("", tool_calls=[{"name": tool, "args": args, "id": "c1"}]),
            AIMessage("ok"),
        ]
    )
    grafo = construir_grafo(modelo, construir_herramientas(calendario, tareas), InMemorySaver())
    estado = await grafo.ainvoke(
        {"messages": [HumanMessage("pedido")]},
        config={"configurable": {"thread_id": "hilo"}},
        context=ContextoDeAgente(usuario_id=USUARIO),
    )
    resultado = next((m for m in estado["messages"] if isinstance(m, ToolMessage)), None)
    return estado, calendario, tareas, resultado


async def test_crear_un_evento_en_el_pasado_no_propone_y_nombra_el_año() -> None:
    """El caso típico: "el 5 de enero" inferido en el año que ya pasó."""
    ayer = datetime.now(ZONA_HORARIA).date() - timedelta(days=1)

    estado, calendario, _, resultado = await _correr(
        "crear_evento_en_calendario",
        {"titulo": "Dentista", "fecha": ayer.isoformat(), "hora_inicio": "10:00"},
    )

    assert "__interrupt__" not in estado
    assert calendario.creados == []
    assert resultado is not None and str(ayer.year) in str(resultado.content)


async def test_crear_hoy_a_una_hora_que_ya_paso_tampoco() -> None:
    hace_un_rato = datetime.now(ZONA_HORARIA) - timedelta(hours=2)

    estado, calendario, _, _ = await _correr(
        "crear_evento_en_calendario",
        {
            "titulo": "Dentista",
            "fecha": hace_un_rato.date().isoformat(),
            "hora_inicio": hace_un_rato.strftime("%H:%M"),
        },
    )

    assert "__interrupt__" not in estado
    assert calendario.creados == []


async def test_crear_mas_tarde_si_se_crea() -> None:
    en_un_rato = datetime.now(ZONA_HORARIA) + timedelta(hours=2)

    _, calendario, _, _ = await _correr(
        "crear_evento_en_calendario",
        {
            "titulo": "Dentista",
            "fecha": en_un_rato.date().isoformat(),
            "hora_inicio": en_un_rato.strftime("%H:%M"),
        },
    )

    assert len(calendario.creados) == 1


async def test_crear_avisa_si_termina_al_dia_siguiente() -> None:
    manana = datetime.now(ZONA_HORARIA).date() + timedelta(days=1)
    pasado = manana + timedelta(days=1)

    _, _, _, resultado = await _correr(
        "crear_evento_en_calendario",
        {"titulo": "Cena", "fecha": manana.isoformat(), "hora_inicio": "23:30"},
    )

    dia_siguiente = fecha_en_palabras(datetime(pasado.year, pasado.month, pasado.day, tzinfo=UTC))
    assert f"a 00:30 del {dia_siguiente}" in str(resultado.content)


async def test_anotar_una_tarea_para_ayer_no_propone() -> None:
    ayer = datetime.now(ZONA_HORARIA).date() - timedelta(days=1)

    estado, _, tareas, resultado = await _correr(
        "crear_tarea", {"titulo": "Pagar la luz", "fecha_limite": ayer.isoformat()}
    )

    assert "__interrupt__" not in estado
    assert tareas.creadas == []
    assert resultado is not None and str(ayer.year) in str(resultado.content)


async def test_anotar_una_tarea_para_hoy_si_se_anota() -> None:
    hoy = datetime.now(ZONA_HORARIA).date()

    _, _, tareas, _ = await _correr(
        "crear_tarea", {"titulo": "Pagar la luz", "fecha_limite": hoy.isoformat()}
    )

    assert len(tareas.creadas) == 1


def test_el_prompt_trae_los_proximos_siete_dias_ya_resueltos() -> None:
    """Con pedidos compuestos, el modelo propuso "el lunes" un mes tarde: se le da la tabla."""
    texto = instrucciones(datetime(2026, 9, 30, 20, 0, tzinfo=UTC))  # miércoles 30/09, 17:00

    assert "lunes 5 de octubre (2026-10-05)" in texto
    assert "jueves 1 de octubre (2026-10-01)" in texto
    assert "miércoles 7 de octubre (2026-10-07)" in texto
    assert "(2026-10-08)" not in texto  # siete días, no más
