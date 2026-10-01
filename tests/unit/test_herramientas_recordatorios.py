"""Tests de las herramientas de recordatorios (PB-030): el ciclo RF-08 y la ventana.

Mismo esqueleto que tareas y calendario: se atraviesa el grafo real con un
modelo falso, y las aserciones van sobre el doble del repositorio. Las horas
son relativas a ahora (regla 5 de docs/CLAUDE.md §9): con horas fijas, los
tests se romperían solos cuando el reloj las pase.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.domain.entities.recordatorio import EstadoDeRecordatorio, Recordatorio
from src.domain.exceptions import RepositoryError
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import ModeloFalso, RecordatoriosEnMemoria

USUARIO = uuid4()


def _en(minutos: int) -> datetime:
    """Un momento relativo a ahora, en hora local y redondeado al minuto."""
    ahora = datetime.now(ZONA_HORARIA).replace(second=0, microsecond=0)
    return ahora + timedelta(minutes=minutos)


def _grafo_con(recordatorios: RecordatoriosEnMemoria, *guion: AIMessage) -> Any:
    modelo = ModeloFalso(guion=list(guion))
    herramientas = construir_herramientas(None, recordatorios=recordatorios)
    return construir_grafo(modelo, herramientas, InMemorySaver())


async def _preguntar(grafo: Any, texto: str) -> dict[str, Any]:
    resultado: dict[str, Any] = await grafo.ainvoke(
        {"messages": [HumanMessage(texto)]},
        config={"configurable": {"thread_id": "hilo-1"}},
        context=ContextoDeAgente(usuario_id=USUARIO),
    )
    return resultado


async def _reanudar(grafo: Any, aprobado: bool) -> dict[str, Any]:
    resultado: dict[str, Any] = await grafo.ainvoke(
        Command(resume={"aprobado": aprobado}),
        config={"configurable": {"thread_id": "hilo-1"}},
        context=ContextoDeAgente(usuario_id=USUARIO),
    )
    return resultado


def _pedido(nombre: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": nombre, "args": args, "id": "r1"}])


def _crear(momento: datetime, texto: str = "sacar la pizza") -> AIMessage:
    return _pedido(
        "crear_recordatorio", texto=texto, fecha=f"{momento:%Y-%m-%d}", hora=f"{momento:%H:%M}"
    )


def _lo_que_dijo_la_tool(estado: dict[str, Any]) -> str:
    respuestas = [m for m in estado["messages"] if isinstance(m, ToolMessage)]
    assert respuestas, "la herramienta no devolvió nada"
    return str(respuestas[-1].content)


# --- Crear -------------------------------------------------------------------


async def test_crear_pausa_con_el_resumen_exacto() -> None:
    recordatorios = RecordatoriosEnMemoria()
    momento = _en(20)
    grafo = _grafo_con(recordatorios, _crear(momento), AIMessage("listo"))

    estado = await _preguntar(grafo, "recordame en 20 minutos que saque la pizza")

    resumen = estado["__interrupt__"][0].value["resumen"]
    assert resumen.startswith("Recordarte «sacar la pizza» ")
    assert f"a las {momento:%H:%M}" in resumen
    assert recordatorios.guardados == {}  # nada sin el sí


async def test_crear_aprobado_guarda_exactamente_el_momento_confirmado() -> None:
    """RF-08: lo que se confirma es lo que se guarda, aunque al reanudar se re-ejecute."""
    recordatorios = RecordatoriosEnMemoria()
    momento = _en(20)
    grafo = _grafo_con(recordatorios, _crear(momento), AIMessage("listo"))

    await _preguntar(grafo, "recordame en 20 minutos que saque la pizza")
    await _reanudar(grafo, aprobado=True)

    [guardado] = recordatorios.guardados.values()
    assert guardado.texto == "sacar la pizza"
    assert guardado.momento == momento
    assert guardado.momento.tzinfo is not None
    assert guardado.usuario_id == USUARIO
    assert guardado.estado is EstadoDeRecordatorio.PENDIENTE


async def test_crear_rechazado_no_guarda_nada() -> None:
    recordatorios = RecordatoriosEnMemoria()
    grafo = _grafo_con(recordatorios, _crear(_en(20)), AIMessage("ok"))

    await _preguntar(grafo, "recordame en 20 minutos que saque la pizza")
    await _reanudar(grafo, aprobado=False)

    assert recordatorios.guardados == {}


async def test_para_manana_lo_dice_con_el_dia_en_palabras() -> None:
    recordatorios = RecordatoriosEnMemoria()
    # Mañana a esta hora menos un rato: siempre dentro de la ventana.
    momento = _en(24 * 60 - 120)
    grafo = _grafo_con(recordatorios, _crear(momento, "llamar al banco"), AIMessage("ok"))

    estado = await _preguntar(grafo, "recordame mañana que llame al banco")

    resumen = estado["__interrupt__"][0].value["resumen"]
    hoy = datetime.now(ZONA_HORARIA).date()
    if momento.date() == hoy:  # cerca de la medianoche, "mañana" cae hoy mismo
        assert f"hoy a las {momento:%H:%M}" in resumen
    else:
        assert "mañana (" in resumen
        assert f"a las {momento:%H:%M}" in resumen


async def test_en_el_pasado_no_propone_nada() -> None:
    recordatorios = RecordatoriosEnMemoria()
    grafo = _grafo_con(recordatorios, _crear(_en(-30)), AIMessage("ok"))

    estado = await _preguntar(grafo, "recordame que saque la pizza")

    assert "__interrupt__" not in estado
    assert "ya pasaron" in _lo_que_dijo_la_tool(estado)
    assert recordatorios.guardados == {}


async def test_a_mas_de_24_horas_explica_la_ventana_y_ofrece_alternativas() -> None:
    recordatorios = RecordatoriosEnMemoria()
    grafo = _grafo_con(recordatorios, _crear(_en(3 * 24 * 60)), AIMessage("ok"))

    estado = await _preguntar(grafo, "recordame el lunes que pague el alquiler")

    assert "__interrupt__" not in estado
    texto = _lo_que_dijo_la_tool(estado)
    assert "24 horas" in texto
    assert "tarea" in texto
    assert "calendario" in texto
    assert recordatorios.guardados == {}


async def test_justo_pasado_el_margen_de_la_ventana_tampoco() -> None:
    """23 h 50 min es el límite: el despachador necesita unos minutos de margen."""
    recordatorios = RecordatoriosEnMemoria()
    grafo = _grafo_con(recordatorios, _crear(_en(23 * 60 + 52)), AIMessage("ok"))

    estado = await _preguntar(grafo, "recordame mañana a esta hora")

    assert "__interrupt__" not in estado
    assert "24 horas" in _lo_que_dijo_la_tool(estado)


async def test_con_formato_invalido_pide_el_formato() -> None:
    recordatorios = RecordatoriosEnMemoria()
    pedido = _pedido("crear_recordatorio", texto="x", fecha="mañana", hora="a la tarde")
    grafo = _grafo_con(recordatorios, pedido, AIMessage("ok"))

    estado = await _preguntar(grafo, "recordame x")

    assert "__interrupt__" not in estado
    assert "AAAA-MM-DD" in _lo_que_dijo_la_tool(estado)


async def test_sin_texto_pregunta_que_recordar() -> None:
    recordatorios = RecordatoriosEnMemoria()
    grafo = _grafo_con(recordatorios, _crear(_en(20), texto="   "), AIMessage("ok"))

    estado = await _preguntar(grafo, "recordame en 20 minutos")

    assert "__interrupt__" not in estado
    assert "¿Qué querés que te recuerde?" in _lo_que_dijo_la_tool(estado)


async def test_si_la_base_falla_al_guardar_lo_dice_sin_mentir() -> None:
    recordatorios = RecordatoriosEnMemoria()
    grafo = _grafo_con(recordatorios, _crear(_en(20)), AIMessage("ok"))
    await _preguntar(grafo, "recordame en 20 minutos que saque la pizza")

    recordatorios.fallar_con = RepositoryError("base caída")
    estado = await _reanudar(grafo, aprobado=True)

    assert "No pude programar" in _lo_que_dijo_la_tool(estado)


# --- Listar ------------------------------------------------------------------


async def test_listar_muestra_los_pendientes_con_su_hora() -> None:
    momento = _en(45)
    recordatorios = RecordatoriosEnMemoria(
        Recordatorio(usuario_id=USUARIO, texto="sacar la pizza", momento=momento),
        Recordatorio(usuario_id=uuid4(), texto="de otra persona", momento=momento),
    )
    grafo = _grafo_con(recordatorios, _pedido("recordatorios_pendientes"), AIMessage("ok"))

    estado = await _preguntar(grafo, "¿qué recordatorios tengo?")

    texto = _lo_que_dijo_la_tool(estado)
    assert "«sacar la pizza»" in texto
    assert f"a las {momento:%H:%M}" in texto
    assert "otra persona" not in texto  # sólo los de quien pregunta


async def test_sin_pendientes_lo_dice() -> None:
    grafo = _grafo_con(
        RecordatoriosEnMemoria(), _pedido("recordatorios_pendientes"), AIMessage("ok")
    )

    estado = await _preguntar(grafo, "¿qué recordatorios tengo?")

    assert _lo_que_dijo_la_tool(estado) == "No hay recordatorios programados."


# --- Cancelar ----------------------------------------------------------------


def _con_pizza_y_banco() -> RecordatoriosEnMemoria:
    return RecordatoriosEnMemoria(
        Recordatorio(usuario_id=USUARIO, texto="sacar la pizza", momento=_en(20)),
        Recordatorio(usuario_id=USUARIO, texto="llamar al banco", momento=_en(90)),
    )


async def test_cancelar_aprobado_lo_deja_cancelado() -> None:
    recordatorios = _con_pizza_y_banco()
    grafo = _grafo_con(
        recordatorios, _pedido("cancelar_recordatorio", texto="banco"), AIMessage("ok")
    )

    estado = await _preguntar(grafo, "cancelá el del banco")
    assert (
        estado["__interrupt__"][0]
        .value["resumen"]
        .startswith("Cancelar el recordatorio «llamar al banco» de ")
    )
    await _reanudar(grafo, aprobado=True)

    estados = {r.texto: r.estado for r in recordatorios.guardados.values()}
    assert estados == {
        "llamar al banco": EstadoDeRecordatorio.CANCELADO,
        "sacar la pizza": EstadoDeRecordatorio.PENDIENTE,
    }


async def test_cancelar_rechazado_lo_deja_programado() -> None:
    recordatorios = _con_pizza_y_banco()
    grafo = _grafo_con(
        recordatorios, _pedido("cancelar_recordatorio", texto="banco"), AIMessage("ok")
    )

    await _preguntar(grafo, "cancelá el del banco")
    await _reanudar(grafo, aprobado=False)

    assert all(r.estado is EstadoDeRecordatorio.PENDIENTE for r in recordatorios.guardados.values())


async def test_cancelar_sin_coincidencias_no_pausa() -> None:
    grafo = _grafo_con(
        _con_pizza_y_banco(), _pedido("cancelar_recordatorio", texto="dentista"), AIMessage("ok")
    )

    estado = await _preguntar(grafo, "cancelá el del dentista")

    assert "__interrupt__" not in estado
    assert "No encontré" in _lo_que_dijo_la_tool(estado)


async def test_cancelar_con_varias_coincidencias_pregunta_cual() -> None:
    recordatorios = RecordatoriosEnMemoria(
        Recordatorio(usuario_id=USUARIO, texto="llamar a mamá", momento=_en(30)),
        Recordatorio(usuario_id=USUARIO, texto="llamar al banco", momento=_en(60)),
    )
    grafo = _grafo_con(
        recordatorios, _pedido("cancelar_recordatorio", texto="llamar"), AIMessage("ok")
    )

    estado = await _preguntar(grafo, "cancelá el de llamar")

    assert "__interrupt__" not in estado
    texto = _lo_que_dijo_la_tool(estado)
    assert "«llamar a mamá»" in texto
    assert "«llamar al banco»" in texto
    assert "¿Cuál de estos?" in texto


async def test_cancelar_uno_que_ya_salio_lo_dice() -> None:
    """El despachador lo mandó entre la pregunta y el sí: el compare-and-set pierde."""
    recordatorios = _con_pizza_y_banco()
    grafo = _grafo_con(
        recordatorios, _pedido("cancelar_recordatorio", texto="pizza"), AIMessage("ok")
    )
    await _preguntar(grafo, "cancelá el de la pizza")

    [pizza] = [r for r in recordatorios.guardados.values() if r.texto == "sacar la pizza"]
    assert pizza.id is not None
    recordatorios.reclamos_perdidos.add(pizza.id)  # al reanudar, el CAS no gana
    estado = await _reanudar(grafo, aprobado=True)

    assert "ya no está pendiente" in _lo_que_dijo_la_tool(estado)
    assert recordatorios.estado_de(pizza.id) is EstadoDeRecordatorio.PENDIENTE


# --- Disponibilidad ----------------------------------------------------------


def test_sin_repositorio_no_se_ofrecen() -> None:
    nombres = {h.name for h in construir_herramientas(None)}

    assert "crear_recordatorio" not in nombres


def test_no_necesitan_google() -> None:
    """Sin calendario, tareas ni correo (sin OAuth), los recordatorios igual están."""
    nombres = {h.name for h in construir_herramientas(None, recordatorios=RecordatoriosEnMemoria())}

    assert {"crear_recordatorio", "recordatorios_pendientes", "cancelar_recordatorio"} <= nombres
