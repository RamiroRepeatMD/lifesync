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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from src.domain.entities.evento import Evento
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import CalendarioFalso, ModeloFalso

USUARIO = uuid4()
# Las creaciones van a futuro relativo: desde PB-027 crear en el pasado se
# frena, y una fecha fija se vuelve pasado sola con el tiempo.
FUTURO = (datetime.now(ZONA_HORARIA) + timedelta(days=7)).date().isoformat()

PEDIDO_CREAR = AIMessage(
    "",
    tool_calls=[
        {
            "name": "crear_evento_en_calendario",
            "args": {"titulo": "Dentista", "fecha": FUTURO, "hora_inicio": "10:00"},
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
        (
            "crear_evento_en_calendario",
            {
                "titulo",
                "fecha",
                "hora_inicio",
                "duracion_minutos",
                "repetir",
                "dias",
                "hasta",
                "veces",
            },
        ),
        ("eliminar_evento_del_calendario", {"fecha", "titulo", "todos", "toda_la_serie"}),
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
                    "fecha": FUTURO,
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
                    "fecha": FUTURO,
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


# --- Modificar (PB-017) ------------------------------------------------------
#
# Mismo esqueleto que eliminar, más el merge. Los tests del merge son los que
# importan: "sólo el título" no puede tocar horarios, y "sólo la hora" no puede
# cambiar la duración.


def _pedido_modificar(**cambios: Any) -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {
                "name": "modificar_evento_del_calendario",
                "args": {"fecha": "2026-09-05", "titulo": "dentista", **cambios},
                "id": "t1",
            }
        ],
    )


def _dentista() -> Evento:
    inicio = datetime(2026, 9, 5, 13, 0, tzinfo=UTC)  # 10:00 en Buenos Aires
    return Evento(
        titulo="Dentista",
        inicio=inicio,
        fin=inicio + timedelta(minutes=30),  # dura 30, no 60: para ver el merge
        id="id-dentista",
    )


async def test_modificar_sin_confirmacion_no_patchea() -> None:
    calendario = CalendarioFalso(eventos=(_dentista(),))
    grafo = _grafo_con(calendario, _pedido_modificar(nueva_hora_inicio="16:00"), AIMessage("ok"))

    estado = await _preguntar(grafo, "cambiale la hora al dentista")

    assert "__interrupt__" in estado
    assert calendario.modificados == []


async def test_el_resumen_muestra_antes_y_despues() -> None:
    calendario = CalendarioFalso(eventos=(_dentista(),))
    grafo = _grafo_con(calendario, _pedido_modificar(nueva_hora_inicio="16:00"), AIMessage("ok"))

    estado = await _preguntar(grafo, "pasalo a las 16")

    resumen = estado["__interrupt__"][0].value["resumen"]
    assert "10:00" in resumen  # el antes
    assert "16:00" in resumen  # el después
    assert "→" in resumen


async def test_cambiar_solo_la_hora_conserva_la_duracion() -> None:
    """El dentista dura 30 minutos: movido a las 16 tiene que durar 30, no 60."""
    calendario = CalendarioFalso(eventos=(_dentista(),))
    grafo = _grafo_con(calendario, _pedido_modificar(nueva_hora_inicio="16:00"), AIMessage("ok"))

    await _preguntar(grafo, "pasalo a las 16")
    await _reanudar(grafo, aprobado=True)

    _, deseado = calendario.modificados[0]
    assert deseado.fin is not None
    assert deseado.fin - deseado.inicio == timedelta(minutes=30)
    assert deseado.titulo == "Dentista"  # el título no se tocó


async def test_cambiar_solo_el_titulo_no_toca_los_horarios() -> None:
    original = _dentista()
    calendario = CalendarioFalso(eventos=(original,))
    grafo = _grafo_con(calendario, _pedido_modificar(nuevo_titulo="Odontóloga"), AIMessage("ok"))

    await _preguntar(grafo, "renombralo")
    await _reanudar(grafo, aprobado=True)

    _, deseado = calendario.modificados[0]
    assert deseado.titulo == "Odontóloga"
    assert deseado.inicio == original.inicio
    assert deseado.fin == original.fin


async def test_cambiar_la_duracion_explicita_gana() -> None:
    calendario = CalendarioFalso(eventos=(_dentista(),))
    grafo = _grafo_con(calendario, _pedido_modificar(nueva_duracion_minutos=90), AIMessage("ok"))

    await _preguntar(grafo, "que dure una hora y media")
    await _reanudar(grafo, aprobado=True)

    _, deseado = calendario.modificados[0]
    assert deseado.fin is not None
    assert deseado.fin - deseado.inicio == timedelta(minutes=90)


async def test_darle_hora_a_un_dia_completo_lo_convierte() -> None:
    """Es lo que la persona pide al decir "ponelo a las 15"."""
    cumple = Evento(
        titulo="Cumple", inicio=datetime(2026, 9, 5, tzinfo=UTC), todo_el_dia=True, id="id-c"
    )
    calendario = CalendarioFalso(eventos=(cumple,))
    grafo = _grafo_con(
        calendario,
        _pedido_modificar(titulo="cumple", nueva_hora_inicio="15:00"),
        AIMessage("ok"),
    )

    await _preguntar(grafo, "ponele hora al cumple")
    await _reanudar(grafo, aprobado=True)

    _, deseado = calendario.modificados[0]
    assert deseado.todo_el_dia is False
    assert deseado.inicio.hour == 15  # hora local, construida en la zona


async def test_modificar_sin_ningun_cambio_pregunta_que() -> None:
    calendario = CalendarioFalso(eventos=(_dentista(),))
    grafo = _grafo_con(calendario, _pedido_modificar(), AIMessage("¿qué le cambio?"))

    estado = await _preguntar(grafo, "modificá el dentista")

    assert "__interrupt__" not in estado
    assert calendario.modificados == []
    assert calendario.consultados == []  # ni siquiera fue a buscar


async def test_modificar_con_varias_coincidencias_pide_precision() -> None:
    dos = (
        Evento(titulo="Dentista Norte", inicio=datetime(2026, 9, 5, 10, 0, tzinfo=UTC), id="a"),
        Evento(titulo="Dentista Sur", inicio=datetime(2026, 9, 5, 15, 0, tzinfo=UTC), id="b"),
    )
    calendario = CalendarioFalso(eventos=dos)
    grafo = _grafo_con(calendario, _pedido_modificar(nueva_hora_inicio="16:00"), AIMessage("ok"))

    estado = await _preguntar(grafo, "cambiá el dentista")

    assert "__interrupt__" not in estado
    assert calendario.modificados == []


async def test_rechazar_no_modifica_nada() -> None:
    calendario = CalendarioFalso(eventos=(_dentista(),))
    grafo = _grafo_con(calendario, _pedido_modificar(nueva_hora_inicio="16:00"), AIMessage("ok"))

    await _preguntar(grafo, "pasalo a las 16")
    await _reanudar(grafo, aprobado=False)

    assert calendario.modificados == []


def test_el_modelo_tampoco_ve_el_usuario_en_modificar() -> None:
    herramientas = {h.name: h for h in construir_herramientas(CalendarioFalso())}
    esquema = herramientas["modificar_evento_del_calendario"].tool_call_schema
    assert isinstance(esquema, type) and issubclass(esquema, BaseModel)

    props = set(esquema.model_json_schema()["properties"])
    assert props == {
        "fecha",
        "titulo",
        "nuevo_titulo",
        "nueva_fecha",
        "nueva_hora_inicio",
        "nueva_duracion_minutos",
    }


async def test_los_titulos_no_se_loguean_al_modificar() -> None:
    calendario = CalendarioFalso(eventos=(_dentista(),))
    grafo = _grafo_con(
        calendario,
        _pedido_modificar(nuevo_titulo="Sesión con la psicóloga"),
        AIMessage("listo"),
    )

    with structlog.testing.capture_logs() as eventos:
        await _preguntar(grafo, "renombralo")
        await _reanudar(grafo, aprobado=True)

    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "psicóloga" not in registrado
    assert "Dentista" not in registrado


async def test_modificar_hacia_el_mismo_estado_no_propone_nada() -> None:
    """Visto contra Gemini real: pidió "cambiar" la hora por la que ya tenía.

    Proponer "10:00 → 10:00" y pedir un sí por eso es ruido: se frena antes.
    """
    original = _dentista()  # 10:00 en Buenos Aires, dura 30 minutos
    calendario = CalendarioFalso(eventos=(original,))
    grafo = _grafo_con(
        calendario,
        _pedido_modificar(nueva_hora_inicio="10:00", nueva_duracion_minutos=30),
        AIMessage("¿A qué hora lo querés pasar?"),
    )

    estado = await _preguntar(grafo, "cambiale la hora al dentista")

    assert "__interrupt__" not in estado
    assert calendario.modificados == []


# --- Borrar varios (bug de la prueba real del 30/09) --------------------------
#
# Con dos "Dentista" idénticos, eliminar preguntaba "¿cuál?" para siempre:
# nada los distinguía y no había forma de pedir "los dos".


def _pedido_borrar(**args: Any) -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {
                "name": "eliminar_evento_del_calendario",
                "args": {"fecha": "2026-09-05", "titulo": "dentista", **args},
                "id": "t1",
            }
        ],
    )


def _gemelos() -> tuple[Evento, Evento]:
    """Dos eventos idénticos salvo el id, como los que dejó el bug del duplicado."""
    return _evento("Dentista", 16, "id-a"), _evento("Dentista", 16, "id-b")


async def test_todos_borra_los_dos_con_una_sola_confirmacion() -> None:
    calendario = CalendarioFalso(eventos=_gemelos())
    grafo = _grafo_con(calendario, _pedido_borrar(todos=True), AIMessage("listo"))

    estado = await _preguntar(grafo, "borrá los dos dentista")
    resumen = estado["__interrupt__"][0].value["resumen"]
    await _reanudar(grafo, aprobado=True)

    assert resumen.startswith("Eliminar estos 2 eventos:")
    assert sorted(evento_id for _, evento_id in calendario.eliminados) == ["id-a", "id-b"]


async def test_con_identicos_y_sin_todos_borra_uno() -> None:
    """Son intercambiables: borrar cualquiera es borrar "uno de ellos"."""
    calendario = CalendarioFalso(eventos=_gemelos())
    grafo = _grafo_con(calendario, _pedido_borrar(), AIMessage("listo"))

    estado = await _preguntar(grafo, "borrá el dentista")
    resumen = estado["__interrupt__"][0].value["resumen"]
    await _reanudar(grafo, aprobado=True)

    assert resumen.startswith("Eliminar uno de los 2 eventos idénticos")
    assert len(calendario.eliminados) == 1


async def test_con_distintos_y_sin_todos_pregunta_y_ofrece_todos() -> None:
    distintos = (_evento("Dentista Norte", 10, "id-n"), _evento("Dentista Sur", 15, "id-s"))
    calendario = CalendarioFalso(eventos=distintos)
    grafo = _grafo_con(calendario, _pedido_borrar(), AIMessage("¿cuál?"))

    estado = await _preguntar(grafo, "borrá el dentista")

    assert "__interrupt__" not in estado
    assert calendario.eliminados == []
    respuesta = next(m for m in estado["messages"] if isinstance(m, ToolMessage))
    assert "los borro todos" in str(respuesta.content)


async def test_rechazar_el_borrado_multiple_no_borra_nada() -> None:
    calendario = CalendarioFalso(eventos=_gemelos())
    grafo = _grafo_con(calendario, _pedido_borrar(todos=True), AIMessage("ok"))

    await _preguntar(grafo, "borrá los dos")
    await _reanudar(grafo, aprobado=False)

    assert calendario.eliminados == []
