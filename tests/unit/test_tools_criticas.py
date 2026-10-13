"""Las invariantes de las tools críticas, sobre TODAS las tools (PB-040).

Los tests de cada tool prueban su camino; éstos prueban las promesas que valen
para todas a la vez, y fallan si mañana se suma una tool que no las cumple:

- ninguna le muestra al modelo de quién es la conversación (PB-015);
- ninguna escritura escribe fuera del nodo de la política: falla cerrado (RF-08);
- con un correo a la vista, toda escritura pide el sí (modo estricto, RF-08 v2);
- dos escrituras juntas son una lista, y cada una corre una sola vez;
- ninguna falla de Google se escapa como excepción, y cada una trae su remedio;
- lo que se borra a medias se cuenta tal cual, y una falla no prevista no se
  reintenta sola.

Se escribieron después de medir (cobertura con ramas, 13/10): 89% en total y
88% en `herramientas.py`, con la rama de "falla cerrado" sin ningún test. Tres
sondas previas no encontraron bugs: lo que faltaba era la red que lo garantice.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.domain.entities.correo import Correo
from src.domain.entities.evento import Evento
from src.domain.entities.recordatorio import (
    MAX_CARACTERES_DE_RECORDATORIO,
    EstadoDeRecordatorio,
    Recordatorio,
)
from src.domain.entities.tarea import Tarea
from src.domain.exceptions import (
    AgenteNoDisponibleError,
    AutorizacionFallidaError,
    CuentaNoConectadaError,
    EntityNotFoundError,
    EnvioInciertoError,
    PermisoInsuficienteError,
    RepositoryError,
    ServiceUnavailableError,
)
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.llm.agente_gemini import CIERRE_DEL_LOTE, AgenteGemini
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import TIPO_POR_HERRAMIENTA, construir_herramientas
from tests.dobles import (
    CalendarioFalso,
    CorreosFalsos,
    ModeloFalso,
    RecordatoriosEnMemoria,
    TareasFalsas,
)

AHORA = datetime.now(ZONA_HORARIA)
MANANA = (AHORA + timedelta(days=1)).replace(hour=13, minute=0, second=0, microsecond=0)
EN_DOS_HORAS = (AHORA + timedelta(hours=2)).replace(second=0, microsecond=0)
CONFIRMAR = "¿Confirmás?"


def _dia(momento: datetime) -> str:
    return momento.date().isoformat()


def _correo(id_: str, remitente: str, asunto: str) -> Correo:
    return Correo(
        id=id_,
        remitente=remitente,
        asunto=asunto,
        fecha=AHORA,
        no_leido=True,
        fragmento="",
        cuerpo="Hola, ¿cómo va?",
        hilo_id=f"hilo-{id_}",
        mensaje_id=f"<{id_}@ejemplo.com>",
    )


class Puertos:
    """Los cuatro dobles, con datos que encajan con los argumentos de `CASOS`."""

    def __init__(self) -> None:
        self.usuario: UUID = uuid4()
        dentista = Evento(
            titulo="Dentista", inicio=MANANA, fin=MANANA + timedelta(hours=1), id="e1"
        )
        medianoche = MANANA.replace(hour=0)
        cumple = Evento(
            titulo="Cumple de Ana",
            inicio=medianoche,
            fin=medianoche + timedelta(days=1),
            id="e2",
            todo_el_dia=True,
        )
        self.calendario = CalendarioFalso(eventos=(dentista, cumple))
        self.tareas = TareasFalsas(pendientes=(Tarea(titulo="Pagar la luz", id="t1"),))
        self.correos = CorreosFalsos(
            (
                _correo("m1", "Ana <ana@ejemplo.com>", "Hola"),
                _correo("m2", "Sistema de avisos", "Aviso"),  # sin dirección
                _correo("m3", "Beto <beto@ejemplo.com>", "x" * 1000),  # asunto enorme
            )
        )
        self.recordatorios = RecordatoriosEnMemoria(
            Recordatorio(
                usuario_id=self.usuario, texto="llamar al banco", momento=AHORA + timedelta(hours=3)
            )
        )

    def herramientas(self) -> tuple[BaseTool, ...]:
        return construir_herramientas(
            self.calendario, self.tareas, self.correos, self.recordatorios
        )

    def escrituras(self) -> int:
        """Cuántas escrituras llegaron a los puertos, de cualquiera de las 12 tools."""
        c, t = self.calendario, self.tareas
        recordatorios = sum(
            1
            for r in self.recordatorios.guardados.values()
            if r.texto != "llamar al banco" or r.estado is not EstadoDeRecordatorio.PENDIENTE
        )
        return (
            len(c.creados)
            + len(c.eliminados)
            + len(c.modificados)
            + len(t.creadas)
            + len(t.completadas)
            + len(t.pospuestas)
            + len(t.eliminadas)
            + len(self.correos.enviados)
            + recordatorios
        )


# Argumentos válidos de cada escritura: con ellos, cada una tiene algo que hacer.
CASOS: dict[str, dict[str, Any]] = {
    "crear_evento_en_calendario": {
        "titulo": "Gimnasio",
        "fecha": _dia(MANANA),
        "hora_inicio": "10:00",
    },
    "modificar_evento_del_calendario": {
        "fecha": _dia(MANANA),
        "titulo": "Dentista",
        "nueva_hora_inicio": "15:00",
    },
    "eliminar_evento_del_calendario": {"fecha": _dia(MANANA), "titulo": "Dentista"},
    "crear_tarea": {"titulo": "Comprar pan"},
    "completar_tarea": {"titulo": "luz"},
    "posponer_tarea": {"titulo": "luz", "nueva_fecha": _dia(AHORA + timedelta(days=3))},
    "eliminar_tarea": {"titulo": "luz"},
    "enviar_correo": {"para": "beto@ejemplo.com", "asunto": "Sábado", "texto": "¿Vamos?"},
    "responder_correo": {"id_correo": "m1", "texto": "¡Todo bien!"},
    "reenviar_correo": {"id_correo": "m1", "para": "beto@ejemplo.com"},
    "crear_recordatorio": {
        "texto": "sacar la pizza",
        "fecha": _dia(EN_DOS_HORAS),
        "hora": f"{EN_DOS_HORAS:%H:%M}",
    },
    "cancelar_recordatorio": {"texto": "banco"},
}
LECTURAS: dict[str, dict[str, Any]] = {
    "fecha_y_hora_actual": {},
    "eventos_del_calendario": {"desde": _dia(AHORA), "hasta": _dia(MANANA)},
    "tareas_pendientes": {},
    "buscar_correos": {},
    "leer_correo": {"id_correo": "m1"},
    "recordatorios_pendientes": {},
}
SIN_GOOGLE = {
    "fecha_y_hora_actual",
    "recordatorios_pendientes",
    "crear_recordatorio",
    "cancelar_recordatorio",
}
DE_GOOGLE = sorted((set(CASOS) | set(LECTURAS)) - SIN_GOOGLE)
ESCRITURAS_DE_GOOGLE = sorted(set(CASOS) - SIN_GOOGLE)


def _llamadas(*pedidos: tuple[str, dict[str, Any]]) -> AIMessage:
    """Un mensaje del modelo con todas las llamadas juntas, como hace Gemini."""
    return AIMessage(
        "",
        tool_calls=[
            {"name": nombre, "args": args, "id": f"c{i}"}
            for i, (nombre, args) in enumerate(pedidos)
        ],
    )


class Bot:
    """El agente de verdad (adaptador, grafo y tools) con modelo y puertos falsos."""

    def __init__(self, puertos: Puertos, *guion: AIMessage) -> None:
        self.puertos = puertos
        self.grafo = construir_grafo(
            ModeloFalso(guion=list(guion)), puertos.herramientas(), InMemorySaver()
        )
        self.agente = AgenteGemini(self.grafo)

    async def decir(self, texto: str) -> str:
        usuario = self.puertos.usuario
        return await self.agente.responder(
            ConsultaDelUsuario(conversacion_id=usuario, usuario_id=usuario, texto=texto)
        )

    async def respuestas(self) -> dict[str, str]:
        """Lo que contestó cada tool, por el id de su llamada."""
        configuracion: Any = {"configurable": {"thread_id": str(self.puertos.usuario)}}
        estado = await self.grafo.aget_state(configuracion)
        return {
            m.tool_call_id: str(m.content)
            for m in estado.values["messages"]
            if isinstance(m, ToolMessage)
        }


async def _usar(puertos: Puertos, nombre: str, args: dict[str, Any]) -> str:
    """Pide una tool, dice "sí" si pausa, y devuelve lo que contestó."""
    bot = Bot(puertos, _llamadas((nombre, args)), AIMessage("ok"))
    if CONFIRMAR in await bot.decir("hacelo"):
        await bot.decir("sí")
    return (await bot.respuestas())["c0"]


# --- La tabla está completa -----------------------------------------------------------


def test_toda_tool_esta_en_las_tablas_de_esta_suite() -> None:
    """Una escritura nueva sin su caso acá haría pasar las invariantes en el vacío."""
    nombres = {h.name for h in Puertos().herramientas()}

    assert set(CASOS) == set(TIPO_POR_HERRAMIENTA)
    assert nombres == set(CASOS) | set(LECTURAS)


# --- A. Lo que ve el modelo ----------------------------------------------------------


@pytest.mark.parametrize("nombre", sorted(set(CASOS) | set(LECTURAS)))
def test_ninguna_tool_le_muestra_al_modelo_de_quien_es_la_conversacion(nombre: str) -> None:
    """PB-015: si el usuario fuera un parámetro, lo completaría el modelo (y un mensaje)."""
    herramienta = next(h for h in Puertos().herramientas() if h.name == nombre)

    esquema = herramienta.tool_call_schema
    assert isinstance(esquema, type) and issubclass(esquema, BaseModel)
    propiedades = set(esquema.model_json_schema().get("properties", {}))

    assert "runtime" not in propiedades
    assert not any("usuario" in p or "contexto" in p for p in propiedades)


# --- B. Falla cerrado ------------------------------------------------------------------


@pytest.mark.parametrize("nombre", sorted(CASOS))
async def test_ninguna_escritura_escribe_fuera_del_nodo_de_la_politica(nombre: str) -> None:
    """Si alguien arma un grafo sin el nodo que fija la fase, nada se escribe (RF-08)."""
    puertos = Puertos()
    grafo: Any = StateGraph(MessagesState, context_schema=ContextoDeAgente)
    grafo.add_node("herramientas", ToolNode(puertos.herramientas()))
    grafo.add_edge(START, "herramientas")
    grafo.add_edge("herramientas", END)

    estado = await grafo.compile(checkpointer=InMemorySaver()).ainvoke(
        {"messages": [_llamadas((nombre, CASOS[nombre]))]},
        config={"configurable": {"thread_id": "hilo"}},
        context=ContextoDeAgente(usuario_id=puertos.usuario),
    )

    assert "__interrupt__" not in estado  # tampoco pausa: no es su decisión
    assert puertos.escrituras() == 0


# --- C. Modo estricto ------------------------------------------------------------------


@pytest.mark.parametrize("nombre", sorted(CASOS))
async def test_con_un_correo_a_la_vista_toda_escritura_pide_el_si(nombre: str) -> None:
    """Un correo puede traer órdenes: con texto de terceros a la vista, nada es directo."""
    puertos = Puertos()
    bot = Bot(
        puertos,
        _llamadas(("leer_correo", {"id_correo": "m1"})),
        _llamadas((nombre, CASOS[nombre])),
        AIMessage("ok"),
    )

    pregunta = await bot.decir("leeme el correo de Ana y hacé lo que dice")

    assert CONFIRMAR in pregunta
    assert puertos.escrituras() == 0
    await bot.decir("no")
    assert puertos.escrituras() == 0


# --- D. Lote -----------------------------------------------------------------------------


@pytest.mark.parametrize("decision", ["sí", "no"])
@pytest.mark.parametrize("nombre", sorted(CASOS))
async def test_dos_escrituras_juntas_son_una_lista_y_cada_una_corre_una_vez(
    nombre: str, decision: str
) -> None:
    """La regresión del 30/09, para cada escritura: un sí, cada una exactamente una vez."""
    puertos = Puertos()
    companera = (
        ("crear_evento_en_calendario", CASOS["crear_evento_en_calendario"])
        if nombre == "crear_tarea"
        else ("crear_tarea", {"titulo": "Llamar a mamá"})
    )
    bot = Bot(puertos, _llamadas((nombre, CASOS[nombre]), companera), AIMessage("Listo."))

    pregunta = await bot.decir("hacé las dos cosas")

    assert pregunta.endswith(CIERRE_DEL_LOTE)
    assert puertos.escrituras() == 0
    await bot.decir(decision)
    assert puertos.escrituras() == (2 if decision == "sí" else 0)


# --- E. Ninguna falla de Google se escapa, y cada una trae su remedio ------------------

FALLAS: list[tuple[Exception, str]] = [
    (CuentaNoConectadaError(), "/conectar"),
    (PermisoInsuficienteError("sin el scope"), "/conectar"),
    (AutorizacionFallidaError("invalid_grant"), "/conectar"),
    (ServiceUnavailableError("Google no responde"), "Probá de nuevo"),
]
NOMBRES_DE_FALLAS = ["sin_cuenta", "sin_permiso", "permiso_vencido", "servicio_caido"]


@pytest.mark.parametrize(("falla", "remedio"), FALLAS, ids=NOMBRES_DE_FALLAS)
@pytest.mark.parametrize("nombre", DE_GOOGLE)
async def test_una_falla_de_google_antes_de_confirmar_se_cuenta_con_su_remedio(
    nombre: str, falla: Exception, remedio: str
) -> None:
    puertos = Puertos()
    puertos.calendario.fallar_con = falla
    puertos.tareas.fallar_con = falla
    puertos.correos.fallar_con = falla

    texto = await _usar(puertos, nombre, {**CASOS, **LECTURAS}[nombre])

    assert remedio in texto
    assert puertos.escrituras() == 0


@pytest.mark.parametrize(("falla", "remedio"), FALLAS, ids=NOMBRES_DE_FALLAS)
@pytest.mark.parametrize("nombre", ESCRITURAS_DE_GOOGLE)
async def test_una_falla_de_google_al_escribir_se_cuenta_con_su_remedio(
    nombre: str, falla: Exception, remedio: str
) -> None:
    """La búsqueda anduvo y la persona ya dijo que sí: falla recién la escritura."""
    puertos = Puertos()
    puertos.calendario.fallar_al_escribir = falla
    puertos.tareas.fallar_al_escribir = falla
    puertos.correos.fallar_al_enviar = falla

    texto = await _usar(puertos, nombre, CASOS[nombre])

    assert remedio in texto
    assert puertos.escrituras() == 0


class _RecordatoriosQueFallanAlCancelar(RecordatoriosEnMemoria):
    async def cambiar_estado(
        self,
        recordatorio_id: UUID,
        desde: EstadoDeRecordatorio,
        hacia: EstadoDeRecordatorio,
    ) -> bool:
        raise RepositoryError("la base se cayó justo")


@pytest.mark.parametrize(
    "nombre", ["recordatorios_pendientes", "crear_recordatorio", "cancelar_recordatorio"]
)
async def test_con_la_base_caida_los_recordatorios_lo_dicen(nombre: str) -> None:
    puertos = Puertos()
    puertos.recordatorios.fallar_con = RepositoryError("base caída")

    texto = await _usar(puertos, nombre, {**CASOS, **LECTURAS}[nombre])

    assert "Probá de nuevo" in texto


async def test_si_la_base_se_cae_al_cancelar_el_recordatorio_sigue_programado() -> None:
    puertos = Puertos()
    puertos.recordatorios = _RecordatoriosQueFallanAlCancelar(
        Recordatorio(usuario_id=puertos.usuario, texto="llamar al banco", momento=MANANA)
    )

    texto = await _usar(puertos, "cancelar_recordatorio", CASOS["cancelar_recordatorio"])

    assert "No pude cancelar el recordatorio" in texto
    [recordatorio] = puertos.recordatorios.guardados.values()
    assert recordatorio.estado is EstadoDeRecordatorio.PENDIENTE


# --- F. Carreras y borrados a medias ---------------------------------------------------


@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [
        ("modificar_evento_del_calendario", "Ese evento ya no está en el calendario"),
        ("completar_tarea", "Esa tarea ya no está en la lista"),
        ("posponer_tarea", "Esa tarea ya no está en la lista"),
    ],
)
async def test_si_lo_borraron_entre_la_busqueda_y_la_escritura_lo_dice(
    nombre: str, esperado: str
) -> None:
    puertos = Puertos()
    puertos.calendario.fallar_al_escribir = EntityNotFoundError("404")
    puertos.tareas.fallar_al_escribir = EntityNotFoundError("404")

    texto = await _usar(puertos, nombre, CASOS[nombre])

    assert esperado in texto


async def test_un_borrado_de_varios_que_se_corta_cuenta_lo_que_borro() -> None:
    """Lo irreversible a medias: lo ya borrado se cuenta y entra en la bitácora (PB-026)."""
    puertos = Puertos()
    puertos.calendario.eventos = tuple(
        Evento(
            titulo=f"Clase de {materia}",
            inicio=MANANA + timedelta(hours=n),
            fin=MANANA + timedelta(hours=n, minutes=45),
            id=f"clase-{n}",
        )
        for n, materia in enumerate(["yoga", "inglés", "piano"])
    )
    puertos.calendario.fallar_al_escribir = ServiceUnavailableError("se cayó")
    puertos.calendario.escrituras_antes_de_fallar = 1
    grafo = construir_grafo(
        ModeloFalso(
            guion=[
                _llamadas(
                    (
                        "eliminar_evento_del_calendario",
                        {"fecha": _dia(MANANA), "titulo": "Clase", "todos": True},
                    )
                ),
                AIMessage("ok"),
            ]
        ),
        puertos.herramientas(),
        InMemorySaver(),
    )
    configuracion: Any = {"configurable": {"thread_id": "hilo"}}
    contexto = ContextoDeAgente(usuario_id=puertos.usuario)
    await grafo.ainvoke(
        {"messages": [HumanMessage("borrá todas las clases de mañana")]},
        config=configuracion,
        context=contexto,
    )

    estado = await grafo.ainvoke(
        Command(resume={"aprobado": True}), config=configuracion, context=contexto
    )

    [respuesta] = [str(m.content) for m in estado["messages"] if isinstance(m, ToolMessage)]
    assert respuesta.startswith("Alcancé a eliminar 1 de 3.")
    assert "Probá de nuevo" in respuesta
    assert puertos.calendario.eliminados == [(puertos.usuario, "clase-0")]
    [hecho] = contexto.acciones_realizadas  # sólo lo que de verdad se borró
    assert hecho.startswith("Eventos eliminados (1): ")
    assert "yoga" in hecho and "inglés" not in hecho


# --- G. Una falla que nadie previó ----------------------------------------------------------


async def test_una_falla_no_prevista_no_se_reintenta_ni_traba_la_conversacion() -> None:
    """Supabase que parpadea en medio de un turno: el aviso de RF-19 y nada más.

    Que el turno siguiente ande también con Gemini real —con el `tool_call`
    huérfano en el historial— se verificó con una sonda el 13/10 (PB-040).
    """
    puertos = Puertos()
    puertos.tareas.fallar_con = RepositoryError("Supabase parpadeó")
    bot = Bot(puertos, _llamadas(("crear_tarea", CASOS["crear_tarea"])), AIMessage("Hola."))

    with pytest.raises(AgenteNoDisponibleError):  # el caso de uso manda el aviso de RF-19
        await bot.decir("anotá comprar pan")
    puertos.tareas.fallar_con = None
    respuesta = await bot.decir("hola")

    assert respuesta == "Hola."
    assert puertos.tareas.creadas == []  # la escritura fallida no se reintentó sola


async def test_un_envio_en_duda_no_entra_en_la_bitacora_ni_se_reintenta() -> None:
    puertos = Puertos()
    puertos.correos.fallar_al_enviar = EnvioInciertoError("sin respuesta")

    texto = await _usar(puertos, "enviar_correo", CASOS["enviar_correo"])

    assert "Enviados" in texto
    assert puertos.correos.enviados == []


# --- H. Lo inválido no llega a la pausa ni escribe --------------------------------------

DIRECCIONES_LARGAS = ", ".join(f"persona{i}.{'x' * 150}@ejemplo.com" for i in range(5))


@pytest.mark.parametrize(
    ("nombre", "args", "esperado"),
    [
        (
            "eliminar_evento_del_calendario",
            {"fecha": "mañana", "titulo": "Dentista"},
            "Necesito el día",
        ),
        (
            "modificar_evento_del_calendario",
            {"fecha": _dia(MANANA), "titulo": " ", "nuevo_titulo": "x"},
            "Necesito el día",
        ),
        (
            "modificar_evento_del_calendario",
            {"fecha": _dia(MANANA), "titulo": "Cumpleaños", "nuevo_titulo": "x"},
            "No encontré ningún evento",
        ),
        (
            "modificar_evento_del_calendario",
            {"fecha": _dia(MANANA), "titulo": "Dentista", "nueva_hora_inicio": "25:99"},
            "No entendí los datos nuevos",
        ),
        (
            "modificar_evento_del_calendario",
            {"fecha": _dia(MANANA), "titulo": "Dentista", "nueva_duracion_minutos": 99999},
            "No entendí los datos nuevos",
        ),
        (
            "modificar_evento_del_calendario",
            {"fecha": _dia(MANANA), "titulo": "Cumple", "nueva_fecha": "el viernes"},
            "No entendí los datos nuevos",
        ),
        (
            "crear_evento_en_calendario",
            {**CASOS["crear_evento_en_calendario"], "repetir": "semanal", "hasta": "pronto"},
            "No entendí hasta cuándo se repite",
        ),
        (
            "crear_evento_en_calendario",
            {**CASOS["crear_evento_en_calendario"], "repetir": "semanal", "hasta": _dia(AHORA)},
            "es anterior a la primera",
        ),
        ("crear_tarea", {"titulo": " "}, "Necesito saber qué hay que hacer"),
        ("completar_tarea", {"titulo": " "}, "¿Cuál tarea?"),
        ("enviar_correo", {"para": " ", "asunto": "a", "texto": "b"}, "Falta a quién"),
        (
            "enviar_correo",
            {"para": "ana@ejemplo.com", "asunto": "x" * 201, "texto": "b"},
            "El asunto es muy largo",
        ),
        (
            "enviar_correo",
            {"para": DIRECCIONES_LARGAS, "asunto": "x" * 200, "texto": "y" * 3000},
            "demasiado largo para mostrártelo entero",
        ),
        ("leer_correo", {"id_correo": " "}, "Necesito el id del correo"),
        (
            "responder_correo",
            {"id_correo": "m2", "texto": "ok"},
            "No encontré una dirección válida para responder",
        ),
        (
            "responder_correo",
            {"id_correo": "m3", "texto": "y" * 3000},
            "demasiado larga para mostrártela entera",
        ),
        (
            "reenviar_correo",
            {"id_correo": "m1", "para": "beto@ejemplo.com", "nota": "x" * 1001},
            "La nota es muy larga",
        ),
        (
            "reenviar_correo",
            {"id_correo": "no-existe", "para": "beto@ejemplo.com"},
            "No encontré ese correo",
        ),
        (
            "reenviar_correo",
            {"id_correo": "m3", "para": DIRECCIONES_LARGAS, "nota": "x" * 1000},
            "No puedo mostrarte el reenvío entero",
        ),
        (
            "crear_recordatorio",
            {**CASOS["crear_recordatorio"], "texto": "x" * (MAX_CARACTERES_DE_RECORDATORIO + 1)},
            "Es muy largo para un recordatorio",
        ),
        ("cancelar_recordatorio", {"texto": " "}, "¿Cuál recordatorio?"),
    ],
)
async def test_lo_invalido_se_explica_sin_pausa_y_sin_escribir(
    nombre: str, args: dict[str, Any], esperado: str
) -> None:
    puertos = Puertos()
    bot = Bot(puertos, _llamadas((nombre, args)), AIMessage("ok"))

    respuesta = await bot.decir("hacelo")

    assert CONFIRMAR not in respuesta
    assert esperado in (await bot.respuestas())["c0"]
    assert puertos.escrituras() == 0


async def test_una_direccion_repetida_sale_una_sola_vez() -> None:
    puertos = Puertos()
    args = {"para": "ana@ejemplo.com; ANA@ejemplo.com", "asunto": "Hola", "texto": "x"}
    bot = Bot(puertos, _llamadas(("enviar_correo", args)), AIMessage("ok"))

    pregunta = await bot.decir("mandale a Ana")

    assert "Para: ana@ejemplo.com\n" in pregunta


async def test_dias_repetidos_o_vacios_no_duplican_la_repeticion() -> None:
    puertos = Puertos()
    args = {**CASOS["crear_evento_en_calendario"], "repetir": "semanal", "dias": "lunes, , lunes"}

    await _usar(puertos, "crear_evento_en_calendario", args)

    [(_, evento)] = puertos.calendario.creados
    assert evento.recurrencia is not None
    assert evento.recurrencia.dias == (0,)


async def test_el_listado_trae_cada_correo_con_su_id_aunque_no_tenga_adelanto() -> None:
    puertos = Puertos()

    listado = await _usar(puertos, "buscar_correos", LECTURAS["buscar_correos"])

    for correo_id in ("m1", "m2", "m3"):
        assert f"(id: {correo_id})" in listado
