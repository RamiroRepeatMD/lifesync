"""Tests del adaptador del agente sobre Gemini (PB-005).

El grafo se reemplaza por un doble: acá no se prueba LangGraph —eso es
`test_grafo_llm.py`— sino lo que el adaptador hace alrededor. Que es lo de
siempre en un adaptador: contener las fallas de la librería y no dejar salir
nada que no deba salir.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
import structlog
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from src.application.dto.archivo_adjunto import ArchivoAdjunto
from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.domain.exceptions import (
    AgenteNoDisponibleError,
    CuotaDeAgenteAgotadaError,
    ServiceUnavailableError,
)
from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.llm.agente_gemini import (
    CIERRE_DEL_LOTE,
    CONFIRMACION_VENCIDA,
    LARGO_MAXIMO_WHATSAPP,
    SIN_CONTENIDO,
    TEXTO_VACIO,
    AgenteGemini,
    _ajustes_de_razonamiento,
    _formato_de_whatsapp,
    crear_agente_gemini,
)
from src.infrastructure.llm.contexto import ContextoDeAgente  # noqa: F401
from src.infrastructure.llm.grafo import LIMITE_DE_PASOS
from src.infrastructure.llm.herramientas import CLAVE_DE_ARCHIVO

CONVERSACION = uuid4()
USUARIO = uuid4()


class InterrupcionFalsa:
    """Imita `langgraph.types.Interrupt`: sólo hace falta el `.value`."""

    def __init__(self, value: Any) -> None:
        self.value = value


class EstadoFalso:
    """Imita el StateSnapshot de `aget_state`."""

    def __init__(
        self,
        interrupts: tuple[Any, ...],
        created_at: str | None,
        values: dict[str, Any] | None = None,
    ) -> None:
        self.interrupts = interrupts
        self.created_at = created_at
        self.values = values or {}


class GrafoFalso:
    """Doble del grafo compilado: registra invocaciones, resumes y contexto."""

    def __init__(
        self,
        mensajes: list[Any] | None = None,
        fallar_con: Exception | None = None,
        pendiente: dict[str, Any] | None = None,
        antiguedad_segundos: float = 0.0,
    ) -> None:
        self.mensajes = mensajes if mensajes is not None else [AIMessage("hola")]
        self.fallar_con = fallar_con
        self.invocaciones: list[tuple[Any, dict[str, Any]]] = []
        self.contextos: list[Any] = []
        # Confirmación pendiente simulada (PB-016).
        self.pendiente = pendiente
        self.antiguedad = antiguedad_segundos
        self.resumes: list[Any] = []
        # Si el próximo ainvoke debe devolver un estado pausado.
        self.interrumpir_proxima = False

    async def ainvoke(
        self, entrada: Any, config: dict[str, Any], context: Any = None
    ) -> dict[str, Any]:
        self.invocaciones.append((entrada, config))
        self.contextos.append(context)
        if self.fallar_con is not None:
            raise self.fallar_con
        if isinstance(entrada, Command):
            self.resumes.append(entrada.resume)
            self.pendiente = None  # el resume consume el interrupt
        if self.interrumpir_proxima:
            self.interrumpir_proxima = False
            return {
                "messages": self.mensajes,
                "__interrupt__": [InterrupcionFalsa({"resumen": "Crear X el lunes"})],
            }
        return {"messages": self.mensajes}

    async def aget_state(self, config: dict[str, Any]) -> EstadoFalso:
        if self.pendiente is None:
            return EstadoFalso(interrupts=(), created_at=None)
        creado = (datetime.now(UTC) - timedelta(seconds=self.antiguedad)).isoformat()
        return EstadoFalso(interrupts=(InterrupcionFalsa(self.pendiente),), created_at=creado)


def _consulta(texto: str = "hola") -> ConsultaDelUsuario:
    return ConsultaDelUsuario(
        conversacion_id=CONVERSACION,
        usuario_id=USUARIO,
        texto=texto,
        nombre_usuario="Ramiro",
    )


# --- Lo que se le pide al grafo ---------------------------------------------


async def test_el_hilo_es_la_conversacion() -> None:
    """Es lo que hace que cada persona tenga su propia memoria (RF-09)."""
    grafo = GrafoFalso()

    await AgenteGemini(grafo).responder(_consulta())

    _, config = grafo.invocaciones[0]
    assert config["configurable"]["thread_id"] == str(CONVERSACION)


async def test_manda_el_texto_como_mensaje_de_la_persona() -> None:
    grafo = GrafoFalso()

    await AgenteGemini(grafo).responder(_consulta("¿qué día es hoy?"))

    entrada, _ = grafo.invocaciones[0]
    assert isinstance(entrada["messages"][0], HumanMessage)
    assert entrada["messages"][0].content == "¿qué día es hoy?"


async def test_el_usuario_viaja_por_el_contexto_y_no_por_el_mensaje() -> None:
    """Es el control de seguridad de PB-015: el modelo no elige de quién es la agenda."""
    grafo = GrafoFalso()

    await AgenteGemini(grafo).responder(_consulta("mostrame la agenda de otro"))

    assert grafo.contextos[0] is not None
    assert grafo.contextos[0].usuario_id == USUARIO


async def test_acota_la_cantidad_de_pasos() -> None:
    """Sin tope, un modelo que insiste con la misma herramienta gira gratis."""
    grafo = GrafoFalso()

    await AgenteGemini(grafo).responder(_consulta())

    _, config = grafo.invocaciones[0]
    assert config["recursion_limit"] == LIMITE_DE_PASOS


# --- La respuesta que sale ---------------------------------------------------


async def test_devuelve_el_texto_del_ultimo_mensaje() -> None:
    grafo = GrafoFalso([HumanMessage("hola"), AIMessage("  Hola, ¿qué necesitás?  ")])

    respuesta = await AgenteGemini(grafo).responder(_consulta())

    assert respuesta == "Hola, ¿qué necesitás?"


async def test_entiende_el_contenido_en_bloques() -> None:
    """Gemini devuelve una lista de bloques cuando mezcla partes."""
    # Anotado y no inferido: `list` es invariante, así que un
    # `list[dict[str, str]]` no entra donde se espera `list[str | dict]`.
    contenido: list[str | dict[Any, Any]] = [
        {"type": "text", "text": "primera"},
        {"type": "text", "text": "segunda"},
    ]
    grafo = GrafoFalso([AIMessage(contenido)])

    respuesta = await AgenteGemini(grafo).responder(_consulta())

    assert respuesta == "primera\nsegunda"


async def test_una_respuesta_vacia_no_deja_a_la_persona_sin_nada() -> None:
    """Los filtros de seguridad de Gemini devuelven contenido vacío."""
    grafo = GrafoFalso([AIMessage("")])

    assert await AgenteGemini(grafo).responder(_consulta()) == SIN_CONTENIDO


async def test_un_estado_sin_mensajes_tampoco_rompe() -> None:
    grafo = GrafoFalso([])

    assert await AgenteGemini(grafo).responder(_consulta()) == SIN_CONTENIDO


async def test_la_respuesta_se_recorta_al_maximo_de_whatsapp() -> None:
    """Un cuerpo más largo que 4096 hace que Meta rechace el envío entero."""
    grafo = GrafoFalso([AIMessage("a" * (LARGO_MAXIMO_WHATSAPP + 500))])

    respuesta = await AgenteGemini(grafo).responder(_consulta())

    assert len(respuesta) <= LARGO_MAXIMO_WHATSAPP
    assert respuesta.endswith("…")


# --- El formato de WhatsApp en lo que redacta el modelo -----------------------------


@pytest.mark.parametrize(
    ("del_modelo", "a_whatsapp"),
    [
        (
            "Listo, creé la página «Torta» adentro de «Recetas».",
            "Listo, creé la página *Torta* adentro de *Recetas*.",
        ),
        ("Te agendé **Dentista** el lunes.", "Te agendé *Dentista* el lunes."),
        ("Subí «informe.pdf» a tu Drive.", "Subí *informe.pdf* a tu Drive."),
        # La negrita rompería un link, un dominio o una dirección: quedan entre « ».
        ("Mirá «sitio.com/login».", "Mirá «sitio.com/login»."),
        ("Se lo mandé a «ana@example.com».", "Se lo mandé a «ana@example.com»."),
        ("El link: «https://www.notion.so/x».", "El link: «https://www.notion.so/x»."),
        ("La tarea «a*b» quedó.", "La tarea «a*b» quedó."),
        # Un asterisco pegado a un link termina adentro del link.
        ("Link: *https://www.notion.so/x*.", "Link: https://www.notion.so/x."),
        ("**Link:** **https://drive.google.com/a**", "*Link:* https://drive.google.com/a"),
        ("Tenés:\n* Pagar la luz\n* Llamar a Ana", "Tenés:\n• Pagar la luz\n• Llamar a Ana"),
        ("Sin nada que cambiar: 2 * 3 = 6.", "Sin nada que cambiar: 2 * 3 = 6."),
    ],
)
def test_la_respuesta_del_modelo_pasa_a_negritas_de_whatsapp(
    del_modelo: str, a_whatsapp: str
) -> None:
    assert _formato_de_whatsapp(del_modelo) == a_whatsapp


async def test_las_negritas_se_aplican_a_lo_que_redacta_el_modelo() -> None:
    grafo = GrafoFalso([AIMessage("Listo, creé «Torta» en **Recetas**.")])

    respuesta = await AgenteGemini(grafo).responder(_consulta())

    assert respuesta == "Listo, creé *Torta* en *Recetas*."


async def test_la_pregunta_de_confirmacion_se_manda_tal_cual() -> None:
    """Lo que se confirma es lo que corre (RF-08): la pregunta no se toca."""
    grafo = GrafoFalso()

    async def pausado(entrada: Any, config: dict[str, Any], context: Any = None) -> Any:
        resumen = "Crear en tu Notion la página «Torta» adentro de «Recetas»"
        return {"messages": [], "__interrupt__": [InterrupcionFalsa({"resumen": resumen})]}

    grafo.ainvoke = pausado  # type: ignore[method-assign]

    respuesta = await AgenteGemini(grafo).responder(_consulta("creá Torta en Recetas"))

    assert respuesta.startswith("Crear en tu Notion la página «Torta» adentro de «Recetas».")


async def test_las_negritas_se_ponen_antes_de_recortar() -> None:
    """Con `**` el texto no entra (4900); con `*`, sí (3500): no se corta de más."""
    grafo = GrafoFalso([AIMessage("**ab** " * 700)])

    respuesta = await AgenteGemini(grafo).responder(_consulta())

    assert respuesta == ("*ab* " * 700).strip()


async def test_un_mensaje_en_blanco_no_llega_al_modelo() -> None:
    """Es una llamada paga que además Gemini rechaza por contenido vacío."""
    grafo = GrafoFalso()

    respuesta = await AgenteGemini(grafo).responder(_consulta("   "))

    assert respuesta == TEXTO_VACIO
    assert grafo.invocaciones == []


# --- Errores (RF-19) ---------------------------------------------------------


@pytest.mark.parametrize(
    "falla",
    [TimeoutError("tardó"), ValueError("payload raro"), RuntimeError("vaya a saber")],
    ids=["timeout", "valor", "desconocida"],
)
async def test_cualquier_falla_del_grafo_es_agente_no_disponible(falla: Exception) -> None:
    """Una excepción nueva de la librería no puede volverse un 500 sin aviso."""
    grafo = GrafoFalso(fallar_con=falla)

    with pytest.raises(AgenteNoDisponibleError):
        await AgenteGemini(grafo).responder(_consulta())


async def test_la_falta_de_cuota_se_distingue_del_resto() -> None:
    """El usuario puede hacer algo —esperar—, así que merece otro mensaje."""
    error_real = (
        "Error calling model 'gemini-3.6-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
        "{'error': {'code': 429, 'message': 'You exceeded your current quota'}}"
    )
    grafo = GrafoFalso(fallar_con=RuntimeError(error_real))

    with pytest.raises(CuotaDeAgenteAgotadaError) as capturado:
        await AgenteGemini(grafo).responder(_consulta())

    assert "en un minuto" in capturado.value.mensaje_usuario


async def test_la_falta_de_cuota_sigue_siendo_agente_no_disponible() -> None:
    """El borde del webhook la trata igual: avisa y no reintenta el envío."""
    grafo = GrafoFalso(fallar_con=RuntimeError("429 RESOURCE_EXHAUSTED"))

    with pytest.raises(AgenteNoDisponibleError):
        await AgenteGemini(grafo).responder(_consulta())


async def test_el_texto_de_la_cuota_no_se_loguea() -> None:
    """Se inspecciona el mensaje de la librería, pero no se escribe (RF-18)."""
    grafo = GrafoFalso(fallar_con=RuntimeError("RESOURCE_EXHAUSTED: quota for AIzaSyFAKEKEY"))

    with (
        structlog.testing.capture_logs() as eventos,
        pytest.raises(CuotaDeAgenteAgotadaError),
    ):
        await AgenteGemini(grafo).responder(_consulta())

    # Sin esto la aserción negativa de abajo pasaría aunque `capture_logs`
    # no hubiera capturado nada. Ver `test_logging.py`, sección de la trampa.
    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "AIzaSyFAKEKEY" not in registrado
    assert '"sin_cuota": true' in registrado.lower()


async def test_el_error_no_arrastra_el_mensaje_de_la_libreria() -> None:
    """El texto de esa excepción puede traer el prompt, y el prompt trae al usuario."""
    grafo = GrafoFalso(fallar_con=ValueError("falló procesando: 'mi diagnostico medico'"))

    with (
        structlog.testing.capture_logs() as eventos,
        pytest.raises(AgenteNoDisponibleError),
    ):
        await AgenteGemini(grafo).responder(_consulta("mi diagnostico medico"))

    # Sin esto la aserción negativa de abajo pasaría aunque `capture_logs`
    # no hubiera capturado nada. Ver `test_logging.py`, sección de la trampa.
    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "mi diagnostico medico" not in registrado
    assert "agente.fallo" in registrado


# --- Logging (RF-18) ---------------------------------------------------------


async def test_no_se_loguea_ni_la_pregunta_ni_la_respuesta() -> None:
    grafo = GrafoFalso([AIMessage("tu turno es el martes a las 15")])

    with structlog.testing.capture_logs() as eventos:
        await AgenteGemini(grafo).responder(_consulta("cuando es mi turno medico"))

    # Sin esto la aserción negativa de abajo pasaría aunque `capture_logs`
    # no hubiera capturado nada. Ver `test_logging.py`, sección de la trampa.
    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "cuando es mi turno medico" not in registrado
    assert "tu turno es el martes" not in registrado


async def test_se_loguea_lo_necesario_para_diagnosticar() -> None:
    """Sin duración y sin tool calls no hay forma de saber por qué tarda."""
    mensajes = [
        AIMessage("", tool_calls=[{"name": "fecha_y_hora_actual", "args": {}, "id": "t1"}]),
        ToolMessage("lunes", tool_call_id="t1"),
        AIMessage("Hoy es lunes."),
    ]

    with structlog.testing.capture_logs() as eventos:
        await AgenteGemini(GrafoFalso(mensajes)).responder(_consulta())

    evento = next(e for e in eventos if e["event"] == "agente.respuesta")
    assert evento["cantidad_tool_calls"] == 1
    assert evento["largo_respuesta"] == len("Hoy es lunes.")
    assert isinstance(evento["duracion_ms"], int)


# --- Construcción ------------------------------------------------------------


def test_sin_api_key_no_se_puede_construir() -> None:
    """Falta de configuración, no de respuesta: por eso ServiceUnavailableError."""
    with pytest.raises(ServiceUnavailableError):
        crear_agente_gemini(Settings(_env_file=None, environment=Environment.TESTING))


@pytest.mark.parametrize(
    ("modelo", "esperado"),
    [
        ("gemini-3.6-flash", {"thinking_level": "minimal"}),
        ("gemini-3.5-flash", {"thinking_level": "minimal"}),
        ("gemini-2.5-flash", {"thinking_budget": 0}),
        ("gemini-2.5-flash-lite", {"thinking_budget": 0}),
        ("algun-modelo-futuro", {}),
    ],
    ids=["g3.6", "g3.5", "g2.5", "g2.5-lite", "desconocido"],
)
def test_cada_familia_recibe_su_parametro_de_razonamiento(
    modelo: str, esperado: dict[str, Any]
) -> None:
    """Son excluyentes: mandarle a Gemini 3 el parámetro de 2.5 da 400."""
    assert _ajustes_de_razonamiento(modelo) == esperado


def test_se_construye_con_el_modelo_configurado() -> None:
    settings = Settings(
        _env_file=None,
        environment=Environment.TESTING,
        google_api_key="key-de-prueba",
        gemini_model="gemini-2.5-flash",
    )

    agente = crear_agente_gemini(settings)

    assert isinstance(agente, AgenteGemini)


# --- Confirmaciones pendientes (PB-016, RF-08) -------------------------------
#
# La decisión de reanudar es determinística y del adaptador, nunca del modelo.
# El GrafoFalso registra los resumes: son la evidencia de qué se decidió.


async def test_un_si_reanuda_con_aprobacion() -> None:
    grafo = GrafoFalso(mensajes=[AIMessage("Listo, agendado.")], pendiente={"resumen": "Crear X"})

    respuesta = await AgenteGemini(grafo).responder(_consulta("sí"))

    assert grafo.resumes == [{"aprobado": True}]
    assert respuesta == "Listo, agendado."


async def test_un_no_reanuda_con_rechazo_y_termina_el_turno() -> None:
    grafo = GrafoFalso(mensajes=[AIMessage("Ok, no lo hago.")], pendiente={"resumen": "Crear X"})

    respuesta = await AgenteGemini(grafo).responder(_consulta("no"))

    assert grafo.resumes == [{"aprobado": False}]
    assert len(grafo.invocaciones) == 1  # nada más que el resume
    assert respuesta == "Ok, no lo hago."


async def test_otra_cosa_cancela_y_procesa_el_mensaje_como_turno_nuevo() -> None:
    """El caso que la sonda mostró peligroso: acá no corrompe nada."""
    grafo = GrafoFalso(
        mensajes=[AIMessage("Mañana tenés el turno.")], pendiente={"resumen": "Crear X"}
    )

    respuesta = await AgenteGemini(grafo).responder(_consulta("¿qué tengo mañana?"))

    assert grafo.resumes == [{"aprobado": False}]  # primero se cierra limpio
    assert len(grafo.invocaciones) == 2  # resume + turno nuevo
    entrada_final = grafo.invocaciones[1][0]
    assert isinstance(entrada_final, dict)  # un HumanMessage, no un Command
    assert respuesta == "Mañana tenés el turno."


async def test_una_confirmacion_vieja_se_cancela_sola() -> None:
    """Aprobar algo propuesto hace 10+ minutos sin re-mostrarlo es peligroso."""
    grafo = GrafoFalso(pendiente={"resumen": "Crear X"}, antiguedad_segundos=11 * 60)

    respuesta = await AgenteGemini(grafo).responder(_consulta("sí"))

    assert grafo.resumes == [{"aprobado": False}]  # ni con un "sí" se aprueba
    assert respuesta == CONFIRMACION_VENCIDA


async def test_una_confirmacion_vieja_no_secuestra_un_mensaje_normal() -> None:
    grafo = GrafoFalso(
        mensajes=[AIMessage("Tenés el dentista.")],
        pendiente={"resumen": "Crear X"},
        antiguedad_segundos=11 * 60,
    )

    respuesta = await AgenteGemini(grafo).responder(_consulta("¿qué tengo mañana?"))

    assert grafo.resumes == [{"aprobado": False}]
    assert respuesta == "Tenés el dentista."


async def test_sin_pendiente_no_se_manda_ningun_resume() -> None:
    grafo = GrafoFalso()

    await AgenteGemini(grafo).responder(_consulta("hola"))

    assert grafo.resumes == []


async def test_cuando_el_grafo_pausa_la_respuesta_sale_del_payload() -> None:
    """Lo que se confirma es exactamente lo que se ejecuta: texto nuestro, no del modelo."""
    grafo = GrafoFalso(mensajes=[AIMessage("")])
    grafo.interrumpir_proxima = True

    respuesta = await AgenteGemini(grafo).responder(_consulta("agendame X el lunes"))

    assert "Crear X el lunes" in respuesta
    assert "¿Confirmás?" in respuesta


async def test_un_interrupt_sin_resumen_igual_pide_confirmacion() -> None:
    """Un payload roto es un bug nuestro; la persona no se queda sin respuesta."""
    grafo = GrafoFalso(mensajes=[AIMessage("")])
    grafo.interrumpir_proxima = True

    async def ainvoke_roto(entrada: Any, config: dict[str, Any], context: Any = None) -> Any:
        grafo.invocaciones.append((entrada, config))
        return {"messages": [], "__interrupt__": [InterrupcionFalsa("no-es-un-dict")]}

    grafo.ainvoke = ainvoke_roto  # type: ignore[method-assign]

    respuesta = await AgenteGemini(grafo).responder(_consulta("agendame X"))

    assert "confirmes" in respuesta or "Confirmás" in respuesta


# --- Lote (RF-08, segunda versión) ---------------------------------------------


async def test_al_aprobar_un_lote_viajan_los_ids_que_la_persona_vio() -> None:
    """El grafo ejecuta esos y ninguno más: lo que se confirma es lo que se hace."""
    grafo = GrafoFalso(
        pendiente={"resumen": "Perfecto, agendo esto:\n• A\n• B", "lote": True, "ids": ["a", "b"]}
    )

    await AgenteGemini(grafo).responder(_consulta("sí"))

    assert grafo.resumes == [{"aprobado": True, "ids": ["a", "b"]}]


async def test_al_aprobar_una_sola_accion_no_viajan_ids() -> None:
    grafo = GrafoFalso(pendiente={"resumen": "Eliminar la tarea: X"})

    await AgenteGemini(grafo).responder(_consulta("sí"))

    assert grafo.resumes == [{"aprobado": True}]


async def test_la_pregunta_de_un_lote_cierra_con_su_frase() -> None:
    lista = "Perfecto, agendo esto:\n• Crear X\n• Anotar la tarea: Y"
    grafo = GrafoFalso()
    grafo.interrumpir_proxima = True

    async def con_lote(entrada: Any, config: dict[str, Any], context: Any = None) -> Any:
        return {
            "messages": [],
            "__interrupt__": [InterrupcionFalsa({"resumen": lista, "lote": True, "ids": ["1"]})],
        }

    grafo.ainvoke = con_lote  # type: ignore[method-assign]

    respuesta = await AgenteGemini(grafo).responder(_consulta("agendame X y anotá Y"))

    assert respuesta == f"{lista}\n\n{CIERRE_DEL_LOTE}"


# --- Archivos (PB-038) ------------------------------------------------------------

ADJUNTO = ArchivoAdjunto(
    referencia="1037543291543636", tipo_mime="application/pdf", nombre="informe.pdf"
)


def _con_archivo(texto: str = "") -> ConsultaDelUsuario:
    return ConsultaDelUsuario(
        conversacion_id=CONVERSACION, usuario_id=USUARIO, texto=texto, adjunto=ADJUNTO
    )


async def test_un_archivo_sin_pie_es_un_turno_y_no_un_texto_vacio() -> None:
    grafo = GrafoFalso()

    respuesta = await AgenteGemini(grafo).responder(_con_archivo())

    assert respuesta != TEXTO_VACIO
    [mensaje] = grafo.invocaciones[0][0]["messages"]
    assert mensaje.content.startswith("[Archivo recibido: «informe.pdf» (PDF)")


async def test_la_referencia_viaja_oculta_y_el_pie_va_despues_de_la_nota() -> None:
    grafo = GrafoFalso()

    await AgenteGemini(grafo).responder(_con_archivo("subilo a mi Drive"))

    [mensaje] = grafo.invocaciones[0][0]["messages"]
    assert mensaje.content.endswith("\nsubilo a mi Drive")
    assert ADJUNTO.referencia not in mensaje.content  # el modelo no la ve
    guardado = mensaje.additional_kwargs[CLAVE_DE_ARCHIVO]
    assert guardado["referencia"] == ADJUNTO.referencia
    assert guardado["nombre"] == "informe.pdf"
    # Sólo texto: un objeto propio volvería distinto después de un checkpoint.
    assert all(isinstance(valor, str) for valor in guardado.values())


@pytest.mark.parametrize("texto", ["sí", "no"])
async def test_un_archivo_nunca_resuelve_una_confirmacion_pendiente(texto: str) -> None:
    """Con pie "sí" no aprueba; con "no", el archivo no se pierde: entra igual."""
    grafo = GrafoFalso(pendiente={"resumen": "Borrar el dentista"})

    await AgenteGemini(grafo).responder(_con_archivo(texto))

    assert grafo.resumes == [{"aprobado": False}]
    entrada, _ = grafo.invocaciones[-1]
    assert entrada["messages"][0].additional_kwargs[CLAVE_DE_ARCHIVO]


async def test_un_archivo_con_si_a_una_confirmacion_vencida_entra_igual() -> None:
    grafo = GrafoFalso(pendiente={"resumen": "Borrar el dentista"}, antiguedad_segundos=10_000)

    respuesta = await AgenteGemini(grafo).responder(_con_archivo("sí"))

    assert respuesta != CONFIRMACION_VENCIDA
    entrada, _ = grafo.invocaciones[-1]
    assert entrada["messages"][0].additional_kwargs[CLAVE_DE_ARCHIVO]


class GrafoLento(GrafoFalso):
    """Tarda en contestar y cuenta cuántas invocaciones corren a la vez."""

    def __init__(self) -> None:
        super().__init__()
        self.activas = 0
        self.maximo_simultaneas = 0

    async def ainvoke(
        self, entrada: Any, config: dict[str, Any], context: Any = None
    ) -> dict[str, Any]:
        self.activas += 1
        self.maximo_simultaneas = max(self.maximo_simultaneas, self.activas)
        try:
            await asyncio.sleep(0.01)
            return await super().ainvoke(entrada, config, context)
        finally:
            self.activas -= 1


async def test_dos_turnos_de_la_misma_conversacion_no_corren_a_la_vez() -> None:
    """La foto y enseguida "subila": sin candado, los dos parten del mismo checkpoint."""
    grafo = GrafoLento()
    agente = AgenteGemini(grafo)

    await asyncio.gather(agente.responder(_con_archivo()), agente.responder(_consulta("subila")))

    assert grafo.maximo_simultaneas == 1


class CheckpointerFalso:
    """Anota cada hilo borrado en el registro de su grafo."""

    def __init__(self, registro: list[str], fallar_con: Exception | None = None) -> None:
        self.registro = registro
        self.fallar_con = fallar_con
        self.borrados: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.registro.append("borrado")
        self.borrados.append(thread_id)


class GrafoConMemoria(GrafoLento):
    """Recuerda si hubo un turno y anota en orden los turnos y los borrados."""

    def __init__(self, fallar_al_borrar: Exception | None = None) -> None:
        super().__init__()
        self.registro: list[str] = []
        self.checkpointer = CheckpointerFalso(self.registro, fallar_al_borrar)

    async def ainvoke(
        self, entrada: Any, config: dict[str, Any], context: Any = None
    ) -> dict[str, Any]:
        resultado = await super().ainvoke(entrada, config, context)
        self.registro.append("turno")
        return resultado

    async def aget_state(self, config: dict[str, Any]) -> EstadoFalso:
        hubo = self.registro.count("turno") > self.registro.count("borrado")
        return EstadoFalso((), None, {"messages": self.mensajes} if hubo else None)


async def test_olvidar_espera_al_turno_en_curso() -> None:
    """Cada checkpoint guarda la lista entera: un turno sin terminar reescribiría lo borrado."""
    grafo = GrafoConMemoria()
    agente = AgenteGemini(grafo)

    _, habia = await asyncio.gather(agente.responder(_consulta()), agente.olvidar(CONVERSACION))

    assert habia is True
    assert grafo.registro == ["turno", "borrado"]
    assert grafo.checkpointer.borrados == [str(CONVERSACION)]


async def test_olvidar_sin_conversacion_no_borra_nada() -> None:
    grafo = GrafoConMemoria()

    assert await AgenteGemini(grafo).olvidar(CONVERSACION) is False
    assert grafo.checkpointer.borrados == []


async def test_si_no_se_puede_borrar_sube_como_servicio_no_disponible() -> None:
    grafo = GrafoConMemoria(fallar_al_borrar=RuntimeError("se cayó la base"))
    agente = AgenteGemini(grafo)
    await agente.responder(_consulta())

    with structlog.testing.capture_logs() as eventos:
        with pytest.raises(ServiceUnavailableError):
            await agente.olvidar(CONVERSACION)

    [fallo] = [e for e in eventos if e["event"] == "agente.olvido_fallido"]
    assert fallo["tipo"] == "RuntimeError"


async def test_conversaciones_distintas_no_se_esperan() -> None:
    grafo = GrafoLento()
    agente = AgenteGemini(grafo)
    otra = ConsultaDelUsuario(conversacion_id=uuid4(), usuario_id=uuid4(), texto="hola")

    await asyncio.gather(agente.responder(_consulta()), agente.responder(otra))

    assert grafo.maximo_simultaneas == 2


def test_gemini_no_recibe_lo_que_viaja_en_additional_kwargs() -> None:
    """Canario de la suposición de PB-038, contra la librería instalada.

    Si una versión nueva de la integración empezara a mandar los
    `additional_kwargs` de la persona, el modelo vería la referencia del archivo
    (y podría elegirla). Este test lo delata en la actualización.
    """
    from langchain_google_genai.chat_models import _parse_chat_history

    oculto = {CLAVE_DE_ARCHIVO: {"referencia": "REFERENCIA-OCULTA"}}
    _, contenidos = _parse_chat_history([HumanMessage("hola", additional_kwargs=oculto)])

    assert "hola" in repr(contenidos)
    assert "REFERENCIA-OCULTA" not in repr(contenidos)
