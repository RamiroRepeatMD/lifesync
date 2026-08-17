"""Tests del adaptador del agente sobre Gemini (PB-005).

El grafo se reemplaza por un doble: acá no se prueba LangGraph —eso es
`test_grafo_llm.py`— sino lo que el adaptador hace alrededor. Que es lo de
siempre en un adaptador: contener las fallas de la librería y no dejar salir
nada que no deba salir.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
import structlog
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.domain.exceptions import (
    AgenteNoDisponibleError,
    CuotaDeAgenteAgotadaError,
    ServiceUnavailableError,
)
from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.llm.agente_gemini import (
    LARGO_MAXIMO_WHATSAPP,
    SIN_CONTENIDO,
    TEXTO_VACIO,
    AgenteGemini,
    _ajustes_de_razonamiento,
    crear_agente_gemini,
)
from src.infrastructure.llm.grafo import LIMITE_DE_PASOS

CONVERSACION = uuid4()


class GrafoFalso:
    """Doble del grafo compilado: registra la invocación y devuelve un estado."""

    def __init__(
        self, mensajes: list[Any] | None = None, fallar_con: Exception | None = None
    ) -> None:
        self.mensajes = mensajes if mensajes is not None else [AIMessage("hola")]
        self.fallar_con = fallar_con
        self.invocaciones: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, entrada: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.invocaciones.append((entrada, config))
        if self.fallar_con is not None:
            raise self.fallar_con
        return {"messages": self.mensajes}


def _consulta(texto: str = "hola") -> ConsultaDelUsuario:
    return ConsultaDelUsuario(conversacion_id=CONVERSACION, texto=texto, nombre_usuario="Ramiro")


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

    registrado = json.dumps(eventos, default=str)
    assert "mi diagnostico medico" not in registrado
    assert "agente.fallo" in registrado


# --- Logging (RF-18) ---------------------------------------------------------


async def test_no_se_loguea_ni_la_pregunta_ni_la_respuesta() -> None:
    grafo = GrafoFalso([AIMessage("tu turno es el martes a las 15")])

    with structlog.testing.capture_logs() as eventos:
        await AgenteGemini(grafo).responder(_consulta("cuando es mi turno medico"))

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
