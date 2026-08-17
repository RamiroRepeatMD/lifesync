"""Adaptador del agente conversacional sobre Gemini (PB-005).

Implementa el puerto `AgenteConversacional` invocando el grafo de LangGraph.
Su trabajo es el de todo adaptador: traducir entre el vocabulario de la
aplicación —una consulta, una respuesta— y el de la librería, y **contener sus
fallas** para que no se filtren hacia adentro.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.ports.agente import AgenteConversacional
from src.domain.exceptions import (
    AgenteNoDisponibleError,
    CuotaDeAgenteAgotadaError,
    ServiceUnavailableError,
)
from src.infrastructure.config.settings import Settings
from src.infrastructure.llm.grafo import LIMITE_DE_PASOS, construir_grafo
from src.infrastructure.llm.herramientas import HERRAMIENTAS

logger = structlog.get_logger(__name__)

# Tope duro de la Cloud API para el cuerpo de un mensaje de texto. Un modelo
# suelto puede pasarse; si eso llega a Meta, el envío se rechaza entero y la
# persona no recibe nada. Mejor una respuesta cortada que ninguna.
LARGO_MAXIMO_WHATSAPP = 4096

# Cuánto esperamos al modelo antes de darlo por perdido. El RNF de eficiencia
# pide responder en ≤ 3 s y el resto del circuito ya consume ~1,7 s, así que
# esto no es el objetivo sino el techo: a partir de acá la espera es peor que
# el aviso de error.
TIMEOUT_MODELO_SEGUNDOS = 10.0

# Un reintento, y no más. Probando contra la API real aparecieron
# `504 DEADLINE_EXCEEDED` propios de Google en turnos que después anduvieron
# bien: para eso sirve. Subirlo sería contraproducente, porque el otro fallo
# frecuente es el 429 por cuota, y ahí cada reintento gasta una petición más de
# las pocas que quedan (ver CUOTA_AGOTADA).
REINTENTOS_MODELO = 1

# Cómo se reconoce que se acabó la cuota. Es el nombre canónico del status en
# la API de Google, estable entre versiones; se compara contra el texto porque
# `ChatGoogleGenerativeAIError` no expone ni el código ni el status como
# atributo: lo único que trae es el mensaje.
#
# Medido en el plan gratuito: 20 peticiones y a esperar ~22 s. Alcanza de sobra
# para una persona escribiendo por WhatsApp, pero no para una ráfaga de pruebas.
CUOTA_AGOTADA = "RESOURCE_EXHAUSTED"

# Respuestas cortas, de chat. Además acota la latencia, que es lo que aprieta.
MAX_TOKENS_DE_SALIDA = 512

SIN_CONTENIDO = (
    "Me quedé sin respuesta para eso. ¿Lo probamos de otra manera o me lo contás distinto?"
)

TEXTO_VACIO = "No te llegué a leer. ¿Me lo escribís de nuevo?"


class AgenteGemini(AgenteConversacional):
    """Responde invocando el grafo de LangGraph sobre Gemini."""

    def __init__(self, grafo: Any) -> None:
        """Recibe el grafo ya compilado (inyección explícita)."""
        self._grafo = grafo

    async def responder(self, consulta: ConsultaDelUsuario) -> str:
        """Corre un turno de conversación y devuelve el texto de respuesta."""
        if not consulta.texto.strip():
            # Sin esto, un mensaje en blanco se convierte en una llamada paga
            # que Gemini además rechaza por contenido vacío.
            logger.info("agente.consulta_vacia", conversacion_id=str(consulta.conversacion_id))
            return TEXTO_VACIO

        comenzo = time.perf_counter()
        estado = await self._invocar(consulta)
        duracion_ms = round((time.perf_counter() - comenzo) * 1000)

        mensajes: list[BaseMessage] = estado.get("messages", [])
        texto = _recortar(_texto_de(mensajes[-1]) if mensajes else "")

        logger.info(
            "agente.respuesta",
            conversacion_id=str(consulta.conversacion_id),
            duracion_ms=duracion_ms,
            # Nunca el texto: ni el de la persona ni el del modelo (RF-18).
            largo_respuesta=len(texto),
            cantidad_tool_calls=_contar_tool_calls(mensajes),
            mensajes_en_el_hilo=len(mensajes),
        )

        return texto or SIN_CONTENIDO

    async def _invocar(self, consulta: ConsultaDelUsuario) -> dict[str, Any]:
        """Llama al grafo, traduciendo cualquier falla a un error del dominio.

        Se atrapa `Exception` a propósito y no una lista de tipos: entre
        LangGraph, LangChain, `google-genai` y la red hay decenas de
        excepciones posibles, y que aparezca una nueva no puede convertirse en
        un 500 sin aviso para la persona.
        """
        configuracion = {
            "configurable": {"thread_id": _hilo_de(consulta.conversacion_id)},
            "recursion_limit": LIMITE_DE_PASOS,
        }
        try:
            estado: dict[str, Any] = await self._grafo.ainvoke(
                {"messages": [HumanMessage(consulta.texto)]},
                config=configuracion,
            )
        except Exception as exc:
            sin_cuota = _es_falta_de_cuota(exc)
            # Se loguea el tipo y la clasificación, nunca `str(exc)`: el
            # mensaje de error de la librería puede incluir el prompt, y el
            # prompt lleva lo que escribió la persona (RF-18).
            logger.error(
                "agente.fallo",
                conversacion_id=str(consulta.conversacion_id),
                tipo=type(exc).__name__,
                sin_cuota=sin_cuota,
            )
            if sin_cuota:
                raise CuotaDeAgenteAgotadaError("Se agotó la cuota del modelo.") from None
            raise AgenteNoDisponibleError("El modelo no pudo responder.") from None
        return estado


def _es_falta_de_cuota(exc: Exception) -> bool:
    """Distingue "se acabó la cuota" de cualquier otra falla del modelo.

    El texto se inspecciona pero **no se loguea**: puede traer el prompt.
    """
    return CUOTA_AGOTADA in str(exc)


def _hilo_de(conversacion_id: UUID) -> str:
    """Traduce el id de conversación al `thread_id` que espera LangGraph."""
    return str(conversacion_id)


def _texto_de(mensaje: BaseMessage) -> str:
    """Saca el texto plano de un mensaje del modelo.

    `content` puede ser un string o una lista de bloques —Gemini usa bloques
    cuando mezcla texto con otras partes—, así que hay que contemplar las dos
    formas o un día la respuesta sale como `[{'type': 'text', ...}]`.
    """
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido.strip()

    partes = [
        bloque.get("text", "")
        for bloque in contenido
        if isinstance(bloque, dict) and bloque.get("type") == "text"
    ]
    return "\n".join(parte for parte in partes if parte).strip()


def _recortar(texto: str) -> str:
    """Corta la respuesta al máximo que acepta WhatsApp, avisando por log."""
    if len(texto) <= LARGO_MAXIMO_WHATSAPP:
        return texto

    logger.warning("agente.respuesta_recortada", largo_original=len(texto))
    return texto[: LARGO_MAXIMO_WHATSAPP - 1].rstrip() + "…"


def _contar_tool_calls(mensajes: list[BaseMessage]) -> int:
    """Cuenta las herramientas que el modelo pidió usar en este turno."""
    return sum(len(m.tool_calls) for m in mensajes if isinstance(m, AIMessage) and m.tool_calls)


def _ajustes_de_razonamiento(modelo: str) -> dict[str, Any]:
    """Baja al mínimo el razonamiento previo, que es lo que cuesta segundos.

    Los Flash "piensan" antes de contestar por defecto y eso es la diferencia
    entre cumplir el RNF de ≤ 3 s y no cumplirlo. Medido contra la API real,
    con el system prompt de LifeSync y dos preguntas de chat:

        gemini-3.6-flash, por defecto           16.100 ms
        gemini-3.6-flash, thinking_level=low     1.878 / 17.890 ms
        gemini-3.6-flash, thinking_level=minimal 2.140 / 1.147 ms

    `low` quedó descartado no por lento sino por **impredecible**: el segundo
    turno se fue a 17 s. Un asistente de chat necesita techo, no promedio.

    Cada familia usa su propio parámetro, y son excluyentes: mandar el que no
    corresponde da 400.
    """
    if modelo.startswith("gemini-2.5"):
        return {"thinking_budget": 0}
    if modelo.startswith("gemini-3"):
        return {"thinking_level": "minimal"}
    return {}


def crear_agente_gemini(settings: Settings) -> AgenteGemini:
    """Construye el agente completo: modelo, memoria y grafo.

    Se llama una sola vez, en el `lifespan`. El `InMemorySaver` vive tanto como
    el proceso: crear uno por mensaje sería empezar cada conversación de cero.

    Raises:
        ServiceUnavailableError: Si falta `GOOGLE_API_KEY`.
    """
    clave = settings.google_api_key
    if clave is None:
        raise ServiceUnavailableError("Falta GOOGLE_API_KEY para usar el agente conversacional.")

    # Import diferido: mantiene el arranque en modo degradado libre del stack de
    # Gemini, que tarda casi un segundo en importarse.
    from langchain_google_genai import ChatGoogleGenerativeAI

    modelo = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=clave.get_secret_value(),
        timeout=TIMEOUT_MODELO_SEGUNDOS,
        max_retries=REINTENTOS_MODELO,
        max_output_tokens=MAX_TOKENS_DE_SALIDA,
        **_ajustes_de_razonamiento(settings.gemini_model),
    )

    grafo = construir_grafo(modelo, HERRAMIENTAS, InMemorySaver())
    logger.info("agente.creado", modelo=settings.gemini_model)
    return AgenteGemini(grafo)
