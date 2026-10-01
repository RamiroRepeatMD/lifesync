"""Adaptador del agente conversacional sobre Gemini (PB-005).

Implementa el puerto `AgenteConversacional` invocando el grafo de LangGraph.
Su trabajo es el de todo adaptador: traducir entre el vocabulario de la
aplicación —una consulta, una respuesta— y el de la librería, y **contener sus
fallas** para que no se filtren hacia adentro.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.ports.agente import AgenteConversacional
from src.application.ports.calendario import Calendario
from src.application.ports.correos import Correos
from src.application.ports.tareas import Tareas
from src.domain.exceptions import (
    AgenteNoDisponibleError,
    CuotaDeAgenteAgotadaError,
    ServiceUnavailableError,
)
from src.domain.repositories.recordatorio_repository import RecordatorioRepository
from src.infrastructure.config.settings import Settings
from src.infrastructure.llm.confirmacion import (
    VIGENCIA_DE_CONFIRMACION_SEGUNDOS,
    Decision,
    clasificar,
)
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import LIMITE_DE_PASOS, NODO_AGENTE, construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas

logger = structlog.get_logger(__name__)

# Tope duro de la Cloud API para el cuerpo de un mensaje de texto. Un modelo
# suelto puede pasarse; si eso llega a Meta, el envío se rechaza entero y la
# persona no recibe nada. Mejor una respuesta cortada que ninguna.
LARGO_MAXIMO_WHATSAPP = 4096

# Cuánto esperamos al modelo antes de darlo por perdido. NO es el objetivo de
# latencia (eso lo mide duracion_ms contra el RNF de 3 s): es la red de
# contención. Se subió de 10 a 18 con evidencia: contra la API real se
# midieron llamadas legítimas de 13-16 s (primeras llamadas, colas del free
# tier), y con 10 s el ReadTimeout las cortaba y dejaba a la persona sin
# respuesta — peor que esperar.
TIMEOUT_MODELO_SEGUNDOS = 18.0

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

CONFIRMACION_VENCIDA = (
    "Esa confirmación quedó vieja, así que la cancelé. Si todavía lo querés, pedímelo de nuevo."
)


class AgenteGemini(AgenteConversacional):
    """Responde invocando el grafo de LangGraph sobre Gemini."""

    def __init__(self, grafo: Any) -> None:
        """Recibe el grafo ya compilado (inyección explícita)."""
        self._grafo = grafo

    async def responder(self, consulta: ConsultaDelUsuario) -> str:
        """Corre un turno de conversación y devuelve el texto de respuesta.

        Si el hilo quedó pausado esperando una confirmación (RF-08), este turno
        la resuelve **antes** de cualquier otra cosa. No es un detalle: la
        sonda de diseño mostró que un mensaje común sobre un hilo interrumpido
        deja el interrupt en el limbo y el historial malformado —un AIMessage
        con tool_calls sin su ToolMessage—, y Gemini rechaza ese historial con
        400 en el turno siguiente. Acá no existe ese camino: o se reanuda con
        una decisión, o no se invoca.
        """
        if not consulta.texto.strip():
            # Sin esto, un mensaje en blanco se convierte en una llamada paga
            # que Gemini además rechaza por contenido vacío.
            logger.info("agente.consulta_vacia", conversacion_id=str(consulta.conversacion_id))
            return TEXTO_VACIO

        comenzo = time.perf_counter()
        configuracion = {
            "configurable": {"thread_id": _hilo_de(consulta.conversacion_id)},
            "recursion_limit": LIMITE_DE_PASOS,
        }
        contexto = ContextoDeAgente(usuario_id=consulta.usuario_id)

        try:
            estado = await self._resolver_pendiente(consulta, configuracion, contexto)
            if estado is None:
                # No había confirmación pendiente, o había y se canceló para dar
                # paso a este mensaje: turno normal.
                estado = await self._invocar(
                    {"messages": [HumanMessage(consulta.texto)]}, configuracion, contexto
                )
        except AgenteNoDisponibleError:
            # Incluye la cuota agotada, que es subclase. Si una escritura YA se
            # ejecutó en esta invocación, avisar un error invitaría a repetirla
            # —y a duplicarla—: se cuenta lo que se hizo (PB-026).
            if not contexto.acciones_realizadas:
                raise
            estado = await self._cerrar_turno_sin_modelo(consulta, configuracion, contexto)

        texto = self._respuesta_de(estado, contexto.acciones_realizadas)
        duracion_ms = round((time.perf_counter() - comenzo) * 1000)

        mensajes: list[BaseMessage] = estado.get("messages", [])
        logger.info(
            "agente.respuesta",
            conversacion_id=str(consulta.conversacion_id),
            duracion_ms=duracion_ms,
            # Nunca el texto: ni el de la persona ni el del modelo (RF-18).
            largo_respuesta=len(texto),
            cantidad_tool_calls=_contar_tool_calls(mensajes),
            mensajes_en_el_hilo=len(mensajes),
            pidio_confirmacion="__interrupt__" in estado,
        )

        return texto

    # --- Confirmaciones pendientes (PB-016, RF-08) ------------------------

    async def _resolver_pendiente(
        self,
        consulta: ConsultaDelUsuario,
        configuracion: dict[str, Any],
        contexto: ContextoDeAgente,
    ) -> dict[str, Any] | None:
        """Si el hilo está pausado, decide qué hacer con este mensaje.

        Returns:
            El estado resultante de reanudar, o **None** si no había nada
            pendiente —o si lo pendiente se canceló y este mensaje debe
            procesarse como un turno normal.

        La decisión de reanudar es **determinística, nunca del modelo**:
        clasificar un "sí" es una regla, no una interpretación. Y cancelar es
        el default ante cualquier cosa que no sea un sí explícito.
        """
        pendiente = await self._interrupcion_pendiente(configuracion)
        if pendiente is None:
            return None

        vencida = pendiente >= VIGENCIA_DE_CONFIRMACION_SEGUNDOS
        decision = clasificar(consulta.texto)

        if vencida:
            # Aprobar con un "sí" algo propuesto hace 10 minutos, sin volver a
            # mostrarlo, es ejecutar lo que la persona quizás ya ni recuerda.
            await self._invocar(Command(resume={"aprobado": False}), configuracion, contexto)
            logger.info(
                "agente.confirmacion_vencida", conversacion_id=str(consulta.conversacion_id)
            )
            if decision is Decision.OTRA_COSA:
                return None  # el mensaje merece su turno normal
            return {"messages": [], "_texto_directo": CONFIRMACION_VENCIDA}

        if decision is Decision.APRUEBA:
            logger.info(
                "agente.confirmacion_aprobada", conversacion_id=str(consulta.conversacion_id)
            )
            return await self._invocar(Command(resume={"aprobado": True}), configuracion, contexto)

        # Rechazo explícito u otra cosa: en los dos casos se cancela. La
        # diferencia es sólo qué pasa después.
        logger.info(
            "agente.confirmacion_cancelada",
            conversacion_id=str(consulta.conversacion_id),
            explicita=decision is Decision.RECHAZA,
        )
        estado = await self._invocar(Command(resume={"aprobado": False}), configuracion, contexto)
        if decision is Decision.RECHAZA:
            return estado
        return None  # cancelada en silencio; el mensaje se procesa como turno nuevo

    async def _interrupcion_pendiente(self, configuracion: dict[str, Any]) -> float | None:
        """Devuelve la antigüedad en segundos del interrupt pendiente, o None.

        Tolera cualquier fallo consultando el estado: ante la duda se asume
        que no hay nada pendiente, que es el camino que no ejecuta acciones.
        """
        try:
            estado = await self._grafo.aget_state(configuracion)
        except Exception:  # un hilo nuevo o un checkpointer vacío no es un error
            return None

        if not getattr(estado, "interrupts", ()):
            return None

        creado = getattr(estado, "created_at", None)
        if isinstance(creado, str):
            try:
                momento = datetime.fromisoformat(creado)
                return max(0.0, (datetime.now(UTC) - momento).total_seconds())
            except ValueError:
                pass
        # Sin timestamp legible se trata como recién creada: mejor pedir la
        # confirmación de nuevo que ejecutar por un dato que no se pudo leer.
        return 0.0

    async def _cerrar_turno_sin_modelo(
        self,
        consulta: ConsultaDelUsuario,
        configuracion: dict[str, Any],
        contexto: ContextoDeAgente,
    ) -> dict[str, Any]:
        """Responde sin el modelo cuando falló DESPUÉS de una escritura (PB-026).

        Cuenta lo que se hizo y deja el turno cerrado en el historial, como si
        el agente lo hubiera dicho. Sin ese cierre el hilo quedaría con el paso
        del modelo pendiente, y el turno siguiente no sabría qué se le contestó
        a la persona. El cierre es best effort: si falla, la persona igual
        recibe la respuesta correcta.
        """
        texto = _tras_accion_sin_redaccion(contexto.acciones_realizadas)
        logger.warning(
            "agente.redaccion_fallida_tras_accion",
            conversacion_id=str(consulta.conversacion_id),
            # La cantidad, nunca los textos: llevan títulos (RF-18).
            acciones=len(contexto.acciones_realizadas),
        )
        try:
            await self._grafo.aupdate_state(
                configuracion, {"messages": [AIMessage(texto)]}, as_node=NODO_AGENTE
            )
        except Exception as exc:
            logger.warning("agente.cierre_de_turno_fallido", tipo=type(exc).__name__)
        return {"messages": [], "_texto_directo": texto}

    # --- Invocación y extracción ------------------------------------------

    async def _invocar(
        self,
        entrada: Any,
        configuracion: dict[str, Any],
        contexto: ContextoDeAgente,
    ) -> dict[str, Any]:
        """Llama al grafo, traduciendo cualquier falla a un error del dominio.

        Se atrapa `Exception` a propósito y no una lista de tipos: entre
        LangGraph, LangChain, `google-genai` y la red hay decenas de
        excepciones posibles, y que aparezca una nueva no puede convertirse en
        un 500 sin aviso para la persona.
        """
        try:
            estado: dict[str, Any] = await self._grafo.ainvoke(
                entrada,
                config=configuracion,
                # De quién es la conversación viaja por acá y no por el mensaje:
                # es lo que impide que el texto del usuario elija de quién es la
                # agenda que se consulta o modifica (PB-015 · PB-016).
                context=contexto,
            )
        except Exception as exc:
            sin_cuota = _es_falta_de_cuota(exc)
            # Se loguea el tipo y la clasificación, nunca `str(exc)`: el
            # mensaje de error de la librería puede incluir el prompt, y el
            # prompt lleva lo que escribió la persona (RF-18).
            logger.error("agente.fallo", tipo=type(exc).__name__, sin_cuota=sin_cuota)
            if sin_cuota:
                raise CuotaDeAgenteAgotadaError("Se agotó la cuota del modelo.") from None
            raise AgenteNoDisponibleError("El modelo no pudo responder.") from None
        return estado

    def _respuesta_de(self, estado: dict[str, Any], acciones: Sequence[str] = ()) -> str:
        """Extrae el texto que hay que mandarle a la persona.

        Si el grafo quedó pausado esperando confirmación, la respuesta sale
        **del payload del interrupt y no del modelo**: lo que la persona
        confirma es exactamente lo que se va a ejecutar, sin reinterpretación.
        """
        directo = estado.get("_texto_directo")
        if isinstance(directo, str):
            return directo

        interrupciones = estado.get("__interrupt__")
        if interrupciones:
            payload = getattr(interrupciones[0], "value", None)
            resumen = payload.get("resumen") if isinstance(payload, dict) else None
            if isinstance(resumen, str):
                # El "." sólo en resúmenes de una línea: en uno de varias
                # (un correo, PB-032) se pegaría al final del texto confirmado.
                cierre = "" if "\n" in resumen or resumen.endswith((".", "!", "?")) else "."
                pregunta = f"{resumen}{cierre}\n\n¿Confirmás? Respondé sí o no."
                if acciones:
                    # Pedido compuesto: lo primero ya se hizo y el grafo se
                    # volvió a pausar por lo segundo. Sin esto, la persona
                    # sólo vería la pregunta nueva (PB-026).
                    return f"{_lista_de_acciones(acciones)}\n\n{pregunta}"
                return pregunta
            # Un interrupt sin resumen es un bug nuestro, pero la persona no
            # puede quedarse sin respuesta por eso.
            return "Necesito que me confirmes la acción. ¿Sí o no?"

        mensajes: list[BaseMessage] = estado.get("messages", [])
        texto = _recortar(_texto_de(mensajes[-1]) if mensajes else "")
        return texto or SIN_CONTENIDO


def _lista_de_acciones(acciones: Sequence[str]) -> str:
    """Las acciones ya hechas, como las lee la persona."""
    if len(acciones) == 1:
        return f"Listo — {acciones[0]}."
    return "Listo:\n" + "\n".join(f"- {accion}" for accion in acciones)


def _tras_accion_sin_redaccion(acciones: Sequence[str]) -> str:
    """El reemplazo de la respuesta cuando el modelo falló tras una escritura."""
    una = len(acciones) == 1
    return (
        f"{_lista_de_acciones(acciones)}\n\n"
        "Tuve un problema técnico al armar la respuesta, pero "
        + (
            "la acción ya se realizó: no hace falta repetirla."
            if una
            else "las acciones ya se realizaron: no hace falta repetirlas."
        )
    )


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


def crear_agente_gemini(
    settings: Settings,
    calendario: Calendario | None = None,
    tareas: Tareas | None = None,
    correos: Correos | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    recordatorios: RecordatorioRepository | None = None,
) -> AgenteGemini:
    """Construye el agente completo: modelo, herramientas, memoria y grafo.

    Se llama una sola vez, en el `lifespan`.

    Args:
        settings: Configuración; de acá salen la API key y el modelo.
        calendario: Adaptador de lectura de calendario. Si es `None` —porque
            falta la configuración de OAuth— el agente simplemente no ofrece
            esa herramienta, en vez de ofrecerla y fallar en cada uso.
        checkpointer: Dónde viven las conversaciones y las confirmaciones
            pendientes (PB-013). Con `None` se usa memoria RAM, que no
            sobrevive a un redeploy: es el modo degradado, no el normal.
        recordatorios: Dónde se guardan los recordatorios (PB-030). Con
            `None` —sin Supabase— el agente no ofrece programarlos.

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

    herramientas = construir_herramientas(calendario, tareas, correos, recordatorios)
    grafo = construir_grafo(
        modelo, herramientas, checkpointer if checkpointer is not None else InMemorySaver()
    )
    logger.info(
        "agente.creado",
        modelo=settings.gemini_model,
        con_calendario=calendario is not None,
        con_tareas=tareas is not None,
        con_correo=correos is not None,
        con_recordatorios=recordatorios is not None,
        memoria_persistida=checkpointer is not None,
    )
    return AgenteGemini(grafo)
