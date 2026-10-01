"""Grafo del agente conversacional (PB-005).

La forma es el ciclo clásico de tool-calling, mínimo pero completo:

    START ──▶ agente ──(¿pidió herramientas?)──▶ herramientas ──┐
                ▲                                                │
                └────────────────────────────────────────────────┘
                              │ no
                              ▼
                             END

Sumar una capacidad nueva es agregar una herramienta a la lista: la topología
no cambia. Ése es el motivo de armar el grafo a mano en vez de usar
`create_react_agent`, además de que así se puede explicar qué hace cada nodo.

**El modelo entra por parámetro, no se construye acá.** Eso es lo que permite
ejercitar el grafo entero —incluido el ciclo de herramientas— con un modelo
falso, sin red y sin API key. Es el mismo reparto que en WhatsApp, donde
`create_whatsapp_client` arma el cliente y `ClienteWhatsApp` lo recibe.

**La política de RF-08 se aplica acá** (`domain/services/
politica_de_confirmacion.py`): el nodo de herramientas decide, antes de
ejecutar, en qué fase corren las escrituras del paso.

- Una escritura reversible suelta corre DIRECTO; una irreversible (borrar,
  enviar) pausa para el sí.
- **Varias escrituras juntas se confirman en UNA sola pausa**, con la lista
  completa: primero cada herramienta arma su resumen sin escribir nada (vista
  previa), después el nodo pausa una vez, y con el sí ejecuta cada una una
  sola vez. Ninguna herramienta pausa por su cuenta dentro de un lote, y eso
  es lo que impide el bug del 30/09: LangGraph re-ejecuta el paso entero al
  reanudar cada pausa, y con dos pausas en un paso un evento se creó dos veces
  (y en la reproducción, un solo "sí" mandó dos correos).
- Con texto de un tercero a la vista del modelo (un correo leído), toda
  escritura confirma: es la barrera contra la inyección por correo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import get_runtime
from langgraph.types import interrupt

from src.domain.services.politica_de_confirmacion import (
    UMBRAL_DE_LOTE,
    TipoDeEscritura,
    requiere_confirmacion,
)
from src.infrastructure.llm.contexto import ContextoDeAgente, Fase, ModoDeConfirmacion
from src.infrastructure.llm.herramientas import LECTURAS_DE_TERCEROS, TIPO_POR_HERRAMIENTA
from src.infrastructure.llm.prompt import instrucciones

logger = structlog.get_logger(__name__)

# Cuántos mensajes del historial se le mandan al modelo. Es el techo de RF-09:
# sin él, una conversación larga crece hasta agotar la ventana de contexto y
# encarece cada turno. 20 son unas diez idas y vueltas, de sobra para el MVP.
MAX_MENSAJES_DE_HISTORIAL = 20

# Tope de pasos por turno. Sin esto, un modelo que insiste en llamar a la misma
# herramienta deja el grafo girando: son llamadas pagas y tiempo que el usuario
# está esperando. LangGraph lanza `GraphRecursionError` al superarlo.
LIMITE_DE_PASOS = 8

NODO_AGENTE = "agente"
NODO_HERRAMIENTAS = "herramientas"

# Lo que recibe el modelo por cada escritura de un lote que la persona rechazó.
LOTE_CANCELADO = "La persona canceló el pedido completo. No se hizo nada."

# Para una escritura que no estaba en la lista que la persona vio y aprobó:
# no se ejecuta, aunque ahora sí valide. Lo que se confirma es lo que se hace.
NO_ESTABA_EN_LA_LISTA = "No se hizo: no estaba en la lista que la persona confirmó."


def construir_grafo(
    modelo: BaseChatModel,
    herramientas: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver[Any],
    tipos: Mapping[str, TipoDeEscritura] = TIPO_POR_HERRAMIENTA,
) -> Any:
    """Arma y compila el grafo del agente.

    Args:
        modelo: El chat model ya configurado. No se toca su configuración acá.
        herramientas: Las que el modelo puede invocar. Puede venir vacía: el
            grafo sigue siendo válido y la rama de herramientas nunca se toma.
        checkpointer: Dónde vive el historial de cada conversación (RF-09). El
            llamador elige la implementación; el grafo no se entera de si
            sobrevive a un reinicio.
        tipos: Qué le hace a los datos cada herramienta que escribe; lo que
            no esté acá es una lectura. Es lo que mira la política de RF-08.

    Returns:
        El grafo compilado, listo para `ainvoke`. El tipo concreto de LangGraph
        es genérico y cambia entre versiones menores; anotarlo acá ataría el
        módulo a un detalle interno de la librería.
    """
    modelo_con_herramientas = modelo.bind_tools(herramientas) if herramientas else modelo

    # El parámetro se llama `state` y no `estado`, contra la convención del
    # resto del código: LangGraph valida los nodos contra un Protocol cuyo
    # `__call__(self, state: ...)` no es posicional-only, así que el nombre
    # forma parte del contrato y con otro el grafo no tipa.
    async def nodo_agente(state: MessagesState) -> MessagesState:
        """Le pregunta al modelo qué contestar o qué herramienta usar.

        Devuelve el estado completo y no un `dict` suelto: el reducer
        `add_messages` se encarga de sumar la respuesta al historial en vez de
        pisarlo.
        """
        historial = _historial_visible(state["messages"])
        # Las instrucciones se rearman en cada paso para que lleven la fecha
        # de hoy: sin eso el modelo tendría que gastar un viaje extra
        # preguntándola con una herramienta. No se persisten en el historial.
        respuesta = await modelo_con_herramientas.ainvoke(
            [SystemMessage(instrucciones(datetime.now(UTC))), *historial]
        )
        return {"messages": [respuesta]}

    ejecutor = ToolNode(herramientas)

    async def correr(
        pedido: AIMessage,
        previos: list[AnyMessage],
        llamadas: list[ToolCall],
        modo: ModoDeConfirmacion,
        fase: Fase,
        config: RunnableConfig,
    ) -> list[AnyMessage]:
        """Ejecuta un subconjunto de las llamadas del modelo, en una fase dada.

        Se ejecuta una copia filtrada del mensaje: el original NO se recorta,
        porque Gemini guarda firmas por llamada y un mensaje alterado podría
        hacer que rechace el historial.
        """
        if not llamadas:
            return []
        modo.fase = fase
        try:
            copia = pedido.model_copy(update={"tool_calls": llamadas})
            salida = await ejecutor.ainvoke({"messages": [*previos, copia]}, config)
        finally:
            modo.fase = Fase.NORMAL
        respuestas: list[AnyMessage] = salida["messages"]
        return respuestas

    async def nodo_herramientas(state: MessagesState, config: RunnableConfig) -> MessagesState:
        """Ejecuta lo pedido aplicando la política de confirmación de RF-08.

        El contexto llega por `get_runtime` y no como parámetro: LangGraph no
        acepta un nodo que reciba `config` y `runtime` a la vez, y el `config`
        hace falta para que las herramientas hereden la invocación.
        """
        pedido = state["messages"][-1]
        if not isinstance(pedido, AIMessage):
            return {"messages": []}  # imposible por la arista condicional
        previos = list(state["messages"][:-1])
        llamadas: list[ToolCall] = list(pedido.tool_calls or [])
        escrituras = [llamada for llamada in llamadas if llamada["name"] in tipos]
        lecturas = [llamada for llamada in llamadas if llamada["name"] not in tipos]
        modo = get_runtime(ContextoDeAgente).context.confirmacion

        if len(escrituras) >= UMBRAL_DE_LOTE:
            return await en_lote(pedido, previos, escrituras, lecturas, modo, config)

        confirma = requiere_confirmacion(
            [tipos[llamada["name"]] for llamada in escrituras],
            hay_texto_de_terceros=_hay_texto_de_terceros(state["messages"]),
        )
        # Una escritura como mucho: si pausa, es la única pausa del paso, y
        # re-ejecutar el paso al reanudar no repite nada.
        fase = Fase.CONFIRMAR if confirma else Fase.DIRECTO
        return {"messages": await correr(pedido, previos, llamadas, modo, fase, config)}

    async def en_lote(
        pedido: AIMessage,
        previos: list[AnyMessage],
        escrituras: list[ToolCall],
        lecturas: list[ToolCall],
        modo: ModoDeConfirmacion,
        config: RunnableConfig,
    ) -> MessagesState:
        """Varias escrituras juntas: una lista, una pausa, cada una una vez.

        Todo lo anterior al `interrupt` se re-ejecuta al reanudar, así que es
        puro: la vista previa sólo valida y arma resúmenes (las herramientas
        no escriben en esa fase). Después del `interrupt` no hay ninguna otra
        pausa posible, y por eso cada escritura aprobada corre una sola vez.
        """
        # 1. Vista previa: cada escritura arma su resumen, sin escribir nada.
        modo.vistas_previas.clear()
        vista = await correr(pedido, previos, escrituras, modo, Fase.VISTA_PREVIA, config)
        resumenes = dict(modo.vistas_previas)
        modo.vistas_previas.clear()
        # Las que no llegaron a pedir permiso (no validaron: "ya pasó", un
        # formato) ya tienen su respuesta definitiva.
        definitivas = [m for m in vista if _id_de(m) not in resumenes]
        ejecutables = [llamada for llamada in escrituras if (llamada["id"] or "") in resumenes]
        if not ejecutables:
            leidas = await correr(pedido, previos, lecturas, modo, Fase.NORMAL, config)
            return {"messages": [*definitivas, *leidas]}

        # 2. Una sola pausa, con la lista completa.
        todas_crean = all(
            tipos[llamada["name"]] is TipoDeEscritura.CREAR for llamada in ejecutables
        )
        decision = interrupt(
            {
                "resumen": _lista_del_lote(
                    [resumenes[llamada["id"] or ""] for llamada in ejecutables], todas_crean
                ),
                "lote": True,
                "ids": [llamada["id"] for llamada in ejecutables],
            }
        )

        # 3. Con el sí: lo que la persona VIO, ni una más. Sin el sí: nada.
        if not (isinstance(decision, dict) and decision.get("aprobado") is True):
            leidas = await correr(pedido, previos, lecturas, modo, Fase.NORMAL, config)
            canceladas = [_respuesta(llamada, LOTE_CANCELADO) for llamada in ejecutables]
            logger.info("agente.lote_cancelado", acciones=len(ejecutables))
            return {"messages": [*definitivas, *leidas, *canceladas]}

        vistas = set(decision.get("ids") or ())
        aprobadas = [llamada for llamada in ejecutables if llamada["id"] in vistas]
        sin_mostrar = [llamada for llamada in ejecutables if llamada["id"] not in vistas]
        hechas = await correr(pedido, previos, [*lecturas, *aprobadas], modo, Fase.APROBADO, config)
        logger.info("agente.lote_aprobado", acciones=len(aprobadas))
        return {
            "messages": [
                *definitivas,
                *hechas,
                *(_respuesta(llamada, NO_ESTABA_EN_LA_LISTA) for llamada in sin_mostrar),
            ]
        }

    # `context_schema` es lo que habilita que las herramientas reciban de
    # quién es la conversación por un canal que el modelo no ve. Ver
    # `contexto.py`: es el control de seguridad de PB-015.
    grafo = StateGraph(MessagesState, context_schema=ContextoDeAgente)
    grafo.add_node(NODO_AGENTE, nodo_agente)
    grafo.add_node(NODO_HERRAMIENTAS, nodo_herramientas)

    grafo.add_edge(START, NODO_AGENTE)
    # `tools_condition` devuelve "tools" o END; el mapa traduce a nuestro nodo,
    # que se llama en español como el resto del código.
    grafo.add_conditional_edges(
        NODO_AGENTE,
        tools_condition,
        {"tools": NODO_HERRAMIENTAS, END: END},
    )
    # Vuelve al agente para que interprete el resultado: quien le contesta a la
    # persona es siempre el modelo, nunca la salida cruda de una herramienta.
    grafo.add_edge(NODO_HERRAMIENTAS, NODO_AGENTE)

    compilado = grafo.compile(checkpointer=checkpointer)
    logger.info("agente.grafo.compilado", herramientas=[h.name for h in herramientas])
    return compilado


def _historial_visible(mensajes: Sequence[BaseMessage]) -> list[BaseMessage]:
    """La ventana del historial que ve el modelo: los últimos mensajes, enteros."""
    return trim_messages(
        mensajes,
        max_tokens=MAX_MENSAJES_DE_HISTORIAL,
        # Contamos mensajes, no tokens: para acotar memoria y costo alcanza,
        # y evita cargar un tokenizador sólo para recortar una lista.
        token_counter=len,
        strategy="last",
        # No es cosmético: garantiza que el recorte no deje un ToolMessage
        # huérfano de su AIMessage. Gemini responde 400 ante ese par roto.
        start_on="human",
        include_system=False,
        allow_partial=False,
    )


def _hay_texto_de_terceros(mensajes: Sequence[BaseMessage]) -> bool:
    """¿El modelo tiene a la vista algo que escribió un tercero (un correo)?

    Se mira la MISMA ventana que recibe el modelo: si un correo leído en un
    turno anterior sigue ahí, todavía puede estar influyendo lo que pide.
    """
    return any(
        isinstance(mensaje, ToolMessage) and mensaje.name in LECTURAS_DE_TERCEROS
        for mensaje in _historial_visible(mensajes)
    )


def _lista_del_lote(resumenes: list[str], todas_crean: bool) -> str:
    """La lista que la persona confirma de una vez."""
    encabezado = "Perfecto, agendo esto:" if todas_crean else "Perfecto, hago esto:"
    return encabezado + "\n" + "\n".join(f"• {resumen}" for resumen in resumenes)


def _id_de(mensaje: BaseMessage) -> str:
    return mensaje.tool_call_id if isinstance(mensaje, ToolMessage) else ""


def _respuesta(llamada: ToolCall, texto: str) -> ToolMessage:
    return ToolMessage(content=texto, tool_call_id=llamada["id"] or "", name=llamada["name"])
