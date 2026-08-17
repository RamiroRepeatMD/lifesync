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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.infrastructure.llm.prompt import INSTRUCCIONES

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


def construir_grafo(
    modelo: BaseChatModel,
    herramientas: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver[Any],
) -> Any:
    """Arma y compila el grafo del agente.

    Args:
        modelo: El chat model ya configurado. No se toca su configuración acá.
        herramientas: Las que el modelo puede invocar. Puede venir vacía: el
            grafo sigue siendo válido y la rama de herramientas nunca se toma.
        checkpointer: Dónde vive el historial de cada conversación (RF-09). El
            llamador elige la implementación; el grafo no se entera de si
            sobrevive a un reinicio.

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
        historial = trim_messages(
            state["messages"],
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
        respuesta = await modelo_con_herramientas.ainvoke(
            [SystemMessage(INSTRUCCIONES), *historial]
        )
        return {"messages": [respuesta]}

    grafo = StateGraph(MessagesState)
    grafo.add_node(NODO_AGENTE, nodo_agente)
    grafo.add_node(NODO_HERRAMIENTAS, ToolNode(herramientas))

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
