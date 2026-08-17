"""Agente de reemplazo para cuando no hay LLM configurado (PB-005).

Es un *null object* del puerto `AgenteConversacional`: se comporta como un
agente, pero contesta siempre lo mismo. Existe para que el composition root
pueda entregar siempre un agente y el caso de uso no tenga que preguntarse si
hay uno; sin esto, `ProcesarMensajeEntrante` cargaría un `if` de configuración
que no es asunto suyo.

Vive en `application/` y no en `infrastructure/llm/` **a propósito**: éste es
justamente el camino que tiene que seguir funcionando cuando el stack de
LangChain no está disponible, así que no puede importar nada de él.
"""

from __future__ import annotations

import structlog

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.ports.agente import AgenteConversacional

logger = structlog.get_logger(__name__)

SIN_AGENTE = (
    "Ahora mismo no puedo interpretar mensajes libres: me falta configuración "
    "para pensar la respuesta.\n\n"
    "Escribí /ayuda para ver lo que sí puedo hacer."
)


class AgenteDegradado(AgenteConversacional):
    """Contesta un texto fijo y honesto en lugar de fallar."""

    async def responder(self, consulta: ConsultaDelUsuario) -> str:
        """Devuelve el aviso de que el agente no está configurado.

        No lanza: dejar el mensaje sin respuesta sería peor que decirle a la
        persona que hoy no se puede (RF-19).
        """
        logger.warning("agente.degradado.respuesta", conversacion_id=str(consulta.conversacion_id))
        return SIN_AGENTE
