"""Puerto de salida hacia el agente conversacional (PB-005).

Igual que `MensajeroWhatsApp`, declara una **capacidad**, no persistencia: poder
convertir lo que escribió una persona en una respuesta. Por eso vive acá y no
en `domain/repositories/`.

La implementación real —LangGraph sobre Gemini— vive en
`src/infrastructure/llm/`. El dominio no sabe que existe un LLM, y esta capa
tampoco: sólo sabe que alguien puede contestar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario


class AgenteConversacional(ABC):
    """Capacidad de responder un turno de conversación en lenguaje natural."""

    @abstractmethod
    async def responder(self, consulta: ConsultaDelUsuario) -> str:
        """Devuelve el texto con el que hay que contestarle a la persona.

        El contexto previo de la conversación (RF-09) es responsabilidad de la
        implementación: el llamador manda un turno y recibe una respuesta, sin
        enterarse de dónde ni cómo se recuerda lo anterior.

        Devuelve siempre un texto utilizable: si el agente no tiene nada que
        decir, la implementación se encarga de producir algo presentable en vez
        de una cadena vacía.

        Raises:
            AgenteNoDisponibleError: Si no se pudo obtener una respuesta.
        """
