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
from uuid import UUID

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

    @abstractmethod
    async def olvidar(self, conversacion_id: UUID) -> bool:
        """Borra la memoria de una conversación (el comando /olvidar).

        Se va también lo que estaba esperando una confirmación. No toca nada
        fuera de la conversación: cuentas, recordatorios ni datos de terceros.
        Devuelve True si había algo guardado.

        Raises:
            ServiceUnavailableError: Si no se pudo borrar.
        """

    @abstractmethod
    async def cancelar_pendiente(self, conversacion_id: UUID) -> bool:
        """Cancela, como un «no», lo que esté esperando una confirmación (RF-08).

        Lo usa /olvidar antes de explicar: si no, un «sí» a secas aprobaría lo
        pendiente, justo lo que la persona quiere descartar. Devuelve True si
        había algo esperando.
        """
