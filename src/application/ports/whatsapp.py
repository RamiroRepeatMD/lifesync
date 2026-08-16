"""Puerto de salida hacia WhatsApp (PB-004).

A diferencia de `domain/repositories/` —que declara persistencia—, acá va una
capacidad: poder mandarle un mensaje al usuario. La implementación contra la
Cloud API vive en `src/infrastructure/external/whatsapp/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp


class MensajeroWhatsApp(ABC):
    """Capacidad de enviarle un mensaje de texto a un usuario."""

    @abstractmethod
    async def enviar_texto(self, destino: NumeroWhatsApp, texto: str) -> None:
        """Envía un mensaje de texto.

        No devuelve el identificador del mensaje: nadie lo consume todavía y
        obligaría a los dobles de test a inventar uno. El adaptador lo loguea.

        Raises:
            ServiceUnavailableError: Si no se pudo contactar a Meta.
            RepositoryError: Si Meta rechazó el envío.
        """
