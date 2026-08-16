"""DTO de un mensaje de texto recibido por WhatsApp (PB-004).

Transporta lo que el caso de uso necesita, sin arrastrar la forma del payload
de Meta hacia adentro. Deliberadamente NO lleva el `phone_number_id` del
payload: la URL de envío se arma sólo desde la configuración, para que el
contenido de un webhook nunca pueda redirigir una llamada saliente que va con
nuestro token adjunto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp


@dataclass(frozen=True, slots=True)
class MensajeEntrante:
    """Un mensaje de texto que un usuario nos mandó.

    Attributes:
        wamid: Identificador único del mensaje. Es la clave de deduplicación.
        remitente: Quién escribió; se usa para resolver el `Usuario` y responder.
        texto: Cuerpo del mensaje. `repr=False` porque es contenido privado y
            no debe aparecer en un traceback ni en un log (RF-18).
        enviado_en: Momento en que Meta dice que se envió, con zona horaria.
        nombre_perfil: Nombre del perfil de WhatsApp, si vino.
    """

    wamid: str
    remitente: NumeroWhatsApp
    texto: str = field(repr=False)
    enviado_en: datetime
    nombre_perfil: str | None = None
