"""DTO de una consulta dirigida al agente conversacional (PB-005).

Lleva lo mínimo que el agente necesita para contestar un turno. No transporta
el `MensajeEntrante` completo a propósito: el agente no tiene por qué saber que
la conversación llegó por WhatsApp, ni conocer el `wamid` ni la hora de Meta.
Si mañana entra otro canal, este DTO no cambia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ConsultaDelUsuario:
    """Un turno de conversación que hay que responder.

    Attributes:
        conversacion_id: Hilo al que pertenece el turno. Hoy es el id del
            usuario —una conversación por persona—, pero el nombre es el
            concepto correcto: cuando existan varios hilos por usuario sólo
            cambia quién lo completa, no la firma del puerto.
        texto: Lo que escribió la persona. `repr=False` porque es contenido
            privado y no debe aparecer en un traceback ni en un log (RF-18).
        nombre_usuario: Nombre de pila, si se conoce, para que el agente pueda
            dirigirse a la persona.
    """

    conversacion_id: UUID
    texto: str = field(repr=False)
    nombre_usuario: str | None = None
