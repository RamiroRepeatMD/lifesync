"""Entidad Usuario: la persona que conversa con LifeSync.

El identificador natural es el número de WhatsApp, porque ése es el canal
principal del sistema. La entidad se mantiene deliberadamente mínima: las
preferencias (zona horaria, formato de fecha, notificaciones) son RF-13, de
prioridad Media y fuera del Sprint 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True, slots=True)
class Usuario:
    """Usuario registrado del sistema.

    Es inmutable: para "modificar" un usuario se construye uno nuevo con
    `dataclasses.replace`. Así ninguna capa puede alterar una entidad por
    accidente después de leerla del repositorio.

    Attributes:
        telefono_whatsapp: Identificador natural del usuario.
        nombre: Nombre para dirigirse a la persona; opcional.
        id: Asignado por la base al persistir; None mientras no exista.
        creado_en: Alta en el sistema; lo completa la base.
        actualizado_en: Última modificación; lo completa la base.
    """

    telefono_whatsapp: str
    nombre: str | None = None
    id: UUID | None = None
    creado_en: datetime | None = None
    actualizado_en: datetime | None = None

    def __post_init__(self) -> None:
        """Valida las invariantes de la entidad al construirla."""
        if not self.telefono_whatsapp.strip():
            raise InvalidValueError("El teléfono de WhatsApp no puede estar vacío.")
