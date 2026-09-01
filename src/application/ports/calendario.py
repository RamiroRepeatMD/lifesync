"""Puerto de lectura del calendario (PB-015, RF-03).

Declara una **capacidad** —poder mirar la agenda de alguien—, así que vive acá
junto a `MensajeroWhatsApp`, `AgenteConversacional` y `AutorizadorGoogle`.

Sólo lectura: el scope concedido en PB-009 es `calendar.readonly` y crear o
modificar eventos es PB-016, que además necesita antes la confirmación
explícita de RF-08.

Recibe `usuario_id` en cada llamada, y no en el constructor, porque la
implementación se arma una sola vez al arrancar el proceso y la comparten todas
las conversaciones. **De quién es la agenda es un dato por invocación**, y eso
importa: es lo que impide que el agente lea la de otro.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from src.domain.entities.evento import Evento


class Calendario(ABC):
    """Capacidad de leer los eventos del calendario de una persona."""

    @abstractmethod
    async def eventos_entre(
        self, usuario_id: UUID, desde: datetime, hasta: datetime
    ) -> tuple[Evento, ...]:
        """Devuelve los eventos de la persona en ese rango, ordenados por inicio.

        Args:
            usuario_id: De quién es la agenda.
            desde: Inicio del rango, con zona horaria.
            hasta: Fin del rango, con zona horaria.

        Returns:
            Los eventos ordenados por comienzo. Vacío si no hay ninguno, que no
            es un error: un día libre es una respuesta válida.

        Raises:
            CuentaNoConectadaError: Si la persona nunca autorizó su cuenta.
            AutorizacionFallidaError: Si la autorización venció y no se pudo
                renovar, o el proveedor la rechazó.
            ServiceUnavailableError: Si el proveedor no responde.
        """
