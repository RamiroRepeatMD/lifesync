"""Puerto de acceso al correo de la persona (PB-033).

Capacidad externa, como `Calendario` y `Tareas`: los correos viven en el
proveedor y esta interfaz describe sólo lo que el agente necesita de ellos.
Sólo lectura en este PB; el envío llega con PB-032.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.entities.correo import Correo


class Correos(ABC):
    """Lo que el agente puede hacer con el correo de quien escribe."""

    @abstractmethod
    async def buscar(self, usuario_id: UUID, consulta: str, cantidad: int) -> tuple[Correo, ...]:
        """Busca correos y los devuelve sin cuerpo, del más nuevo al más viejo.

        Args:
            consulta: Búsqueda en la sintaxis del proveedor; vacía trae los
                últimos de la bandeja de entrada.
            cantidad: Cuántos traer; el adaptador la acota a un rango sano.

        Raises:
            CuentaNoConectadaError: Si la persona nunca autorizó su cuenta.
            PermisoInsuficienteError: Si el token no tiene el scope de correo.
            AutorizacionFallidaError: Si la autorización venció o fue revocada.
            ServiceUnavailableError: Si el proveedor no responde.
        """

    @abstractmethod
    async def leer(self, usuario_id: UUID, correo_id: str) -> Correo:
        """Abre un correo y lo devuelve con cuerpo y adjuntos.

        Raises:
            EntityNotFoundError: Si el correo no existe o ya no está.
            Las mismas de `buscar`.
        """
