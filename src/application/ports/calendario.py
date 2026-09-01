"""Puerto de lectura y escritura del calendario (PB-015 · PB-016, RF-03).

Declara una **capacidad** —poder mirar la agenda de alguien—, así que vive acá
junto a `MensajeroWhatsApp`, `AgenteConversacional` y `AutorizadorGoogle`.

La escritura existe desde PB-016, y su frontera de seguridad **no está acá**:
está en el grafo del agente, donde un `interrupt()` pausa la ejecución antes de
cualquier llamada que escriba y espera la confirmación explícita de la persona
(RF-08). El puerto es deliberadamente neutro; la garantía es estructural y vive
en `infrastructure/llm/herramientas.py`.

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

    @abstractmethod
    async def eventos_del_principal(
        self, usuario_id: UUID, desde: datetime, hasta: datetime
    ) -> tuple[Evento, ...]:
        """Como `eventos_entre`, pero sólo del calendario principal.

        Existe porque la escritura opera únicamente sobre el principal (ver
        `crear_evento`), así que la búsqueda previa a un borrado tiene que
        mirar el mismo lugar donde se va a borrar.
        """

    @abstractmethod
    async def crear_evento(self, usuario_id: UUID, evento: Evento) -> Evento:
        """Crea el evento en el calendario **principal** de la persona.

        Siempre el principal, por decisión: se lee de todos los calendarios,
        se escribe en el propio. Escribir en calendarios compartidos o
        secundarios es una decisión que la persona debería tomar en su UI.

        Returns:
            El evento persistido, con el `id` que asignó el proveedor.

        Raises:
            CuentaNoConectadaError: Si la persona nunca autorizó su cuenta.
            PermisoInsuficienteError: Si el token no tiene permiso de
                escritura (la cuenta se conectó antes de PB-016 y hay que
                volver a autorizar).
            AutorizacionFallidaError: Si la autorización venció o fue revocada.
            ServiceUnavailableError: Si el proveedor no responde.
        """

    @abstractmethod
    async def eliminar_evento(self, usuario_id: UUID, evento_id: str) -> None:
        """Elimina un evento del calendario principal.

        Es idempotente: eliminar algo que ya no existe no es un error — el
        estado final es el mismo.

        Raises:
            CuentaNoConectadaError: Si la persona nunca autorizó su cuenta.
            PermisoInsuficienteError: Si el token no tiene permiso de escritura.
            AutorizacionFallidaError: Si la autorización venció o fue revocada.
            ServiceUnavailableError: Si el proveedor no responde.
        """
