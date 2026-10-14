"""Puerto de acceso a los archivos de la persona (PB-037).

Capacidad externa, como `Calendario`, `Tareas` y `Correos`: los archivos viven
en el proveedor, y esta interfaz describe sólo lo que el agente necesita de
ellos. Por ahora, encontrarlos: buscar y listar. Subir y compartir llegan con
PB-038.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from src.domain.entities.archivo import Archivo
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo


class Archivos(ABC):
    """Lo que el agente puede hacer con los archivos de quien escribe."""

    @abstractmethod
    async def buscar(
        self,
        usuario_id: UUID,
        texto: str,
        tipo: TipoDeArchivo | None,
        desde: datetime | None,
        cantidad: int,
    ) -> tuple[Archivo, ...]:
        """Busca archivos (sin los de la papelera) y devuelve los primeros.

        Args:
            texto: Lo que se busca, en el nombre o en el contenido. Vacío
                trae los últimos modificados: es "listar".
            tipo: Sólo de ese tipo; None, de cualquiera.
            desde: Sólo los modificados desde ese momento, **con zona
                horaria**; None, sin límite.
            cantidad: Cuántos traer; el adaptador la acota a un rango sano.

        Raises:
            CuentaNoConectadaError: Si la persona nunca autorizó su cuenta.
            PermisoInsuficienteError: Si el token no tiene el permiso de Drive.
            AutorizacionFallidaError: Si la autorización venció o fue revocada.
            ServiceUnavailableError: Si el proveedor no responde.
        """
