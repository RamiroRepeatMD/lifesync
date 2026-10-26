"""Puerto de acceso a los archivos de la persona (PB-037 · PB-038 · PB-039).

Capacidad externa, como `Calendario`, `Tareas` y `Correos`: los archivos viven
en el proveedor, y esta interfaz describe sólo lo que el agente necesita de
ellos: encontrarlos (PB-037), subir y compartir (PB-038), y renombrar, mover
y mandar a la papelera (PB-039). Nunca leer el contenido ni borrar para
siempre.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from uuid import UUID

from src.domain.entities.archivo import Archivo
from src.domain.value_objects.permiso_de_archivo import PermisoDeArchivo
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

    @abstractmethod
    async def comprobar_acceso(self, usuario_id: UUID) -> None:
        """Comprueba que hay una credencial vigente, sin pedirle nada a Drive.

        Para no pedir el sí de algo que no se va a poder hacer (PB-035).

        Raises:
            CuentaNoConectadaError: Si la persona nunca autorizó su cuenta.
            AutorizacionFallidaError: Si la autorización venció o fue revocada.
        """

    @abstractmethod
    async def obtener(self, usuario_id: UUID, archivo_id: str) -> Archivo:
        """Un archivo por su id.

        Raises:
            EntityNotFoundError: Si no existe, o el id no tiene la forma de uno.
            (y las de `buscar`)
        """

    @abstractmethod
    async def carpetas(self, usuario_id: UUID, nombre: str) -> tuple[Archivo, ...]:
        """Las carpetas DE LA PERSONA cuyo nombre contiene `nombre`, las más recientes primero.

        Sólo las propias: una carpeta que le compartió otro tiene su propia forma
        de compartir, y subir ahí le daría el archivo a un tercero.

        Raises:
            (las de `buscar`)
        """

    @abstractmethod
    async def subir(
        self,
        usuario_id: UUID,
        nombre: str,
        tipo_mime: str,
        contenido: bytes,
        carpeta_id: str | None,
    ) -> Archivo:
        """Sube un archivo tal cual (sin convertirlo), a una carpeta o a Mi unidad.

        Raises:
            DriveLlenoError: Si no hay espacio.
            DriveRechazoError: Si Drive no lo deja (por ejemplo, en esa carpeta).
            EntityNotFoundError: Si la carpeta ya no existe.
            (y las de `buscar`)
        """

    @abstractmethod
    async def compartir(
        self,
        usuario_id: UUID,
        archivo_id: str,
        permiso: PermisoDeArchivo,
        direccion: str | None,
    ) -> None:
        """Comparte un archivo con una persona, o con cualquiera que tenga el link.

        Args:
            direccion: El correo de la persona, a quien Google le avisa. None:
                cualquiera con el link.

        Raises:
            DriveRechazoError: Si Drive no lo deja (sin permiso sobre el archivo,
                una cuenta que no comparte afuera, una dirección rechazada).
            EntityNotFoundError: Si el archivo ya no existe.
            (y las de `buscar`)
        """

    @abstractmethod
    async def renombrar(self, usuario_id: UUID, archivo_id: str, nombre: str) -> None:
        """Le cambia el nombre a un archivo o a una carpeta.

        Raises:
            DriveRechazoError: Si Drive no lo deja.
            EntityNotFoundError: Si ya no existe, o el id no tiene la forma de uno.
            (y las de `buscar`)
        """

    @abstractmethod
    async def mover(self, usuario_id: UUID, archivo_id: str, carpeta_id: str | None) -> None:
        """Mueve un archivo a una carpeta, o a Mi unidad con `carpeta_id` None.

        Si ya está ahí, no hace nada.

        Raises:
            DriveRechazoError: Si Drive no lo deja.
            EntityNotFoundError: Si el archivo o la carpeta ya no existen.
            (y las de `buscar`)
        """

    @abstractmethod
    async def a_la_papelera(self, usuario_id: UUID, archivo_id: str) -> None:
        """Manda un archivo a la papelera: se recupera desde Drive durante 30 días.

        Raises:
            DriveRechazoError: Si Drive no lo deja.
            EntityNotFoundError: Si ya no existe.
            (y las de `buscar`)
        """
