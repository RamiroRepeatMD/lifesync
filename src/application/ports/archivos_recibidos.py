"""Puerto para bajar los archivos que la persona mandó por el canal (PB-038).

Capacidad externa, como `MensajeroWhatsApp`: el archivo queda guardado en el
canal (en WhatsApp, unos días del lado de Meta) y se baja recién cuando hay que
subirlo, así no pasa por nuestro lado más tiempo del necesario.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ArchivoDescargado:
    """Los bytes de un archivo recibido, con su tipo.

    Attributes:
        contenido: El archivo. `repr=False`: es contenido de la persona (RF-18).
        tipo_mime: Su tipo, según el canal.
    """

    contenido: bytes = field(repr=False)
    tipo_mime: str


class ArchivosRecibidos(ABC):
    """Lo que hace falta para recuperar un archivo que mandó la persona."""

    @abstractmethod
    async def descargar(self, referencia: str, maximo_bytes: int) -> ArchivoDescargado:
        """Baja un archivo recibido, si no pesa más de `maximo_bytes`.

        Args:
            referencia: La de `ArchivoAdjunto`, tal como la dio el canal.
            maximo_bytes: Tope: un archivo más grande ni se termina de bajar.

        Raises:
            ArchivoNoDisponibleError: Si el canal ya no lo tiene.
            ArchivoDemasiadoGrandeError: Si pesa más que el tope.
            ServiceUnavailableError: Si el canal no responde.
        """
