"""Puerto de lectura de las páginas de la persona, en Notion (PB-014).

Capacidad externa, como `Archivos` para Drive: las páginas viven en el
proveedor, y esta interfaz describe sólo lo que el agente necesita de ellas:
encontrarlas, leer una y listar las filas de una base. Sólo lectura: escribir,
mover o borrar quedan para después (PB-044).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID

from src.domain.entities.pagina import Pagina


@dataclass(frozen=True, slots=True)
class PaginaLeida:
    """Una página con su contenido.

    Attributes:
        pagina: La página, con el título y el link reales.
        markdown: El contenido, en Markdown. Lo pudo escribir otra persona.
        recortada: Si el proveedor no la devolvió entera.
    """

    pagina: Pagina
    markdown: str = field(repr=False)
    recortada: bool = False


@dataclass(frozen=True, slots=True)
class FilaDeBase:
    """Una fila de una base, con sus propiedades ya pasadas a texto.

    Attributes:
        titulo: El título de la fila.
        propiedades: (nombre, valor) de las que se pudieron leer, en orden.
        enlace: El link de la fila; None si el proveedor no lo dio.
    """

    titulo: str = field(repr=False)
    propiedades: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    enlace: str | None = None


@dataclass(frozen=True, slots=True)
class BaseConsultada:
    """Las filas de una base, las editadas más recientemente primero.

    Attributes:
        base: La base, con el título y el link reales.
        filas: Las filas que se trajeron.
        hay_mas: Si quedaron filas sin traer.
    """

    base: Pagina
    filas: tuple[FilaDeBase, ...]
    hay_mas: bool = False


class Paginas(ABC):
    """Lo que el agente puede leer del Notion de quien escribe."""

    @abstractmethod
    async def buscar(self, usuario_id: UUID, texto: str, cantidad: int) -> tuple[Pagina, ...]:
        """Busca páginas y bases por el título (sólo el título: la API no busca adentro).

        Args:
            texto: Palabras del título. Vacío: las últimas editadas.
            cantidad: Cuántas traer; el adaptador la acota a un rango sano.

        Raises:
            NotionNoConectadoError: Si la persona no conectó su Notion.
            AutorizacionFallidaError: Si la autorización venció o fue revocada.
            ServiceUnavailableError: Si el proveedor no responde.
        """

    @abstractmethod
    async def leer(self, usuario_id: UUID, pagina_id: str) -> PaginaLeida:
        """Una página con su contenido.

        Raises:
            EntityNotFoundError: Si no existe, no está compartida o el id no
                tiene la forma de uno.
            (y las de `buscar`)
        """

    @abstractmethod
    async def consultar(self, usuario_id: UUID, base_id: str, cantidad: int) -> BaseConsultada:
        """Las filas de una base, las editadas más recientemente primero.

        Raises:
            EntityNotFoundError: Si no existe, no está compartida o el id no
                tiene la forma de uno.
            (y las de `buscar`)
        """
