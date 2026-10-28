"""Puerto de las páginas de la persona, en Notion (PB-014 · RF-07).

Capacidad externa, como `Archivos` para Drive: las páginas viven en el
proveedor, y esta interfaz describe sólo lo que el agente necesita de ellas:
encontrarlas, leer una y listar las filas de una base (PB-014); y crear,
agregar al final, cambiar un dato, mover, mandar a la papelera y subir un
archivo (RF-07). Nunca editar ni borrar texto que ya existe, ni borrar para
siempre.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
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
        id: El id de la fila: sin él, el modelo no tiene cómo cambiarle un dato
            (todos los meses tienen su fila «Luz»).
    """

    titulo: str = field(repr=False)
    propiedades: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    enlace: str | None = None
    id: str = ""


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


class TipoDeBloque(StrEnum):
    """Lo que se escribe en una página: un subconjunto chico de lo que admite Notion."""

    PARRAFO = "párrafo"
    TITULO_1 = "título 1"
    TITULO_2 = "título 2"
    TITULO_3 = "título 3"
    VINETA = "viñeta"
    NUMERADO = "numerado"
    TAREA = "tarea"


@dataclass(frozen=True, slots=True)
class Bloque:
    """Una línea de lo que se va a escribir.

    Attributes:
        tipo: Qué clase de línea es.
        texto: Lo que dice.
        hecho: Sólo para una tarea: si ya está marcada.
    """

    tipo: TipoDeBloque
    texto: str = field(repr=False)
    hecho: bool = False


class TipoDePropiedad(StrEnum):
    """Los datos de una fila que se pueden cambiar."""

    TITULO = "título"
    TEXTO = "texto"
    NUMERO = "número"
    CASILLA = "casilla"
    FECHA = "fecha"
    SELECCION = "selección"
    ESTADO = "estado"


@dataclass(frozen=True, slots=True)
class Propiedad:
    """Un dato de una página o de una fila, como está ahora.

    Attributes:
        id: El id de la propiedad: se escribe por el id, que no cambia si
            alguien la renombra.
        nombre: Cómo se llama en la base.
        tipo: Qué clase de dato es.
        valor: El valor actual, en texto; vacío si no tiene.
        opciones: Para selección y estado, las opciones del esquema: sólo se
            puede elegir una de ésas.
    """

    id: str
    nombre: str = field(repr=False)
    tipo: TipoDePropiedad
    valor: str = field(default="", repr=False)
    opciones: tuple[str, ...] = field(default=(), repr=False)


# Lo que se escribe en cada tipo de dato: texto (título, texto, selección y
# estado), número, casilla o fecha.
ValorDePropiedad = str | float | bool | date


class Paginas(ABC):
    """Lo que el agente puede hacer en el Notion de quien escribe."""

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

    @abstractmethod
    async def obtener(self, usuario_id: UUID, pagina_id: str) -> Pagina:
        """Una página o una base por su id, tal como está ahora (RF-07).

        Raises:
            EntityNotFoundError: Si no existe, no está compartida, está en la
                papelera o el id no tiene la forma de uno.
            (y las de `buscar`)
        """

    @abstractmethod
    async def propiedades(
        self, usuario_id: UUID, pagina_id: str
    ) -> tuple[Pagina, tuple[Propiedad, ...]]:
        """Una página con los datos que se le pueden cambiar.

        En una fila, sus propiedades con las opciones del esquema de la base; en
        una página suelta, sólo el título.

        Raises:
            (las de `obtener`)
        """

    @abstractmethod
    async def crear(
        self, usuario_id: UUID, padre: Pagina, titulo: str, bloques: tuple[Bloque, ...]
    ) -> Pagina:
        """Crea una página dentro de otra, o una fila nueva si el padre es una base.

        Raises:
            NotionRechazoError: Si Notion no lo acepta (y su hija, si no deja).
            EscrituraEnDudaError: Si Notion no confirmó: pudo haberse creado.
            EntityNotFoundError: Si el padre ya no está.
            (y las de `buscar`)
        """

    @abstractmethod
    async def agregar(self, usuario_id: UUID, pagina_id: str, bloques: tuple[Bloque, ...]) -> None:
        """Agrega líneas al final de una página, sin tocar lo que ya tiene.

        Raises:
            (las de `crear`)
        """

    @abstractmethod
    async def cambiar(
        self, usuario_id: UUID, pagina_id: str, propiedad: Propiedad, valor: ValorDePropiedad
    ) -> None:
        """Cambia un dato: el `valor` ya viene validado para el tipo de la propiedad.

        Raises:
            (las de `crear`)
        """

    @abstractmethod
    async def mover(self, usuario_id: UUID, pagina_id: str, destino_id: str) -> None:
        """Mueve una página adentro de otra página.

        Raises:
            (las de `crear`)
        """

    @abstractmethod
    async def a_la_papelera(self, usuario_id: UUID, pagina_id: str) -> None:
        """Manda una página o una fila a la papelera de Notion, de donde se recupera.

        Raises:
            (las de `crear`)
        """

    @abstractmethod
    async def limite_de_subida(self, usuario_id: UUID) -> int | None:
        """Cuántos bytes acepta el espacio por archivo; None si Notion no lo dice.

        Raises:
            (las de `buscar`)
        """

    @abstractmethod
    async def subir(
        self, usuario_id: UUID, pagina_id: str, nombre: str, tipo_mime: str, contenido: bytes
    ) -> None:
        """Sube un archivo y lo agrega al final de una página.

        Raises:
            (las de `crear`)
        """
