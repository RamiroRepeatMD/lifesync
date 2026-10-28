"""Una página o una base de datos del Notion de la persona (PB-014).

Lo justo para encontrarla y abrirla: título, tipo, cuándo se editó y el link.
El título lo pudo escribir otra persona (un workspace compartido): queda fuera
del `repr` (RF-18), y quien se lo muestra al modelo lo enmarca como dato.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.domain.exceptions import InvalidValueError
from src.domain.value_objects.tipo_de_pagina import TipoDePagina


@dataclass(frozen=True, slots=True)
class Pagina:
    """Una página o una base, tal como aparece en una búsqueda.

    Attributes:
        id: Identificador del proveedor.
        titulo: El título.
        tipo: Página o base de datos.
        editada: Cuándo se editó por última vez, **con zona horaria**.
        enlace: El link para abrirla; None si el proveedor no lo dio.
        en_base: Si es una fila de una base (RF-07): su padre es una base. Una
            fila tiene datos propios y no se mueve fuera de su base.
    """

    id: str
    titulo: str = field(repr=False)
    tipo: TipoDePagina
    editada: datetime
    enlace: str | None = None
    en_base: bool = False

    def __post_init__(self) -> None:
        """Valida las invariantes de la entidad al construirla."""
        if self.editada.tzinfo is None:
            raise InvalidValueError("La fecha de edición debe tener zona horaria.")
