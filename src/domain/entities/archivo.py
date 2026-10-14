"""Un archivo del Google Drive de la persona (PB-037).

Sólo lo que hace falta para encontrarlo y abrirlo: nombre, tipo, cuándo se
modificó y el link. El contenido no se lee. El nombre y el dueño pueden venir
de un tercero (un archivo compartido): quedan fuera del `repr` (RF-18), y quien
se los muestra al modelo los enmarca como datos, nunca como instrucciones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.domain.exceptions import InvalidValueError
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo


@dataclass(frozen=True, slots=True)
class Archivo:
    """Un archivo o una carpeta, tal como aparece en una búsqueda.

    Attributes:
        id: Identificador del proveedor.
        nombre: El nombre del archivo.
        tipo: Qué clase de archivo es; None si no es ninguno de los conocidos.
        modificado: Cuándo se modificó por última vez, **con zona horaria**.
        enlace: El link para abrirlo; None si el proveedor no lo dio.
        propietario: Quién lo compartió, si NO es de la persona; None si es
            suyo.
    """

    id: str
    nombre: str = field(repr=False)
    tipo: TipoDeArchivo | None
    modificado: datetime
    enlace: str | None = None
    propietario: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Valida las invariantes de la entidad al construirla."""
        if self.modificado.tzinfo is None:
            raise InvalidValueError("La fecha de modificación debe tener zona horaria.")
