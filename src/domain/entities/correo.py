"""Un correo de la bandeja de la persona (PB-033).

Casi todo lo que trae un correo lo escribió un tercero: el remitente, el
asunto, el cuerpo. Por eso esos campos quedan fuera del `repr` —no pueden
aparecer en un traceback ni en un assert de pytest (RF-18)— y por eso quien
se los muestra al modelo los enmarca como datos, nunca como instrucciones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True, slots=True)
class Correo:
    """Un mensaje de la bandeja, en listado o abierto.

    Attributes:
        id: Identificador del proveedor; es lo que permite abrirlo después.
        remitente: Quién lo manda, como "Nombre <direccion>".
        asunto: El asunto, ya decodificado.
        fecha: Cuándo llegó, **siempre con zona horaria**.
        no_leido: Si la persona todavía no lo abrió.
        fragmento: La vista previa corta que arma el proveedor.
        cuerpo: El texto completo; sólo al abrirlo (None en un listado).
        adjuntos: Nombres de los archivos adjuntos. No se descargan.
    """

    id: str
    remitente: str = field(repr=False)
    asunto: str = field(repr=False)
    fecha: datetime
    no_leido: bool = False
    fragmento: str = field(default="", repr=False)
    cuerpo: str | None = field(default=None, repr=False)
    adjuntos: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        """Valida las invariantes de la entidad al construirla."""
        if self.fecha.tzinfo is None:
            raise InvalidValueError("La fecha del correo debe tener zona horaria.")
