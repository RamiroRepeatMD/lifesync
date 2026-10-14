"""Un archivo que la persona mandó junto con su mensaje (PB-038).

No conoce al canal: `referencia` es lo que el canal necesita para devolver el
archivo (en WhatsApp, el id de media) y sólo lo interpreta quien lo descarga.
Los bytes no viajan acá: se bajan recién cuando hay que subirlos, así no queda
guardado nada más que lo necesario (ERS 3.10.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ArchivoAdjunto:
    """Lo que hace falta para nombrar el archivo y pedírselo al canal.

    Attributes:
        referencia: Con qué se le pide el archivo al canal.
        tipo_mime: Su tipo, tal como lo informó el canal.
        nombre: Con qué nombre llegó (o uno armado, si el canal no trajo).
            `repr=False`: es contenido de la persona (RF-18).
    """

    referencia: str
    tipo_mime: str
    nombre: str = field(repr=False)
