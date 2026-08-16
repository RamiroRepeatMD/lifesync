"""Número de WhatsApp en formato E.164 (PB-004).

Es el identificador natural de un `Usuario`: el canal principal del sistema es
WhatsApp, así que el teléfono es lo único que tenemos para saber quién escribe.

Se valida con `re` de la biblioteca estándar y no con `phonenumbers`: la única
fuente de números hoy es el `wa_id` de un webhook de Meta, que ya viene
normalizado y sólo tiene dígitos. Lo que aportaría la librería —interpretar lo
que tipea un humano, con prefijos `0` y `15`— no tiene todavía ningún llamador,
y `src/domain/` no importa dependencias externas por diseño. Cuando aparezca
entrada humana de números, la normalización entra detrás de un puerto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.domain.exceptions import InvalidValueError

# E.164: "+", código de país que no empieza en 0, y entre 8 y 15 dígitos en total.
# Se usa [0-9] y no \d a propósito: en Python \d matchea dígitos Unicode, así que
# "+١٢٣٤٥٦٧٨٩" pasaría la validación y llegaría a la base.
_E164 = re.compile(r"^\+[1-9][0-9]{7,14}$")

LARGO_MAXIMO = 16  # "+" + 15 dígitos


@dataclass(frozen=True, slots=True)
class NumeroWhatsApp:
    """Teléfono en formato E.164 canónico, con el `+` adelante.

    El `+` no es cosmético: si se omite en el campo `to` al enviar, Meta
    antepone el código de país de nuestro número de negocio y el mensaje se
    entrega a otra persona, devolviendo 200. Por eso `valor` siempre lo lleva
    y no existe un accesor "sin +".
    """

    valor: str

    def __post_init__(self) -> None:
        """Valida el formato E.164 al construir."""
        if not _E164.match(self.valor):
            # No se incluye el número en el mensaje: es dato personal y este
            # texto puede terminar en un log o en una respuesta HTTP (RF-18).
            raise InvalidValueError("El número de WhatsApp no tiene formato E.164 válido.")

    @classmethod
    def desde_wa_id(cls, wa_id: str) -> NumeroWhatsApp:
        """Construye el número a partir del `wa_id` que manda Meta.

        Meta entrega sólo dígitos, sin `+`. Se antepone el `+` **verbatim**, sin
        agregar ni quitar dígitos.

        En particular no se fuerza el `9` de los móviles argentinos: esa regla
        se apoya en evidencia de la comunidad que Meta nunca confirmó, y sería
        un no-op si la premisa es cierta y una reescritura de la identidad del
        usuario si es falsa. Si Meta normaliza algo, `ClienteWhatsApp` lo
        detecta comparando el `wa_id` que devuelve al enviar.

        Raises:
            InvalidValueError: Si el `wa_id` no produce un E.164 válido.
        """
        limpio = wa_id.strip()
        if not limpio:
            raise InvalidValueError("El número de WhatsApp no puede estar vacío.")

        return cls(limpio if limpio.startswith("+") else f"+{limpio}")

    def __str__(self) -> str:
        """Devuelve la forma canónica, con `+`."""
        return self.valor
