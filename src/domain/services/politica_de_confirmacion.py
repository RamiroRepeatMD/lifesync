"""Política de confirmación de RF-08: qué necesita el "sí" de la persona.

Hasta el Sprint 4 la regla era única —toda escritura confirma— y vivía en la
forma de las herramientas. Usándolo de verdad, confirmar cada evento o tarea
resultó fricción pura: agendar un turno no debería frenar. Desde que hay
acciones con criticidad distinta, la regla es de negocio y vive acá, en el
dominio, sin saber nada de LangGraph ni de WhatsApp:

- **Lo irreversible confirma**: borrar y mandar algo a un tercero no tienen
  vuelta atrás.
- **Varias acciones juntas confirman**, todas en una sola pregunta: es "muchas
  cosas", y la persona merece ver la lista completa antes.
- **Con texto de terceros a la vista, todo confirma.** Un correo lo escribe
  cualquiera; si el modelo lo está leyendo, una escritura podría venir de ahí
  y no de la persona. Es la barrera contra la inyección por correo.
- Una sola acción reversible —crear, modificar, completar, posponer— sale
  directo, y la respuesta cuenta qué y cuándo: ése es el control.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

# Desde cuántas escrituras en un mismo pedido se confirma la lista completa.
UMBRAL_DE_LOTE = 2


class TipoDeEscritura(Enum):
    """Qué le hace una escritura a los datos de la persona."""

    CREAR = "crear"
    MODIFICAR = "modificar"
    COMPLETAR = "completar"
    POSPONER = "posponer"
    ELIMINAR = "eliminar"
    ENVIAR = "enviar"


# Lo que no tiene vuelta atrás: lo borrado no vuelve, y un correo no se des-envía.
IRREVERSIBLES = frozenset({TipoDeEscritura.ELIMINAR, TipoDeEscritura.ENVIAR})


def requiere_confirmacion(
    tipos: Sequence[TipoDeEscritura], *, hay_texto_de_terceros: bool = False
) -> bool:
    """¿Las escrituras de este paso necesitan el sí de la persona?

    Args:
        tipos: Una entrada por cada escritura que se va a ejecutar junta.
        hay_texto_de_terceros: Si el modelo tiene a la vista contenido que
            escribió otra persona (un correo leído).
    """
    if not tipos:
        return False
    if len(tipos) >= UMBRAL_DE_LOTE or hay_texto_de_terceros:
        return True
    return any(tipo in IRREVERSIBLES for tipo in tipos)
