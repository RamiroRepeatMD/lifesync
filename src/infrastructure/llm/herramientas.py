"""Herramientas que el agente puede invocar (PB-005).

Por ahora hay una sola, y es deliberada: sin ninguna herramienta el ciclo de
tool-calling quedaría cableado pero nunca ejecutado, o sea sin evidencia de que
funciona. Con una alcanza para probarlo de punta a punta.

`fecha_y_hora_actual` es además la que hacía falta primero: un modelo no sabe
qué día es hoy, así que sin esto cualquier "mañana" o "el viernes" se resuelve
inventando. Cuando entre Calendar en el Sprint 3, esa resolución ya va a estar.

Es de **sólo lectura**: no modifica nada, así que no dispara la confirmación de
RF-08. La primera herramienta que escriba va a necesitar un `interrupt` en el
grafo, no sólo la instrucción del prompt.

El docstring de cada función **es la documentación que ve el modelo**: es lo que
lee para decidir cuándo llamarla. Se escribe para él, no para nosotros.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import structlog
from langchain_core.tools import BaseTool, tool

logger = structlog.get_logger(__name__)

# Zona fija mientras las preferencias por usuario sean RF-13 (Sprint 2). Cuando
# existan, esto pasa a salir del `Usuario` y la herramienta la recibe por
# parámetro.
ZONA_HORARIA = ZoneInfo("America/Argentina/Buenos_Aires")

_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MESES = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


@tool
def fecha_y_hora_actual() -> str:
    """Devuelve la fecha y la hora de ahora en la zona horaria de Argentina.

    Usala siempre que necesites saber qué día es hoy o qué hora es, y también
    para resolver fechas relativas como "mañana", "el viernes" o "en dos
    semanas". Nunca supongas la fecha: consultala.
    """
    ahora = datetime.now(UTC).astimezone(ZONA_HORARIA)
    # Nombres armados a mano en vez de `strftime("%A")`: el nombre del día que
    # devuelve strftime depende del locale del sistema operativo, y en el
    # contenedor de Railway ese locale es C, o sea inglés.
    dia = _DIAS[ahora.weekday()]
    mes = _MESES[ahora.month - 1]
    respuesta = f"{dia} {ahora.day} de {mes} de {ahora.year}, {ahora:%H:%M} (hora de Argentina)"

    logger.info("agente.herramienta.invocada", herramienta="fecha_y_hora_actual")
    return respuesta


# Lo que se le pasa al grafo. Sumar una herramienta es agregarla acá.
HERRAMIENTAS: tuple[BaseTool, ...] = (fecha_y_hora_actual,)
