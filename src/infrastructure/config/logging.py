"""Logging estructurado con structlog (RNF – Mantenibilidad y Operabilidad).

Objetivos:
- Un evento por línea, con campos consultables (nada de f-strings en los logs).
- Correlación por `request_id`, propagado con contextvars a todo el request.
- Los logs de uvicorn y de la stdlib salen con el mismo formato que los nuestros.
- Consola legible y coloreada en desarrollo; JSON en staging y producción.
- Las librerías que loguean datos sensibles se silencian (RF-18).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from src.infrastructure.config.settings import LogLevel

# Loggers de terceros que reenrutamos al handler raíz para unificar el formato.
_LOGGERS_A_UNIFICAR = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi")

# Loggers que se silencian por debajo de WARNING porque emiten datos sensibles
# (RF-18). httpx registra en INFO la URL completa de cada request, y las
# consultas a PostgREST llevan los filtros en el query string:
#
#   GET .../rest/v1/usuarios?telefono_whatsapp=eq.%2B5491122334455
#
# Eso es un número de WhatsApp —dato personal— en texto plano en cada línea de
# log. A WARNING seguimos viendo los fallos, pero no el tráfico normal.
_LOGGERS_SILENCIADOS = ("httpx", "httpcore", "hpack")
NIVEL_LOGGERS_SILENCIADOS = logging.WARNING


def configure_logging(*, log_level: LogLevel = "INFO", json_logs: bool = False) -> None:
    """Configura structlog + logging de la stdlib. Idempotente.

    Args:
        log_level: Nivel mínimo a emitir.
        json_logs: True para una línea JSON por evento; False para consola.
    """
    procesadores_compartidos: list[Any] = [
        structlog.contextvars.merge_contextvars,  # inyecta request_id y demás
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *procesadores_compartidos,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Se aplica a los logs que NO vienen de structlog (uvicorn, librerías).
        foreign_pre_chain=procesadores_compartidos,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    for nombre in _LOGGERS_A_UNIFICAR:
        logger_externo = logging.getLogger(nombre)
        logger_externo.handlers.clear()
        logger_externo.propagate = True

    # Se aplica después del nivel raíz para que lo pise incluso con LOG_LEVEL=DEBUG:
    # depurar nunca debe habilitar el volcado de datos personales.
    for nombre in _LOGGERS_SILENCIADOS:
        logging.getLogger(nombre).setLevel(NIVEL_LOGGERS_SILENCIADOS)


def get_logger(nombre: str | None = None) -> structlog.stdlib.BoundLogger:
    """Devuelve un logger estructurado. Usar `get_logger(__name__)`."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(nombre)
    return logger
