"""Logging estructurado con structlog (RNF – Mantenibilidad y Operabilidad).

Objetivos:
- Un evento por línea, con campos consultables (nada de f-strings en los logs).
- Correlación por `request_id`, propagado con contextvars a todo el request.
- Los logs de uvicorn y de la stdlib salen con el mismo formato que los nuestros.
- Consola legible y coloreada en desarrollo; JSON en staging y producción.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from src.infrastructure.config.settings import LogLevel

# Loggers de terceros que reenrutamos al handler raíz para unificar el formato.
_LOGGERS_A_UNIFICAR = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi")


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


def get_logger(nombre: str | None = None) -> structlog.stdlib.BoundLogger:
    """Devuelve un logger estructurado. Usar `get_logger(__name__)`."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(nombre)
    return logger
