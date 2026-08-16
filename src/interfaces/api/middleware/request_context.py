"""Middleware de contexto de request: correlación + access log estructurado.

Vincula un `request_id` a los contextvars de structlog, de modo que TODO lo que
se loguee durante ese request (en cualquier capa) lo lleve automáticamente.
Es la base para depurar una conversación de WhatsApp de punta a punta.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"

logger = structlog.get_logger(__name__)

_Siguiente = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Asigna un `request_id`, mide la duración y registra el resultado."""

    async def dispatch(self, request: Request, call_next: _Siguiente) -> Response:
        # Respetamos el id que venga de un proxy/gateway; si no, generamos uno.
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        inicio = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("http.request.error", duracion_ms=_ms_desde(inicio))
            raise

        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "http.request",
            status_code=response.status_code,
            duracion_ms=_ms_desde(inicio),
        )
        return response


def _ms_desde(inicio: float) -> float:
    """Milisegundos transcurridos desde `inicio` (reloj monotónico)."""
    return round((time.perf_counter() - inicio) * 1000, 2)
