"""Traducción de errores de dominio a respuestas HTTP amigables (RF-19).

Reglas:
- Los errores de negocio conocidos devuelven 4xx con un mensaje entendible.
- Cualquier excepción no prevista devuelve 500 con un mensaje genérico: nunca
  se filtra un stack trace al usuario, pero sí se loguea completo del lado
  del servidor.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.domain.exceptions import (
    ConfirmationRequiredError,
    DomainError,
    EncryptionError,
    EntityNotFoundError,
    InfrastructureError,
    InvalidValueError,
    LifeSyncError,
    RepositoryError,
    ServiceUnavailableError,
)

logger = structlog.get_logger(__name__)

MENSAJE_GENERICO = "Ocurrió un error inesperado. Probá de nuevo en unos minutos."

# Cada error del dominio sabe qué significa; acá sólo decidimos el código HTTP.
_STATUS_POR_ERROR: dict[type[LifeSyncError], int] = {
    EntityNotFoundError: status.HTTP_404_NOT_FOUND,
    InvalidValueError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ConfirmationRequiredError: status.HTTP_409_CONFLICT,
    DomainError: status.HTTP_400_BAD_REQUEST,
    # Infraestructura (PB-003). Registrarlas no es opcional: son hermanas de
    # DomainError, así que sin entrada propia el recorrido del MRO las dejaría
    # en 400 y una caída de la base se reportaría como "pediste algo mal".
    ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    RepositoryError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    EncryptionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    InfrastructureError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def _cuerpo(codigo: str, mensaje: str, detalles: Any | None = None) -> dict[str, Any]:
    """Construye el cuerpo de error uniforme de la API."""
    error: dict[str, Any] = {"code": codigo, "message": mensaje}
    if detalles is not None:
        error["details"] = detalles
    return {"error": error}


def _status_para(exc: LifeSyncError) -> int:
    """Busca el status del error o del primer ancestro registrado."""
    for clase in type(exc).__mro__:
        if clase in _STATUS_POR_ERROR:
            return _STATUS_POR_ERROR[clase]
    return status.HTTP_400_BAD_REQUEST


async def _manejar_error_de_dominio(request: Request, exc: Exception) -> JSONResponse:
    """Error conocido: 4xx o 5xx con el mensaje pensado para el usuario."""
    error = exc if isinstance(exc, LifeSyncError) else LifeSyncError(str(exc))
    codigo_http = _status_para(error)

    # Una caída de la base no es un error de negocio: tiene que ser alertable.
    if codigo_http >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        logger.error(
            "infrastructure.error",
            codigo=error.codigo,
            detalle=error.detalle,
            status_code=codigo_http,
            exc_info=exc,
        )
    else:
        logger.warning("domain.error", codigo=error.codigo, detalle=error.detalle)

    return JSONResponse(
        status_code=codigo_http,
        content=_cuerpo(error.codigo, error.mensaje_usuario),
    )


async def _manejar_error_de_validacion(request: Request, exc: Exception) -> JSONResponse:
    """Payload inválido: devolvemos qué campos fallaron, sin exponer internals."""
    detalles = exc.errors() if isinstance(exc, RequestValidationError) else None
    logger.warning("request.validation_error", detalles=detalles)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_cuerpo(
            "ValidationError",
            "Los datos enviados no tienen el formato esperado.",
            detalles,
        ),
    )


async def _manejar_error_inesperado(request: Request, exc: Exception) -> JSONResponse:
    """Última red de contención: se loguea todo, se responde poco."""
    logger.exception("unhandled.error", tipo=type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_cuerpo("InternalServerError", MENSAJE_GENERICO),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Registra los manejadores de error en la aplicación."""
    app.add_exception_handler(LifeSyncError, _manejar_error_de_dominio)
    app.add_exception_handler(RequestValidationError, _manejar_error_de_validacion)
    app.add_exception_handler(Exception, _manejar_error_inesperado)
