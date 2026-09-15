"""Traducción compartida de los rechazos HTTP de las APIs de Google (PB-028).

Calendar y Tasks fallan igual y significan lo mismo: 401 es una credencial
que Google ya no acepta, 403 es un token sin el scope necesario (una cuenta
conectada antes de que ese permiso se pidiera), y el resto es el proveedor
sin poder responder. Tener UNA traducción evita que los adaptadores
diverjan en cómo cuentan el mismo problema.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from src.domain.exceptions import (
    AutorizacionFallidaError,
    PermisoInsuficienteError,
    ServiceUnavailableError,
)

logger = structlog.get_logger(__name__)


def traducir_rechazo(respuesta: httpx.Response, servicio: str) -> None:
    """Convierte un status de error en la excepción del dominio que toca.

    Args:
        respuesta: La respuesta HTTP de Google.
        servicio: Nombre corto para los logs y mensajes ("calendar", "tasks").
    """
    if respuesta.status_code < httpx.codes.BAD_REQUEST:
        return

    if respuesta.status_code == httpx.codes.UNAUTHORIZED:
        logger.warning(f"{servicio}.credencial_rechazada")
        raise AutorizacionFallidaError(
            "Google rechazó la credencial. Hay que volver a conectar la cuenta."
        )

    if respuesta.status_code == httpx.codes.FORBIDDEN:
        # El caso concreto: la cuenta se conectó cuando ese scope todavía no
        # se pedía, y el primer uso choca acá. El remedio es de la persona
        # (/conectar de nuevo), así que merece su excepción.
        logger.warning(f"{servicio}.permiso_insuficiente")
        raise PermisoInsuficienteError(f"El token no tiene el permiso que {servicio} necesita.")

    logger.error(
        f"{servicio}.rechazado",
        status_code=respuesta.status_code,
        motivo=motivo_de(respuesta),
    )
    raise ServiceUnavailableError("Google no pudo responder la consulta.")


def motivo_de(respuesta: httpx.Response) -> str | None:
    """El mensaje de error que Google pone en el cuerpo, si hay."""
    error = json_o_vacio(respuesta).get("error")
    if isinstance(error, dict):
        mensaje = error.get("message")
        return str(mensaje) if mensaje is not None else None
    return None


def json_o_vacio(respuesta: httpx.Response) -> dict[str, Any]:
    """El cuerpo como dict, o `{}` si no es JSON o no es un objeto."""
    try:
        cuerpo = respuesta.json()
    except ValueError:
        return {}
    return cuerpo if isinstance(cuerpo, dict) else {}
