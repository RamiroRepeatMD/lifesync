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
        # Dos 403 con remedios distintos, y confundirlos manda a la persona a
        # /conectar en vano (pasó el 15/09 con Tasks):
        #
        # - SERVICE_DISABLED: la API no está habilitada en el proyecto de
        #   Cloud Console. Lo arregla EL OPERADOR en la consola; reconectar
        #   no cambia nada. El log fuerte trae el mensaje de Google, que
        #   incluye el link exacto para habilitarla.
        # - El resto: al token le falta el scope (cuenta conectada antes de
        #   que se pidiera). Eso sí lo arregla la persona con /conectar.
        motivo = motivo_de(respuesta)
        if _es_api_deshabilitada(respuesta):
            logger.error(f"{servicio}.api_deshabilitada", motivo=motivo)
            raise ServiceUnavailableError(
                f"La API de {servicio} no está habilitada en el proyecto de Google Cloud."
            )
        logger.warning(f"{servicio}.permiso_insuficiente", motivo=motivo)
        raise PermisoInsuficienteError(f"El token no tiene el permiso que {servicio} necesita.")

    logger.error(
        f"{servicio}.rechazado",
        status_code=respuesta.status_code,
        motivo=motivo_de(respuesta),
    )
    raise ServiceUnavailableError("Google no pudo responder la consulta.")


def _es_api_deshabilitada(respuesta: httpx.Response) -> bool:
    """Detecta el 403 de "API not enabled" por su marca estable.

    Google lo señala con `reason: SERVICE_DISABLED` en los detalles del
    error (el texto del mensaje puede cambiar; la razón es contrato).
    """
    error = json_o_vacio(respuesta).get("error")
    if not isinstance(error, dict):
        return False
    for detalle in error.get("details") or []:
        if isinstance(detalle, dict) and detalle.get("reason") == "SERVICE_DISABLED":
            return True
    return False


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
