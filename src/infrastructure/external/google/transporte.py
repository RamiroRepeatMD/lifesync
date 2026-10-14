"""Traducción compartida de los rechazos HTTP de las APIs de Google (PB-028).

Calendar, Tasks y Gmail fallan igual y significan lo mismo: 401 es una
credencial que Google ya no acepta, y el resto es el proveedor sin poder
responder. El 403 tiene TRES causas con remedios distintos (ver
`traducir_rechazo`). Tener UNA traducción evita que los adaptadores diverjan
en cómo cuentan el mismo problema.
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

# Cómo marca Google cada 403. El formato clásico pone el motivo en
# `errors[].reason` y el nuevo en `details[].reason` (ErrorInfo): se miran los
# dos. Los motivos son contrato; el texto del mensaje puede cambiar.
API_DESHABILITADA = frozenset({"SERVICE_DISABLED", "accessNotConfigured"})
LIMITE_DE_USO = frozenset(
    {
        "rateLimitExceeded",
        "userRateLimitExceeded",
        "dailyLimitExceeded",
        "quotaExceeded",
        "RATE_LIMIT_EXCEEDED",
        # El de compartir archivos de Drive (PB-038): muchos avisos por correo.
        "sharingRateLimitExceeded",
    }
)


def traducir_rechazo(
    respuesta: httpx.Response, servicio: str, *, registrar_mensaje: bool = True
) -> None:
    """Convierte un status de error en la excepción del dominio que toca.

    Args:
        respuesta: La respuesta HTTP de Google.
        servicio: Nombre corto para los logs y mensajes ("calendar", "tasks").
        registrar_mensaje: Si el log lleva el texto de Google. Con `False`, sólo
            los códigos: al compartir, ese texto puede repetir la dirección
            invitada (PB-038, RF-18).
    """
    if respuesta.status_code < httpx.codes.BAD_REQUEST:
        return
    motivo = motivo_de(respuesta) if registrar_mensaje else None

    if respuesta.status_code == httpx.codes.UNAUTHORIZED:
        logger.warning(f"{servicio}.credencial_rechazada")
        raise AutorizacionFallidaError(
            "Google rechazó la credencial. Hay que volver a conectar la cuenta."
        )

    if respuesta.status_code == httpx.codes.TOO_MANY_REQUESTS:
        # El otro formato del límite de uso (PB-035): Google también lo marca
        # con 429, y Gmail lo usa para los límites de envío. Mismo remedio que
        # el 403 por límite: esperar, no reconectar.
        logger.warning(f"{servicio}.limite_de_uso", motivo=motivo)
        raise ServiceUnavailableError(f"Google limitó los pedidos a {servicio} por un rato.")

    if respuesta.status_code == httpx.codes.FORBIDDEN:
        # Tres 403 con remedios distintos, y confundirlos manda a la persona a
        # /conectar en vano (pasó el 15/09 con Tasks):
        #
        # - SERVICE_DISABLED: la API no está habilitada en el proyecto de
        #   Cloud Console. Lo arregla EL OPERADOR en la consola; reconectar
        #   no cambia nada. El log fuerte trae el mensaje de Google, que
        #   incluye el link exacto para habilitarla.
        # - Límite de uso (PB-031): demasiados pedidos en poco tiempo. Se
        #   arregla esperando, no reconectando: es un "probá en un minuto".
        # - El resto: al token le falta el scope (cuenta conectada antes de
        #   que se pidiera). Eso sí lo arregla la persona con /conectar.
        motivos = motivos_de(respuesta)
        if motivos & API_DESHABILITADA:
            logger.error(f"{servicio}.api_deshabilitada", motivo=motivo)
            raise ServiceUnavailableError(
                f"La API de {servicio} no está habilitada en el proyecto de Google Cloud."
            )
        if motivos & LIMITE_DE_USO:
            logger.warning(f"{servicio}.limite_de_uso", motivo=motivo)
            raise ServiceUnavailableError(f"Google limitó los pedidos a {servicio} por un rato.")
        logger.warning(f"{servicio}.permiso_insuficiente", motivo=motivo)
        raise PermisoInsuficienteError(f"El token no tiene el permiso que {servicio} necesita.")

    logger.error(
        f"{servicio}.rechazado",
        status_code=respuesta.status_code,
        motivo=motivo,
    )
    raise ServiceUnavailableError("Google no pudo responder la consulta.")


def motivos_de(respuesta: httpx.Response) -> set[str]:
    """Los `reason` del cuerpo de error, en el formato clásico y en el nuevo."""
    error = json_o_vacio(respuesta).get("error")
    if not isinstance(error, dict):
        return set()
    motivos: set[str] = set()
    for lista in (error.get("errors"), error.get("details")):
        for item in lista if isinstance(lista, list) else []:
            if isinstance(item, dict) and isinstance(item.get("reason"), str):
                motivos.add(item["reason"])
    return motivos


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
