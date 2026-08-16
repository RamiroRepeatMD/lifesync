"""Traducción de fallas de Supabase a las excepciones del proyecto (RF-19).

Los repositorios no deben dejar escapar excepciones de `postgrest` ni de
`httpx`: las capas de arriba sólo conocen la jerarquía de `domain.exceptions`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import structlog
from postgrest import APIError

from src.domain.exceptions import (
    EntityNotFoundError,
    InvalidValueError,
    LifeSyncError,
    RepositoryError,
    ServiceUnavailableError,
)

logger = structlog.get_logger(__name__)

# Códigos SQLSTATE que corresponden a un error del llamador, no a una falla
# nuestra: conviene reportarlos como 4xx y no como 500.
# https://www.postgresql.org/docs/current/errcodes-appendix.html
UNIQUE_VIOLATION = "23505"
FOREIGN_KEY_VIOLATION = "23503"
CHECK_VIOLATION = "23514"


def _traducir(exc: APIError, operacion: str) -> LifeSyncError:
    """Elige la excepción del proyecto que corresponde al error de PostgREST."""
    codigo = exc.code or ""

    if codigo == UNIQUE_VIOLATION:
        return InvalidValueError("Ese registro ya existe.")
    if codigo == FOREIGN_KEY_VIOLATION:
        return EntityNotFoundError("El registro relacionado no existe.")
    if codigo == CHECK_VIOLATION:
        return InvalidValueError("Alguno de los datos no cumple las reglas de la base.")

    return RepositoryError(f"La base rechazó la operación '{operacion}'.")


@asynccontextmanager
async def traducir_errores(operacion: str) -> AsyncIterator[None]:
    """Convierte fallas de la capa Supabase en errores del proyecto.

    Args:
        operacion: Nombre de la operación, para el log. Nunca datos sensibles.

    Raises:
        InvalidValueError: Violación de unicidad o de un CHECK.
        EntityNotFoundError: Violación de clave foránea.
        RepositoryError: Cualquier otro error de la base (permisos, sintaxis).
        ServiceUnavailableError: No se pudo llegar a la base (red, timeout).
    """
    try:
        yield
    except LifeSyncError:
        # Errores nuestros (p. ej. EncryptionError) ya están traducidos.
        raise
    except APIError as exc:
        logger.error(
            "persistencia.error_api",
            operacion=operacion,
            codigo_postgrest=exc.code,
            mensaje=exc.message,
            hint=exc.hint,
        )
        # `from None` y no `from exc`: el repr de APIError incluye el detalle
        # de PostgreSQL, que en una violación de unicidad trae el valor
        # conflictivo — p. ej. "Key (telefono_whatsapp)=(+549...) already
        # exists." Encadenar la causa lo volcaría al log vía exc_info (RF-18).
        raise _traducir(exc, operacion) from None
    except httpx.HTTPError as exc:
        logger.error("persistencia.error_transporte", operacion=operacion, tipo=type(exc).__name__)
        raise ServiceUnavailableError(f"No se pudo llegar a la base en '{operacion}'.") from exc
