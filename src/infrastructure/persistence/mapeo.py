"""Utilidades de conversión entre filas de PostgREST y tipos de Python.

PostgREST devuelve JSON: todo llega como `str`, `list` o `None`. Estas
funciones centralizan el parseo para que los repositorios queden legibles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from src.domain.exceptions import RepositoryError


def filas_de(respuesta: Any) -> list[dict[str, Any]]:
    """Extrae `data` de una respuesta de PostgREST como lista de filas."""
    datos: Any = respuesta.data
    if datos is None:
        return []
    if not isinstance(datos, list):
        raise RepositoryError("PostgREST devolvió un cuerpo inesperado.")
    return [fila for fila in datos if isinstance(fila, dict)]


def a_uuid(valor: object, campo: str) -> UUID:
    """Convierte un valor de la fila en UUID.

    Raises:
        RepositoryError: Si la columna falta o no es un UUID válido.
    """
    if not isinstance(valor, str):
        raise RepositoryError(f"La columna '{campo}' no vino como texto en la respuesta.")
    try:
        return UUID(valor)
    except ValueError as exc:
        raise RepositoryError(f"La columna '{campo}' no es un UUID válido.") from exc


def a_datetime(valor: object, campo: str) -> datetime | None:
    """Convierte un timestamptz de PostgREST en datetime con zona horaria."""
    if valor is None:
        return None
    if not isinstance(valor, str):
        raise RepositoryError(f"La columna '{campo}' no vino como texto en la respuesta.")
    try:
        return datetime.fromisoformat(valor)
    except ValueError as exc:
        raise RepositoryError(f"La columna '{campo}' no es una fecha ISO válida.") from exc


def a_texto(valor: object, campo: str) -> str:
    """Devuelve una columna de texto obligatoria."""
    if not isinstance(valor, str):
        raise RepositoryError(f"Falta la columna '{campo}' en la respuesta.")
    return valor


def a_texto_opcional(valor: object, campo: str) -> str | None:
    """Devuelve una columna de texto que puede ser NULL."""
    if valor is None:
        return None
    return a_texto(valor, campo)


def a_tupla_de_textos(valor: object, campo: str) -> tuple[str, ...]:
    """Convierte una columna `text[]` en una tupla inmutable."""
    if valor is None:
        return ()
    if not isinstance(valor, list):
        raise RepositoryError(f"La columna '{campo}' no vino como arreglo.")
    return tuple(str(elemento) for elemento in valor)


def formatear_fecha(momento: datetime | None) -> str | None:
    """Serializa un datetime para enviarlo a PostgREST."""
    return None if momento is None else momento.isoformat()
