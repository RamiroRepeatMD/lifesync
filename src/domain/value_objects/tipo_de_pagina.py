"""Qué es lo que se encontró en Notion: una página o una base de datos (PB-014).

Un value object del dominio y no el `object` de la API: la persona habla de
"la página de Recetas" o de "la base de Gastos", y cada una se lee distinto.
"""

from __future__ import annotations

from enum import StrEnum


class TipoDePagina(StrEnum):
    """Las dos cosas que la búsqueda de Notion puede devolver."""

    PAGINA = "página"
    BASE = "base de datos"
