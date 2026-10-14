"""Qué puede hacer con un archivo la persona con quien se comparte (PB-038).

Un enum del dominio y no el `role` del proveedor: la persona dice "que lo pueda
ver" o "que lo pueda editar", y es el adaptador el que lo traduce al rol de
Google Drive. Comentarista queda fuera de lo básico.
"""

from __future__ import annotations

from enum import StrEnum


class PermisoDeArchivo(StrEnum):
    """Ver o editar."""

    LECTOR = "lector"
    EDITOR = "editor"
