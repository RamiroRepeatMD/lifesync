"""Qué clase de archivo es, en palabras de la persona (PB-037).

Un enum del dominio y no el `mimeType` del proveedor: la persona pide "la
planilla" o "los PDF", y es el adaptador el que traduce cada tipo a los
formatos concretos de Google Drive (una planilla de Google o un Excel, por
ejemplo). El modelo elige entre estos tipos; nunca escribe un `mimeType`.
"""

from __future__ import annotations

from enum import StrEnum


class TipoDeArchivo(StrEnum):
    """Los tipos por los que se puede buscar, y los que se muestran."""

    DOCUMENTO = "documento"
    PLANILLA = "planilla"
    PRESENTACION = "presentación"
    PDF = "PDF"
    IMAGEN = "imagen"
    CARPETA = "carpeta"
