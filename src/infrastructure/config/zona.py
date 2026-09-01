"""Zona horaria de presentación (PB-005 · PB-015).

Vive en un módulo propio porque la comparten dos capas que no deberían
importarse entre sí: el adaptador de Calendar la necesita para no correr de día
los eventos de jornada completa, y las herramientas del agente para mostrar
horarios.

Es fija mientras las preferencias por usuario sean RF-13 (Sprint 2). Cuando
existan, sale del `Usuario` y este módulo pasa a ser sólo el default.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

ZONA_HORARIA = ZoneInfo("America/Argentina/Buenos_Aires")
