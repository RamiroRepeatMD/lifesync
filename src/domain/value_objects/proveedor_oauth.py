"""Proveedor de identidad OAuth2 con el que el usuario conecta su cuenta (RF-01).

Es un value object: no tiene identidad propia, sólo representa cuál de las
integraciones soportadas emitió un token.
"""

from __future__ import annotations

from enum import StrEnum


class ProveedorOAuth(StrEnum):
    """Proveedores OAuth2 soportados por LifeSync.

    El valor de cada miembro es el que se persiste en la columna `proveedor`
    de la tabla `oauth_tokens`, y debe coincidir con el CHECK de la migración.
    """

    GOOGLE = "google"
    NOTION = "notion"
