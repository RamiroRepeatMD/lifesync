"""Entidad OAuthToken: credencial delegada de un usuario para un proveedor (RF-01).

Importante: en el dominio el token viaja **en texto plano**. El cifrado en
reposo (RF-18) es un detalle de infraestructura que resuelve el repositorio de
`src/infrastructure/persistence/`; el negocio no debe enterarse de que existe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from src.domain.exceptions import InvalidValueError
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth


@dataclass(frozen=True, slots=True)
class OAuthToken:
    """Token OAuth2 emitido por un proveedor para un usuario.

    La combinación (`usuario_id`, `proveedor`) es única: un usuario tiene a lo
    sumo un token vigente por integración.

    Attributes:
        usuario_id: Dueño del token.
        proveedor: Integración que lo emitió.
        access_token: Credencial de acceso, en texto plano.
        refresh_token: Credencial de renovación, en texto plano; opcional.
        expira_en: Vencimiento del access_token; debe tener zona horaria.
        scopes: Permisos concedidos por el usuario.
        id: Asignado por la base al persistir; None mientras no exista.
        creado_en: Primera vez que se guardó; lo completa la base.
        actualizado_en: Última renovación; lo completa la base.
    """

    usuario_id: UUID
    proveedor: ProveedorOAuth
    # repr=False no es cosmético: sin esto el repr autogenerado volcaría las
    # credenciales en cada traceback, cada log que incluya la entidad y cada
    # diff de assert de pytest (RF-18).
    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expira_en: datetime | None = None
    scopes: tuple[str, ...] = ()
    id: UUID | None = None
    creado_en: datetime | None = None
    actualizado_en: datetime | None = None

    def __post_init__(self) -> None:
        """Valida las invariantes de la entidad al construirla."""
        if not self.access_token.strip():
            raise InvalidValueError("El access_token no puede estar vacío.")
        if self.expira_en is not None and self.expira_en.tzinfo is None:
            raise InvalidValueError(
                "expira_en debe tener zona horaria: comparar fechas naive lleva a "
                "renovar tokens antes o después de tiempo."
            )

    def esta_vencido(self, ahora: datetime) -> bool:
        """Indica si el access_token ya venció.

        Un token sin `expira_en` se considera vigente: algunos proveedores
        emiten credenciales sin vencimiento explícito.

        Args:
            ahora: Momento contra el cual comparar; debe tener zona horaria.
        """
        if self.expira_en is None:
            return False
        return self.expira_en <= ahora

    def puede_renovarse(self) -> bool:
        """Indica si hay refresh_token para pedir una credencial nueva."""
        return self.refresh_token is not None
