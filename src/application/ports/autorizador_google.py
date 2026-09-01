"""Puerto de autorización con Google (PB-009).

Declara una **capacidad** —conseguir y renovar credenciales delegadas—, así que
vive acá y no en `domain/repositories/`, igual que `MensajeroWhatsApp` y
`AgenteConversacional`.

El caso de uso no sabe que del otro lado hay peticiones HTTP a
`oauth2.googleapis.com`: pide una URL, canjea un código y renueva. La
implementación vive en `src/infrastructure/external/google/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class CredencialesGoogle:
    """Lo que Google devuelve al canjear un código o renovar un token.

    Attributes:
        access_token: Credencial de acceso. `repr=False` porque es un secreto y
            no puede aparecer en un traceback ni en un assert de pytest (RF-18).
        expira_en: Vencimiento calculado, siempre con zona horaria.
        scopes: Permisos que Google efectivamente concedió, que pueden ser
            menos que los pedidos si la persona destildó alguno.
        refresh_token: **Puede venir None.** Google lo emite en el primer
            consentimiento; en una renovación no lo repite. Quien persista esto
            tiene que conservar el guardado en vez de pisarlo con None.
    """

    access_token: str = field(repr=False)
    expira_en: datetime | None = None
    scopes: tuple[str, ...] = ()
    refresh_token: str | None = field(default=None, repr=False)


class AutorizadorGoogle(ABC):
    """Capacidad de obtener y renovar credenciales OAuth2 de Google."""

    @abstractmethod
    def url_de_autorizacion(self, usuario_id: UUID, ahora: datetime) -> str:
        """Arma la URL de consentimiento a la que hay que mandar a la persona.

        El `usuario_id` viaja firmado dentro del `state`: es lo que permite
        saber, cuando Google vuelva, quién había pedido conectar.
        """

    @abstractmethod
    def usuario_del_estado(self, estado: str, ahora: datetime) -> UUID:
        """Verifica el `state` que devolvió Google y extrae el usuario.

        Raises:
            InvalidValueError: Si está deformado, mal firmado o vencido.
        """

    @abstractmethod
    async def canjear_codigo(self, codigo: str) -> CredencialesGoogle:
        """Cambia el código de un solo uso por credenciales.

        Raises:
            AutorizacionFallidaError: Si Google rechaza el canje.
        """

    @abstractmethod
    async def refrescar(self, refresh_token: str) -> CredencialesGoogle:
        """Pide un access_token nuevo con el refresh_token.

        Raises:
            AutorizacionFallidaError: Si Google rechaza la renovación, por
                ejemplo porque la persona revocó el acceso.
        """
