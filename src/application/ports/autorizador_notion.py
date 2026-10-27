"""Puerto de autorización con Notion (PB-014).

Como `AutorizadorGoogle`: declara una **capacidad** —conseguir, renovar y
revocar credenciales delegadas— sin que el caso de uso sepa que del otro lado
hay HTTP. La implementación vive en `src/infrastructure/external/notion/`.

Notion no es Google en dos cosas que importan acá: no informa cuándo vence el
access_token (se renueva cuando la API contesta 401) y puede devolver un
refresh_token nuevo en cada renovación.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class CredencialesNotion:
    """Lo que Notion devuelve al canjear un código o al renovar.

    Attributes:
        access_token: Credencial de acceso. `repr=False` porque es un secreto y
            no puede aparecer en un traceback ni en un assert de pytest (RF-18).
        refresh_token: Credencial de renovación. **Puede venir None**: la
            referencia de Notion la tipa "string or null". Quien persista una
            renovación conserva la guardada en vez de pisarla con None.
    """

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)


class AutorizadorNotion(ABC):
    """Capacidad de obtener, renovar y revocar credenciales OAuth2 de Notion."""

    @abstractmethod
    def url_de_autorizacion(self, usuario_id: UUID, ahora: datetime) -> str:
        """Arma la URL de consentimiento, con el usuario firmado en el `state`."""

    @abstractmethod
    def usuario_del_estado(self, estado: str, ahora: datetime) -> UUID:
        """Verifica el `state` que devolvió Notion y extrae el usuario.

        Raises:
            InvalidValueError: Si está deformado, mal firmado o vencido.
        """

    @abstractmethod
    async def canjear_codigo(self, codigo: str) -> CredencialesNotion:
        """Cambia el código de un solo uso por credenciales.

        Raises:
            AutorizacionFallidaError: Si Notion rechaza el canje.
        """

    @abstractmethod
    async def refrescar(self, refresh_token: str) -> CredencialesNotion:
        """Pide credenciales nuevas con el refresh_token.

        Raises:
            AutorizacionFallidaError: Si Notion rechaza la renovación, por
                ejemplo porque la persona quitó la conexión.
        """

    @abstractmethod
    async def revocar(self, token: str) -> bool:
        """Invalida una credencial del lado de Notion (RF-12). Best-effort: nunca lanza.

        Returns:
            True si quedó invalidada (también si ya no valía); False si no se
            pudo confirmar.
        """
