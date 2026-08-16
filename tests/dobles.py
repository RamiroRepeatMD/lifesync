"""Dobles de prueba: dejan testear la persistencia sin red ni base de datos.

`FakeSupabaseClient` imita la cadena fluida del cliente real
(`table().select().eq().execute()`) y **registra cada llamada**. Eso es lo que
permite afirmar QUÉ se le mandó a la base — en particular, que el token viajó
cifrado y no en claro.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from src.domain.entities.oauth_token import OAuthToken
from src.domain.repositories.oauth_token_repository import OAuthTokenRepository
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth


@dataclass
class Llamada:
    """Registro de una operación pedida al cliente falso."""

    tabla: str
    operacion: str
    payload: Any = None
    on_conflict: str | None = None
    filtros: dict[str, Any] = field(default_factory=dict)


class RespuestaFalsa:
    """Imita `postgrest.APIResponse`."""

    def __init__(self, data: Any) -> None:
        self.data = data


class ConsultaFalsa:
    """Builder fluido que registra lo pedido y devuelve una respuesta preparada."""

    def __init__(self, cliente: FakeSupabaseClient, tabla: str) -> None:
        self._cliente = cliente
        self._llamada = Llamada(tabla=tabla, operacion="?")

    # --- Operaciones ---

    def select(self, *columnas: str, **kwargs: Any) -> ConsultaFalsa:
        self._llamada.operacion = "select"
        return self

    def insert(self, fila: Any, **kwargs: Any) -> ConsultaFalsa:
        self._llamada.operacion = "insert"
        self._llamada.payload = fila
        return self

    def upsert(self, fila: Any, *, on_conflict: str = "", **kwargs: Any) -> ConsultaFalsa:
        self._llamada.operacion = "upsert"
        self._llamada.payload = fila
        self._llamada.on_conflict = on_conflict
        return self

    def update(self, fila: Any, **kwargs: Any) -> ConsultaFalsa:
        self._llamada.operacion = "update"
        self._llamada.payload = fila
        return self

    def delete(self, **kwargs: Any) -> ConsultaFalsa:
        self._llamada.operacion = "delete"
        return self

    # --- Modificadores ---

    def eq(self, columna: str, valor: Any) -> ConsultaFalsa:
        self._llamada.filtros[columna] = valor
        return self

    def limit(self, cantidad: int) -> ConsultaFalsa:
        return self

    def order(self, *args: Any, **kwargs: Any) -> ConsultaFalsa:
        return self

    # --- Ejecución ---

    async def execute(self) -> RespuestaFalsa:
        """Registra la llamada y devuelve (o lanza) lo que se haya programado."""
        self._cliente.llamadas.append(self._llamada)

        error = self._cliente.errores.get(self._llamada.tabla)
        if error is not None:
            raise error

        return RespuestaFalsa(self._cliente.respuestas.get(self._llamada.tabla, []))


class FakeSupabaseClient:
    """Doble de `supabase.AsyncClient` para tests sin red.

    Attributes:
        respuestas: Filas que devuelve `execute()`, por tabla.
        errores: Excepción a lanzar en `execute()`, por tabla.
        llamadas: Historial de operaciones, para hacer aserciones sobre el
            payload que realmente salió hacia la base.
    """

    def __init__(
        self,
        respuestas: dict[str, list[dict[str, Any]]] | None = None,
        errores: dict[str, Exception] | None = None,
    ) -> None:
        self.respuestas: dict[str, list[dict[str, Any]]] = respuestas or {}
        self.errores: dict[str, Exception] = errores or {}
        self.llamadas: list[Llamada] = []

    def table(self, nombre: str) -> ConsultaFalsa:
        return ConsultaFalsa(self, nombre)

    # --- Ayudas para las aserciones ---

    def ultima_llamada(self) -> Llamada:
        """Devuelve la última operación registrada."""
        if not self.llamadas:
            raise AssertionError("No se registró ninguna llamada al cliente.")
        return self.llamadas[-1]


class RepositorioOAuthTokenEnMemoria(OAuthTokenRepository):
    """Implementación en memoria del puerto, sin cifrado ni base.

    Sirve para los tests de casos de uso que van a aparecer en PB-009 y en el
    Sprint 2: sólo interesa el comportamiento del puerto, no cómo se persiste.
    """

    def __init__(self) -> None:
        self._tokens: dict[tuple[UUID, ProveedorOAuth], OAuthToken] = {}

    async def guardar(self, token: OAuthToken) -> OAuthToken:
        from dataclasses import replace

        persistido = token if token.id is not None else replace(token, id=uuid4())
        self._tokens[(persistido.usuario_id, persistido.proveedor)] = persistido
        return persistido

    async def obtener(self, usuario_id: UUID, proveedor: ProveedorOAuth) -> OAuthToken | None:
        return self._tokens.get((usuario_id, proveedor))

    async def listar_por_usuario(self, usuario_id: UUID) -> list[OAuthToken]:
        return [token for (uid, _), token in self._tokens.items() if uid == usuario_id]

    async def eliminar(self, usuario_id: UUID, proveedor: ProveedorOAuth) -> None:
        self._tokens.pop((usuario_id, proveedor), None)
