"""Dobles de prueba: dejan testear la persistencia sin red ni base de datos.

`FakeSupabaseClient` imita la cadena fluida del cliente real
(`table().select().eq().execute()`) y **registra cada llamada**. Eso es lo que
permite afirmar QUÉ se le mandó a la base — en particular, que el token viajó
cifrado y no en claro.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.ports.agente import AgenteConversacional
from src.application.ports.autorizador_google import (
    AutorizadorGoogle,
    CredencialesGoogle,
)
from src.application.ports.calendario import Calendario
from src.application.ports.correos import Correos
from src.application.ports.tareas import Tareas
from src.application.ports.whatsapp import MensajeroWhatsApp
from src.domain.entities.correo import Correo, CorreoSaliente
from src.domain.entities.evento import Evento
from src.domain.entities.oauth_token import OAuthToken
from src.domain.entities.tarea import Tarea
from src.domain.entities.usuario import Usuario
from src.domain.exceptions import EntityNotFoundError, InvalidValueError
from src.domain.repositories.oauth_token_repository import OAuthTokenRepository
from src.domain.repositories.usuario_repository import UsuarioRepository
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp
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


class RepositorioUsuarioEnMemoria(UsuarioRepository):
    """Implementación en memoria del puerto de usuarios, sin base de datos."""

    def __init__(self, fallar_con: Exception | None = None) -> None:
        self._por_telefono: dict[str, Usuario] = {}
        self._por_id: dict[UUID, Usuario] = {}
        # Permite simular una base caída en los tests del borde del webhook.
        self.fallar_con = fallar_con

    async def crear(self, usuario: Usuario) -> Usuario:
        self._verificar_falla()
        if usuario.telefono_whatsapp in self._por_telefono:
            raise InvalidValueError("Ese registro ya existe.")
        persistido = replace(usuario, id=uuid4())
        self._por_telefono[persistido.telefono_whatsapp] = persistido
        if persistido.id is not None:
            self._por_id[persistido.id] = persistido
        return persistido

    async def obtener_o_crear(self, telefono: str, nombre: str | None = None) -> Usuario:
        self._verificar_falla()
        existente = self._por_telefono.get(telefono)
        if existente is not None:
            return existente
        return await self.crear(Usuario(telefono_whatsapp=telefono, nombre=nombre))

    async def obtener_por_id(self, usuario_id: UUID) -> Usuario | None:
        self._verificar_falla()
        return self._por_id.get(usuario_id)

    async def obtener_por_telefono(self, telefono: str) -> Usuario | None:
        self._verificar_falla()
        return self._por_telefono.get(telefono)

    def _verificar_falla(self) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con


class MensajeroFalso(MensajeroWhatsApp):
    """Doble del puerto de envío: registra lo enviado en vez de llamar a Meta."""

    def __init__(self, fallar_con: Exception | None = None) -> None:
        self.enviados: list[tuple[NumeroWhatsApp, str]] = []
        self.fallar_con = fallar_con

    async def enviar_texto(self, destino: NumeroWhatsApp, texto: str) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.enviados.append((destino, texto))

    @property
    def textos(self) -> list[str]:
        """Sólo los textos, para aserciones más cortas."""
        return [texto for _, texto in self.enviados]


class AgenteFalso(AgenteConversacional):
    """Doble del puerto del agente: registra las consultas en vez de llamar a Gemini."""

    def __init__(self, respuesta: str = "respuesta del agente") -> None:
        self.consultas: list[ConsultaDelUsuario] = []
        self.respuesta = respuesta
        self.fallar_con: Exception | None = None

    async def responder(self, consulta: ConsultaDelUsuario) -> str:
        self.consultas.append(consulta)
        if self.fallar_con is not None:
            raise self.fallar_con
        return self.respuesta

    @property
    def textos(self) -> list[str]:
        """Sólo lo que se le preguntó, para aserciones más cortas."""
        return [consulta.texto for consulta in self.consultas]


class ModeloFalso(BaseChatModel):
    """Chat model de mentira, para ejercitar el grafo sin red ni API key.

    Devuelve las respuestas de `guion` en orden, una por invocación. Poniendo
    un `AIMessage` con `tool_calls` en la primera y uno con texto en la segunda
    se recorre el ciclo completo agente → herramientas → agente.

    Registra cada lista de mensajes que recibió, que es lo que permite afirmar
    que el system prompt viaja siempre y que el historial se recortó.
    """

    guion: list[AIMessage] = Field(default_factory=list)
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)
    herramientas_asociadas: list[Any] = Field(default_factory=list)
    # Índices (desde 0) de las llamadas que fallan en vez de responder: es lo
    # que permite reproducir "el modelo se cae DESPUÉS de una escritura"
    # (PB-026). La llamada que falla consume su lugar del guion.
    fallar_en_llamadas: frozenset[int] = Field(default_factory=frozenset)
    error: str = "504 DEADLINE_EXCEEDED (simulado)"

    @property
    def _llm_type(self) -> str:
        return "modelo-falso"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ModeloFalso:
        """Recuerda con qué herramientas se lo ató y se devuelve a sí mismo.

        Devolver `self` y no un `Runnable` envuelto es a propósito: mantiene
        accesible `recibidos` después del bind, que es donde están las
        aserciones.
        """
        self.herramientas_asociadas = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recibidos.append(list(messages))
        if len(self.recibidos) - 1 in self.fallar_en_llamadas:
            raise RuntimeError(self.error)
        indice = min(len(self.recibidos) - 1, len(self.guion) - 1)
        respuesta = self.guion[indice] if self.guion else AIMessage("")
        return ChatResult(generations=[ChatGeneration(message=respuesta)])


class CalendarioFalso(Calendario):
    """Doble del puerto de calendario: registra a quién se le consultó qué.

    `consultados` es lo que permite afirmar la propiedad de seguridad de
    PB-015: que la agenda que se lee es la de quien escribe y no la que pida
    el texto del mensaje.
    """

    def __init__(
        self,
        eventos: tuple[Evento, ...] | None = None,
        fallar_con: Exception | None = None,
    ) -> None:
        self.eventos = eventos if eventos is not None else ()
        self.fallar_con = fallar_con
        self.consultados: list[UUID] = []
        self.rangos: list[tuple[datetime, datetime]] = []
        # Registros de escritura (PB-016): son la evidencia de los tests de
        # RF-08 — "sin resume no hay escritura" se afirma sobre estas listas.
        self.creados: list[tuple[UUID, Evento]] = []
        self.eliminados: list[tuple[UUID, str]] = []
        self.modificados: list[tuple[UUID, Evento]] = []

    async def eventos_entre(
        self, usuario_id: UUID, desde: datetime, hasta: datetime
    ) -> tuple[Evento, ...]:
        self.consultados.append(usuario_id)
        self.rangos.append((desde, hasta))
        if self.fallar_con is not None:
            raise self.fallar_con
        return self.eventos

    async def eventos_del_principal(
        self, usuario_id: UUID, desde: datetime, hasta: datetime
    ) -> tuple[Evento, ...]:
        return await self.eventos_entre(usuario_id, desde, hasta)

    async def crear_evento(self, usuario_id: UUID, evento: Evento) -> Evento:
        if self.fallar_con is not None:
            raise self.fallar_con
        creado = replace(evento, id="evento-creado-1")
        self.creados.append((usuario_id, creado))
        return creado

    async def eliminar_evento(self, usuario_id: UUID, evento_id: str) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.eliminados.append((usuario_id, evento_id))

    async def modificar_evento(self, usuario_id: UUID, evento: Evento) -> Evento:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.modificados.append((usuario_id, evento))
        return evento


class TareasFalsas(Tareas):
    """Doble del puerto `Tareas` (PB-028 · PB-029), con registro de escrituras."""

    def __init__(
        self,
        pendientes: tuple[Tarea, ...] = (),
        fallar_con: Exception | None = None,
    ) -> None:
        self.lista = pendientes
        self.fallar_con = fallar_con
        self.consultados: list[UUID] = []
        self.creadas: list[tuple[UUID, Tarea]] = []
        self.completadas: list[tuple[UUID, str]] = []
        self.pospuestas: list[tuple[UUID, str, date]] = []
        self.eliminadas: list[tuple[UUID, str]] = []

    async def pendientes(self, usuario_id: UUID) -> tuple[Tarea, ...]:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.consultados.append(usuario_id)
        return self.lista

    async def crear(self, usuario_id: UUID, tarea: Tarea) -> Tarea:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.creadas.append((usuario_id, tarea))
        return tarea

    async def completar(self, usuario_id: UUID, tarea_id: str) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.completadas.append((usuario_id, tarea_id))

    async def posponer(self, usuario_id: UUID, tarea_id: str, vencimiento: date) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.pospuestas.append((usuario_id, tarea_id, vencimiento))

    async def eliminar(self, usuario_id: UUID, tarea_id: str) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.eliminadas.append((usuario_id, tarea_id))


class CorreosFalsos(Correos):
    """Doble del puerto `Correos` (PB-033): una bandeja fija, sin filtrar."""

    def __init__(
        self, correos: tuple[Correo, ...] = (), fallar_con: Exception | None = None
    ) -> None:
        self.bandeja = correos
        self.fallar_con = fallar_con
        self.busquedas: list[tuple[UUID, str, int]] = []
        self.leidos: list[tuple[UUID, str]] = []
        self.enviados: list[tuple[UUID, CorreoSaliente]] = []

    async def buscar(self, usuario_id: UUID, consulta: str, cantidad: int) -> tuple[Correo, ...]:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.busquedas.append((usuario_id, consulta, cantidad))
        return self.bandeja

    async def leer(self, usuario_id: UUID, correo_id: str) -> Correo:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.leidos.append((usuario_id, correo_id))
        for correo in self.bandeja:
            if correo.id == correo_id:
                return correo
        raise EntityNotFoundError("Ese correo no existe.")

    async def enviar(self, usuario_id: UUID, saliente: CorreoSaliente) -> str:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.enviados.append((usuario_id, saliente))
        return f"enviado-{len(self.enviados)}"


class AutorizadorFalso(AutorizadorGoogle):
    """Doble del puerto de autorización: no habla con Google.

    `usuario_del_estado` devuelve `usuario_fijo` sin mirar el `state`, salvo que
    se cargue `estado_invalido`. La verificación real de la firma se prueba en
    `test_estado_oauth.py`, que es donde vive esa lógica.
    """

    def __init__(
        self,
        usuario_fijo: UUID | None = None,
        credenciales: CredencialesGoogle | None = None,
    ) -> None:
        self.usuario_fijo = usuario_fijo or uuid4()
        self.credenciales = credenciales or CredencialesGoogle(
            access_token="access-de-prueba",
            refresh_token="refresh-de-prueba",
            scopes=("https://www.googleapis.com/auth/calendar.readonly",),
        )
        self.estado_invalido: Exception | None = None
        self.fallar_canje: Exception | None = None
        self.fallar_refresco: Exception | None = None
        self.codigos_canjeados: list[str] = []
        self.refresh_usados: list[str] = []
        self.revocados: list[str] = []
        self.revocacion_exitosa = True

    def url_de_autorizacion(self, usuario_id: UUID, ahora: datetime) -> str:
        return f"https://accounts.google.com/o/oauth2/v2/auth?state=firmado-{usuario_id}"

    def usuario_del_estado(self, estado: str, ahora: datetime) -> UUID:
        if self.estado_invalido is not None:
            raise self.estado_invalido
        return self.usuario_fijo

    async def revocar(self, token: str) -> bool:
        self.revocados.append(token)
        return self.revocacion_exitosa

    async def canjear_codigo(self, codigo: str) -> CredencialesGoogle:
        self.codigos_canjeados.append(codigo)
        if self.fallar_canje is not None:
            raise self.fallar_canje
        return self.credenciales

    async def refrescar(self, refresh_token: str) -> CredencialesGoogle:
        self.refresh_usados.append(refresh_token)
        if self.fallar_refresco is not None:
            raise self.fallar_refresco
        return self.credenciales


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
