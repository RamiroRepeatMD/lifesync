"""Dobles de prueba: dejan testear la persistencia sin red ni base de datos.

`FakeSupabaseClient` imita la cadena fluida del cliente real
(`table().select().eq().execute()`) y **registra cada llamada**. Eso es lo que
permite afirmar QUÉ se le mandó a la base — en particular, que el token viajó
cifrado y no en claro.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID, uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.ports.agente import AgenteConversacional
from src.application.ports.archivos import Archivos
from src.application.ports.archivos_recibidos import ArchivoDescargado, ArchivosRecibidos
from src.application.ports.autorizador_google import (
    AutorizadorGoogle,
    CredencialesGoogle,
)
from src.application.ports.autorizador_notion import AutorizadorNotion, CredencialesNotion
from src.application.ports.calendario import Calendario
from src.application.ports.correos import Correos
from src.application.ports.paginas import (
    NOMBRE_DEL_TITULO,
    BaseConsultada,
    Bloque,
    FilaDeBase,
    PaginaLeida,
    Paginas,
    Propiedad,
    TipoDePropiedad,
    ValorDePropiedad,
)
from src.application.ports.tareas import Tareas
from src.application.ports.whatsapp import MensajeroWhatsApp
from src.domain.entities.archivo import Archivo
from src.domain.entities.correo import Correo, CorreoSaliente
from src.domain.entities.evento import Evento
from src.domain.entities.oauth_token import OAuthToken
from src.domain.entities.pagina import Pagina
from src.domain.entities.recordatorio import EstadoDeRecordatorio, Recordatorio
from src.domain.entities.tarea import Tarea
from src.domain.entities.usuario import Usuario
from src.domain.exceptions import (
    ArchivoDemasiadoGrandeError,
    EntityNotFoundError,
    InvalidValueError,
)
from src.domain.repositories.oauth_token_repository import OAuthTokenRepository
from src.domain.repositories.recordatorio_repository import RecordatorioRepository
from src.domain.repositories.usuario_repository import UsuarioRepository
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp
from src.domain.value_objects.permiso_de_archivo import PermisoDeArchivo
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo
from src.domain.value_objects.tipo_de_pagina import TipoDePagina


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

    def lte(self, columna: str, valor: Any) -> ConsultaFalsa:
        self._llamada.filtros[f"{columna}<="] = valor
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
        fallar_al_escribir: Exception | None = None,
        escrituras_antes_de_fallar: int = 0,
    ) -> None:
        self.eventos = eventos if eventos is not None else ()
        self.fallar_con = fallar_con
        # Sólo fallan las escrituras (PB-040): la búsqueda previa anda, como
        # cuando Google se cae entre la confirmación y la escritura. Las
        # primeras `escrituras_antes_de_fallar` salen bien (un borrado de
        # varios que se corta a mitad de camino).
        self.fallar_al_escribir = fallar_al_escribir
        self.escrituras_antes_de_fallar = escrituras_antes_de_fallar
        self.consultados: list[UUID] = []
        self.rangos: list[tuple[datetime, datetime]] = []
        # Registros de escritura (PB-016): son la evidencia de los tests de
        # RF-08 — "sin resume no hay escritura" se afirma sobre estas listas.
        self.creados: list[tuple[UUID, Evento]] = []
        self.eliminados: list[tuple[UUID, str]] = []
        self.modificados: list[tuple[UUID, Evento]] = []

    def _escribir(self) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        if self.fallar_al_escribir is not None:
            if self.escrituras_antes_de_fallar <= 0:
                raise self.fallar_al_escribir
            self.escrituras_antes_de_fallar -= 1

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
        self._escribir()
        creado = replace(evento, id="evento-creado-1")
        self.creados.append((usuario_id, creado))
        return creado

    async def eliminar_evento(self, usuario_id: UUID, evento_id: str) -> None:
        self._escribir()
        self.eliminados.append((usuario_id, evento_id))

    async def modificar_evento(self, usuario_id: UUID, evento: Evento) -> Evento:
        self._escribir()
        self.modificados.append((usuario_id, evento))
        return evento


class TareasFalsas(Tareas):
    """Doble del puerto `Tareas` (PB-028 · PB-029), con registro de escrituras."""

    def __init__(
        self,
        pendientes: tuple[Tarea, ...] = (),
        fallar_con: Exception | None = None,
        fallar_al_escribir: Exception | None = None,
    ) -> None:
        self.lista = pendientes
        self.fallar_con = fallar_con
        # Sólo fallan las escrituras (PB-040): ver `CalendarioFalso`.
        self.fallar_al_escribir = fallar_al_escribir
        self.consultados: list[UUID] = []
        self.creadas: list[tuple[UUID, Tarea]] = []
        self.completadas: list[tuple[UUID, str]] = []
        self.pospuestas: list[tuple[UUID, str, date]] = []
        self.eliminadas: list[tuple[UUID, str]] = []

    def _escribir(self) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        if self.fallar_al_escribir is not None:
            raise self.fallar_al_escribir

    async def pendientes(self, usuario_id: UUID) -> tuple[Tarea, ...]:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.consultados.append(usuario_id)
        return self.lista

    async def crear(self, usuario_id: UUID, tarea: Tarea) -> Tarea:
        self._escribir()
        self.creadas.append((usuario_id, tarea))
        return tarea

    async def completar(self, usuario_id: UUID, tarea_id: str) -> None:
        self._escribir()
        self.completadas.append((usuario_id, tarea_id))

    async def posponer(self, usuario_id: UUID, tarea_id: str, vencimiento: date) -> None:
        self._escribir()
        self.pospuestas.append((usuario_id, tarea_id, vencimiento))

    async def eliminar(self, usuario_id: UUID, tarea_id: str) -> None:
        self._escribir()
        self.eliminadas.append((usuario_id, tarea_id))


class CorreosFalsos(Correos):
    """Doble del puerto `Correos` (PB-033): una bandeja fija, sin filtrar."""

    def __init__(
        self,
        correos: tuple[Correo, ...] = (),
        fallar_con: Exception | None = None,
        fallar_al_enviar: Exception | None = None,
    ) -> None:
        self.bandeja = correos
        self.fallar_con = fallar_con
        # Sólo falla el envío (PB-035): responder y reenviar leen el original antes.
        self.fallar_al_enviar = fallar_al_enviar
        self.busquedas: list[tuple[UUID, str, int]] = []
        self.leidos: list[tuple[UUID, str]] = []
        # Los ids que se leyeron sin recortar (para reenviar, PB-034).
        self.leidos_completos: list[str] = []
        self.enviados: list[tuple[UUID, CorreoSaliente]] = []
        self.accesos_comprobados = 0

    async def comprobar_acceso(self, usuario_id: UUID) -> None:
        self.accesos_comprobados += 1
        if self.fallar_con is not None:
            raise self.fallar_con

    async def buscar(self, usuario_id: UUID, consulta: str, cantidad: int) -> tuple[Correo, ...]:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.busquedas.append((usuario_id, consulta, cantidad))
        return self.bandeja

    async def leer(self, usuario_id: UUID, correo_id: str, *, completo: bool = False) -> Correo:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.leidos.append((usuario_id, correo_id))
        if completo:
            self.leidos_completos.append(correo_id)
        for correo in self.bandeja:
            if correo.id == correo_id:
                return correo
        raise EntityNotFoundError("Ese correo no existe.")

    async def enviar(self, usuario_id: UUID, saliente: CorreoSaliente) -> str:
        if self.fallar_con is not None:
            raise self.fallar_con
        if self.fallar_al_enviar is not None:
            raise self.fallar_al_enviar
        self.enviados.append((usuario_id, saliente))
        return f"enviado-{len(self.enviados)}"


class ArchivosFalsos(Archivos):
    """Doble del puerto `Archivos` (PB-037 · PB-038 · PB-039): un Drive fijo, sin filtrar.

    `fallar_con` hace fallar todo (cuenta, lecturas y escrituras);
    `fallar_al_escribir`, sólo las escrituras: lo de antes del sí anduvo.
    """

    def __init__(
        self,
        archivos: tuple[Archivo, ...] = (),
        fallar_con: Exception | None = None,
        *,
        carpetas: tuple[Archivo, ...] = (),
        fallar_al_escribir: Exception | None = None,
    ) -> None:
        self.archivos = archivos
        # Las carpetas propias, la más reciente primero (como las ordena Drive).
        self.carpetas_propias = carpetas
        self.fallar_con = fallar_con
        self.fallar_al_escribir = fallar_al_escribir
        self.busquedas: list[tuple[UUID, str, TipoDeArchivo | None, datetime | None, int]] = []
        self.subidos: list[tuple[UUID, str, str, bytes, str | None]] = []
        self.compartidos: list[tuple[UUID, str, PermisoDeArchivo, str | None]] = []
        self.renombrados: list[tuple[UUID, str, str]] = []
        self.movidos: list[tuple[UUID, str, str | None]] = []
        self.en_la_papelera: list[tuple[UUID, str]] = []
        self.accesos_comprobados = 0

    def _fallar(self, *, escritura: bool = False) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        if escritura and self.fallar_al_escribir is not None:
            raise self.fallar_al_escribir

    async def comprobar_acceso(self, usuario_id: UUID) -> None:
        self.accesos_comprobados += 1
        self._fallar()

    async def obtener(self, usuario_id: UUID, archivo_id: str) -> Archivo:
        self._fallar()
        for archivo in (*self.archivos, *self.carpetas_propias):
            if archivo.id == archivo_id:
                return archivo
        raise EntityNotFoundError("Ese archivo no está en tu Drive.")

    async def carpetas(self, usuario_id: UUID, nombre: str) -> tuple[Archivo, ...]:
        self._fallar()
        return tuple(c for c in self.carpetas_propias if nombre.lower() in c.nombre.lower())

    async def subir(
        self,
        usuario_id: UUID,
        nombre: str,
        tipo_mime: str,
        contenido: bytes,
        carpeta_id: str | None,
    ) -> Archivo:
        self._fallar(escritura=True)
        self.subidos.append((usuario_id, nombre, tipo_mime, contenido, carpeta_id))
        nuevo = f"subido-{len(self.subidos)}"
        subido = Archivo(
            id=nuevo,
            nombre=nombre,
            tipo=None,
            modificado=datetime.now(UTC),
            enlace=f"https://drive.google.com/file/d/{nuevo}/view",
            propio=True,
        )
        # Queda en el Drive: después se lo puede leer y compartir.
        self.archivos = (*self.archivos, subido)
        return subido

    async def compartir(
        self,
        usuario_id: UUID,
        archivo_id: str,
        permiso: PermisoDeArchivo,
        direccion: str | None,
    ) -> None:
        self._fallar(escritura=True)
        self.compartidos.append((usuario_id, archivo_id, permiso, direccion))

    async def renombrar(self, usuario_id: UUID, archivo_id: str, nombre: str) -> None:
        self._fallar(escritura=True)
        self.renombrados.append((usuario_id, archivo_id, nombre))

    async def mover(self, usuario_id: UUID, archivo_id: str, carpeta_id: str | None) -> None:
        self._fallar(escritura=True)
        self.movidos.append((usuario_id, archivo_id, carpeta_id))

    async def a_la_papelera(self, usuario_id: UUID, archivo_id: str) -> None:
        self._fallar(escritura=True)
        self.en_la_papelera.append((usuario_id, archivo_id))

    async def buscar(
        self,
        usuario_id: UUID,
        texto: str,
        tipo: TipoDeArchivo | None,
        desde: datetime | None,
        cantidad: int,
    ) -> tuple[Archivo, ...]:
        if self.fallar_con is not None:
            raise self.fallar_con
        self.busquedas.append((usuario_id, texto, tipo, desde, cantidad))
        return self.archivos


class PaginasFalsas(Paginas):
    """Doble del puerto `Paginas` (PB-014 · RF-07): un Notion fijo, sin filtrar.

    Los ids son UUID, como los de Notion. `fallar_con` hace fallar todo;
    `fallar_al_escribir`, sólo las escrituras: lo de antes del sí anduvo (con
    `escrituras_antes_de_fallar`, recién después de ésas: un corte a mitad).
    Leer algo que no está, o una base como página, da `EntityNotFoundError`,
    que es lo que hace Notion con lo que no existe o no está compartido.
    """

    def __init__(
        self,
        paginas: Sequence[Pagina] = (),
        *,
        contenidos: dict[str, str] | None = None,
        filas: dict[str, tuple[FilaDeBase, ...]] | None = None,
        propiedades: dict[str, tuple[Propiedad, ...]] | None = None,
        esquemas: dict[str, tuple[Propiedad, ...]] | None = None,
        limite: int | None = 5 * 1024 * 1024,
        fallar_con: Exception | None = None,
        fallar_al_escribir: Exception | None = None,
        escrituras_antes_de_fallar: int = 0,
        fallar_al_ordenar: Exception | None = None,
        fallar_una_sola_vez: bool = False,
    ) -> None:
        self.paginas = tuple(paginas)
        self.contenidos = contenidos or {}
        self.filas = filas or {}
        self.propiedades_de = propiedades or {}
        self.esquemas = esquemas or {}
        self.limite = limite
        self.fallar_con = fallar_con
        self.fallar_al_escribir = fallar_al_escribir
        self.escrituras_antes_de_fallar = escrituras_antes_de_fallar
        self.fallar_al_ordenar = fallar_al_ordenar
        # Una falla pasajera (un 429): la escritura que sigue ya anda.
        self.fallar_una_sola_vez = fallar_una_sola_vez
        self.busquedas: list[tuple[UUID, str, int]] = []
        self.leidas: list[str] = []
        self.consultadas: list[str] = []
        self.creadas: list[tuple[UUID, str, str, tuple[Bloque, ...]]] = []
        # Los datos de cada `crear`, en el mismo orden que `creadas`.
        self.datos_de_creadas: list[tuple[tuple[Propiedad, ValorDePropiedad], ...]] = []
        self.bases_creadas: list[tuple[UUID, str, str, tuple[Propiedad, ...]]] = []
        # La primera columna de cada base, en el mismo orden que `bases_creadas`.
        self.primeras_columnas: list[str] = []
        # (base, columnas, ordenar_por) de cada `ordenar_tabla`.
        self.tablas_ordenadas: list[tuple[str, tuple[str, ...], str | None]] = []
        self.agregados: list[tuple[UUID, str, tuple[Bloque, ...]]] = []
        self.cambios: list[tuple[UUID, str, str, ValorDePropiedad]] = []
        self.movidas: list[tuple[UUID, str, str]] = []
        self.en_la_papelera: list[tuple[UUID, str]] = []
        self.subidos: list[tuple[UUID, str, str, str, bytes]] = []

    def escrituras(self) -> int:
        """Cuántas escrituras llegaron, de cualquier clase."""
        return (
            len(self.creadas)
            + len(self.bases_creadas)
            + len(self.tablas_ordenadas)
            + len(self.agregados)
            + len(self.cambios)
            + len(self.movidas)
            + len(self.en_la_papelera)
            + len(self.subidos)
        )

    async def buscar(self, usuario_id: UUID, texto: str, cantidad: int) -> tuple[Pagina, ...]:
        self._fallar()
        self.busquedas.append((usuario_id, texto, cantidad))
        return self.paginas[: max(1, min(cantidad, 10))]

    async def leer(self, usuario_id: UUID, pagina_id: str) -> PaginaLeida:
        self._fallar()
        self.leidas.append(pagina_id)
        pagina = self._buscar_por_id(pagina_id, TipoDePagina.PAGINA)
        return PaginaLeida(pagina=pagina, markdown=self.contenidos.get(pagina_id, ""))

    async def consultar(self, usuario_id: UUID, base_id: str, cantidad: int) -> BaseConsultada:
        self._fallar()
        self.consultadas.append(base_id)
        base = self._buscar_por_id(base_id, TipoDePagina.BASE)
        filas = self.filas.get(base_id, ())
        return BaseConsultada(base=base, filas=filas[:cantidad], hay_mas=len(filas) > cantidad)

    async def obtener(self, usuario_id: UUID, pagina_id: str) -> Pagina:
        self._fallar()
        for pagina in self.paginas:
            if pagina.id == pagina_id:
                return pagina
        raise EntityNotFoundError("No está en el Notion de la persona, o no está compartido.")

    async def propiedades(
        self, usuario_id: UUID, pagina_id: str
    ) -> tuple[Pagina, tuple[Propiedad, ...]]:
        self._fallar()
        pagina = self._buscar_por_id(pagina_id, TipoDePagina.PAGINA)
        titulo = Propiedad(
            id="title", nombre="title", tipo=TipoDePropiedad.TITULO, valor=pagina.titulo
        )
        return pagina, self.propiedades_de.get(pagina_id, (titulo,))

    async def crear(
        self,
        usuario_id: UUID,
        padre: Pagina,
        titulo: str,
        bloques: tuple[Bloque, ...],
        datos: tuple[tuple[Propiedad, ValorDePropiedad], ...] = (),
    ) -> Pagina:
        self._fallar(escritura=True)
        self.creadas.append((usuario_id, padre.id, titulo, bloques))
        self.datos_de_creadas.append(datos)
        nueva = Pagina(
            id=str(uuid4()),
            titulo=titulo,
            tipo=TipoDePagina.PAGINA,
            editada=datetime.now(UTC),
            enlace="https://www.notion.so/nueva",
            en_base=padre.tipo is TipoDePagina.BASE,
        )
        self.paginas = (*self.paginas, nueva)
        return nueva

    async def crear_base(
        self,
        usuario_id: UUID,
        padre: Pagina,
        titulo: str,
        columnas: tuple[Propiedad, ...],
        primera_columna: str = NOMBRE_DEL_TITULO,
    ) -> Pagina:
        self._fallar(escritura=True)
        self.bases_creadas.append((usuario_id, padre.id, titulo, columnas))
        self.primeras_columnas.append(primera_columna)
        nueva = Pagina(
            id=str(uuid4()),
            titulo=titulo,
            tipo=TipoDePagina.BASE,
            editada=datetime.now(UTC),
            enlace="https://www.notion.so/base-nueva",
        )
        self.paginas = (*self.paginas, nueva)
        # Como Notion: la columna de título, con su nombre, y después las otras.
        titulo_de_la_base = Propiedad(
            id="title", nombre=primera_columna, tipo=TipoDePropiedad.TITULO
        )
        self.esquemas[nueva.id] = (titulo_de_la_base, *columnas)
        return nueva

    async def ordenar_tabla(
        self,
        usuario_id: UUID,
        base_id: str,
        columnas: tuple[str, ...],
        ordenar_por: str | None = None,
    ) -> None:
        # Primero falla, después anota: un orden fallido no cuenta como escritura.
        self._fallar(escritura=True)
        if self.fallar_al_ordenar is not None:
            raise self.fallar_al_ordenar
        self._buscar_por_id(base_id, TipoDePagina.BASE)
        self.tablas_ordenadas.append((base_id, columnas, ordenar_por))

    async def esquema(self, usuario_id: UUID, base_id: str) -> tuple[Propiedad, ...]:
        self._fallar()
        self._buscar_por_id(base_id, TipoDePagina.BASE)
        return self.esquemas.get(base_id, ())

    async def agregar(self, usuario_id: UUID, pagina_id: str, bloques: tuple[Bloque, ...]) -> None:
        self._fallar(escritura=True)
        self.agregados.append((usuario_id, pagina_id, bloques))

    async def cambiar(
        self, usuario_id: UUID, pagina_id: str, propiedad: Propiedad, valor: ValorDePropiedad
    ) -> None:
        self._fallar(escritura=True)
        self.cambios.append((usuario_id, pagina_id, propiedad.id, valor))

    async def mover(self, usuario_id: UUID, pagina_id: str, destino_id: str) -> None:
        self._fallar(escritura=True)
        self.movidas.append((usuario_id, pagina_id, destino_id))

    async def a_la_papelera(self, usuario_id: UUID, pagina_id: str) -> None:
        self._fallar(escritura=True)
        self.en_la_papelera.append((usuario_id, pagina_id))

    async def limite_de_subida(self, usuario_id: UUID) -> int | None:
        self._fallar()
        return self.limite

    async def subir(
        self, usuario_id: UUID, pagina_id: str, nombre: str, tipo_mime: str, contenido: bytes
    ) -> None:
        self._fallar(escritura=True)
        self.subidos.append((usuario_id, pagina_id, nombre, tipo_mime, contenido))

    def _fallar(self, *, escritura: bool = False) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
        hechas = self.escrituras() >= self.escrituras_antes_de_fallar
        if escritura and self.fallar_al_escribir is not None and hechas:
            falla = self.fallar_al_escribir
            if self.fallar_una_sola_vez:
                self.fallar_al_escribir = None
            raise falla

    def _buscar_por_id(self, identificador: str, tipo: TipoDePagina) -> Pagina:
        for pagina in self.paginas:
            if pagina.id == identificador and pagina.tipo is tipo:
                return pagina
        raise EntityNotFoundError("No está en el Notion de la persona, o no está compartido.")


class RecibidosFalsos(ArchivosRecibidos):
    """Doble de `ArchivosRecibidos` (PB-038): siempre el mismo PDF.

    `descargas` registra cada pedido, también los que fallan: es lo que deja
    afirmar que nada se baja antes del sí.
    """

    def __init__(
        self, contenido: bytes = b"%PDF-1.4 de prueba", fallar_con: Exception | None = None
    ) -> None:
        self.contenido = contenido
        self.fallar_con = fallar_con
        self.descargas: list[str] = []

    async def descargar(self, referencia: str, maximo_bytes: int) -> ArchivoDescargado:
        self.descargas.append(referencia)
        if self.fallar_con is not None:
            raise self.fallar_con
        if len(self.contenido) > maximo_bytes:
            raise ArchivoDemasiadoGrandeError("Supera el tope.")
        return ArchivoDescargado(contenido=self.contenido, tipo_mime="application/pdf")


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


class AutorizadorNotionFalso(AutorizadorNotion):
    """Doble del autorizador de Notion (PB-014): no habla con Notion.

    `refrescar` devuelve un par nuevo en cada llamada, como cuando Notion lo
    rota. Con `esperar` cargado se queda esperando ese evento antes de contestar:
    es lo que deja probar que dos renovaciones a la vez hacen un solo refresco
    (sin la espera, el test pasaría aunque faltara el candado). `al_refrescar`
    corre al empezar: simula lo que hace otro proceso mientras tanto.
    """

    def __init__(self, usuario_fijo: UUID | None = None) -> None:
        self.usuario_fijo = usuario_fijo or uuid4()
        self.credenciales = CredencialesNotion(
            access_token="notion-access-1", refresh_token="notion-refresh-1"
        )
        self.estado_invalido: Exception | None = None
        self.fallar_canje: Exception | None = None
        self.fallar_refresco: Exception | None = None
        self.refresh_nulo = False
        self.esperar: asyncio.Event | None = None
        self.al_refrescar: Callable[[], Awaitable[None]] | None = None
        self.codigos_canjeados: list[str] = []
        self.refresh_usados: list[str] = []
        self.revocados: list[str] = []
        self.revocacion_exitosa = True

    def url_de_autorizacion(self, usuario_id: UUID, ahora: datetime) -> str:
        return f"https://api.notion.com/v1/oauth/authorize?state=firmado-{usuario_id}"

    def usuario_del_estado(self, estado: str, ahora: datetime) -> UUID:
        if self.estado_invalido is not None:
            raise self.estado_invalido
        return self.usuario_fijo

    async def canjear_codigo(self, codigo: str) -> CredencialesNotion:
        self.codigos_canjeados.append(codigo)
        if self.fallar_canje is not None:
            raise self.fallar_canje
        return self.credenciales

    async def refrescar(self, refresh_token: str) -> CredencialesNotion:
        self.refresh_usados.append(refresh_token)
        if self.al_refrescar is not None:
            await self.al_refrescar()
        if self.esperar is not None:
            await self.esperar.wait()
        if self.fallar_refresco is not None:
            raise self.fallar_refresco
        numero = len(self.refresh_usados) + 1
        return CredencialesNotion(
            access_token=f"notion-access-{numero}",
            refresh_token=None if self.refresh_nulo else f"notion-refresh-{numero}",
        )

    async def revocar(self, token: str) -> bool:
        self.revocados.append(token)
        return self.revocacion_exitosa


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


class RecordatoriosEnMemoria(RecordatorioRepository):
    """Doble del repositorio de recordatorios (PB-030), con compare-and-set de verdad.

    Attributes:
        guardados: Todos los recordatorios, por id, en su estado actual.
        reclamos_perdidos: Ids cuyo reclamo (pendiente → enviando) "gana otro
            despachador", como pasa con dos contenedores durante un deploy.
        fallar_con: Excepción a lanzar en cualquier operación (base caída).
    """

    def __init__(self, *iniciales: Recordatorio, fallar_con: Exception | None = None) -> None:
        self.guardados: dict[UUID, Recordatorio] = {}
        for recordatorio in iniciales:
            self._guardar(recordatorio if recordatorio.id else replace(recordatorio, id=uuid4()))
        self.reclamos_perdidos: set[UUID] = set()
        self.fallar_con = fallar_con

    async def crear(self, recordatorio: Recordatorio) -> Recordatorio:
        self._verificar_falla()
        return self._guardar(replace(recordatorio, id=uuid4()))

    async def pendientes_de(self, usuario_id: UUID) -> list[Recordatorio]:
        self._verificar_falla()
        return sorted(
            (r for r in self._pendientes() if r.usuario_id == usuario_id),
            key=lambda r: r.momento,
        )

    async def vencidos(self, hasta: datetime, limite: int) -> list[Recordatorio]:
        self._verificar_falla()
        vencidos = [r for r in self._pendientes() if r.momento <= hasta]
        return sorted(vencidos, key=lambda r: r.momento)[:limite]

    async def cambiar_estado(
        self,
        recordatorio_id: UUID,
        desde: EstadoDeRecordatorio,
        hacia: EstadoDeRecordatorio,
    ) -> bool:
        self._verificar_falla()
        if recordatorio_id in self.reclamos_perdidos and desde is EstadoDeRecordatorio.PENDIENTE:
            return False
        actual = self.guardados.get(recordatorio_id)
        if actual is None or actual.estado is not desde:
            return False
        self.guardados[recordatorio_id] = replace(actual, estado=hacia)
        return True

    def estado_de(self, recordatorio_id: UUID | None) -> EstadoDeRecordatorio:
        """El estado actual de un recordatorio, para las aserciones."""
        assert recordatorio_id is not None
        return self.guardados[recordatorio_id].estado

    def _guardar(self, recordatorio: Recordatorio) -> Recordatorio:
        assert recordatorio.id is not None
        self.guardados[recordatorio.id] = recordatorio
        return recordatorio

    def _pendientes(self) -> list[Recordatorio]:
        return [r for r in self.guardados.values() if r.estado is EstadoDeRecordatorio.PENDIENTE]

    def _verificar_falla(self) -> None:
        if self.fallar_con is not None:
            raise self.fallar_con
