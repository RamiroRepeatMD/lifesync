"""Adaptador de lectura y escritura de Google Calendar (PB-015 · PB-016, RF-03).

Implementa el puerto `Calendario`. No sabe nada de tokens: se los pide a
`ConectarGoogle.credencial_vigente()`, que ya resuelve el refresco perezoso
desde PB-009.

Regla dura del módulo, igual que en el resto de los adaptadores de Google:
**acá no se loguea ningún título de evento**. Son datos personales; se cuentan,
no se escriben (RF-18).
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import UUID

import httpx
import structlog

from src.application.ports.calendario import Calendario
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.evento import Evento
from src.domain.exceptions import (
    AutorizacionFallidaError,
    CuentaNoConectadaError,
    EntityNotFoundError,
    ServiceUnavailableError,
)
from src.infrastructure.external.google.transporte import traducir_rechazo

logger = structlog.get_logger(__name__)

BASE = "https://www.googleapis.com/calendar/v3"

# Calendarios que Google agrega solo y que no son eventos de la persona: los
# feriados del país y los cumpleaños sacados de los contactos. Aparecen como
# calendarios de grupo y ensuciarían cualquier "¿qué tengo hoy?".
SUFIJOS_DE_RUIDO = ("#holiday@group.v.calendar.google.com", "#contacts@group.v.calendar.google.com")

# Tope de eventos por calendario. Una consulta de "esta semana" con muchos
# calendarios podría traer cientos, y todos terminan en el prompt del modelo:
# es contexto pago y una respuesta ilegible.
MAX_EVENTOS_POR_CALENDARIO = 25
MAX_EVENTOS_TOTALES = 40

TIMEOUT_SEGUNDOS = 10.0

# La lista de calendarios de alguien cambia muy de vez en cuando, y consultarla
# en cada pregunta agrega un viaje a Google sobre un presupuesto de latencia que
# ya está ajustado (RNF de 3 s). Se cachea con la misma forma que el
# deduplicador de WhatsApp: tope duro y vencimiento.
TTL_CACHE_SEGUNDOS = 10 * 60
CAPACIDAD_CACHE = 128


class CacheDeCalendarios:
    """Recuerda qué calendarios tiene cada persona, con tope y vencimiento."""

    def __init__(
        self, capacidad: int = CAPACIDAD_CACHE, ttl_segundos: float = TTL_CACHE_SEGUNDOS
    ) -> None:
        """Recibe sus parámetros por constructor (inyección explícita)."""
        self._capacidad = capacidad
        self._ttl = ttl_segundos
        self._entradas: OrderedDict[UUID, tuple[float, tuple[tuple[str, str], ...]]] = OrderedDict()

    def obtener(self, usuario_id: UUID) -> tuple[tuple[str, str], ...] | None:
        """Devuelve los (id, nombre) cacheados, o None si no hay o vencieron."""
        entrada = self._entradas.get(usuario_id)
        if entrada is None:
            return None
        guardado_en, calendarios = entrada
        if time.monotonic() - guardado_en > self._ttl:
            self._entradas.pop(usuario_id, None)
            return None
        self._entradas.move_to_end(usuario_id)
        return calendarios

    def guardar(self, usuario_id: UUID, calendarios: tuple[tuple[str, str], ...]) -> None:
        """Guarda la lista, desalojando la más vieja si se llenó."""
        self._entradas[usuario_id] = (time.monotonic(), calendarios)
        self._entradas.move_to_end(usuario_id)
        while len(self._entradas) > self._capacidad:
            self._entradas.popitem(last=False)

    def olvidar(self, usuario_id: UUID) -> None:
        """Descarta lo cacheado. Se usa cuando Google rechaza la credencial."""
        self._entradas.pop(usuario_id, None)


class CalendarioGoogle(Calendario):
    """Lee los eventos de Google Calendar de una persona."""

    def __init__(
        self,
        cliente: httpx.AsyncClient,
        conectar: ConectarGoogle,
        cache: CacheDeCalendarios | None = None,
    ) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._cliente = cliente
        self._conectar = conectar
        self._cache = cache if cache is not None else CacheDeCalendarios()

    async def eventos_entre(
        self, usuario_id: UUID, desde: datetime, hasta: datetime
    ) -> tuple[Evento, ...]:
        """Devuelve los eventos de la persona en ese rango, ordenados."""
        comenzo = time.perf_counter()
        token = await self._credencial(usuario_id)
        calendarios = await self._calendarios_de(usuario_id, token)

        if not calendarios:
            logger.info("calendar.sin_calendarios", usuario_id=str(usuario_id))
            return ()

        # En paralelo y no en serie: con N calendarios, secuencial sería N veces
        # la latencia, y el presupuesto ya está ajustado.
        tandas = await asyncio.gather(
            *(
                self._eventos_de(cal_id, nombre, token, desde, hasta)
                for cal_id, nombre in calendarios
            )
        )

        eventos = sorted((e for tanda in tandas for e in tanda), key=lambda e: e.inicio)
        recortados = tuple(eventos[:MAX_EVENTOS_TOTALES])

        logger.info(
            "calendar.eventos_leidos",
            usuario_id=str(usuario_id),
            calendarios=len(calendarios),
            # La cantidad sí; los títulos NUNCA (RF-18).
            eventos=len(recortados),
            recortado=len(eventos) > MAX_EVENTOS_TOTALES,
            duracion_ms=round((time.perf_counter() - comenzo) * 1000),
        )
        return recortados

    async def eventos_del_principal(
        self, usuario_id: UUID, desde: datetime, hasta: datetime
    ) -> tuple[Evento, ...]:
        """Eventos sólo del calendario principal, para la búsqueda previa a borrar."""
        token = await self._credencial(usuario_id)
        eventos = await self._eventos_de("primary", "", token, desde, hasta)
        return tuple(sorted(eventos, key=lambda e: e.inicio))

    # --- Escritura (PB-016) -----------------------------------------------
    #
    # Estos métodos NO piden confirmación: ésa es responsabilidad del grafo
    # del agente, que pausa con `interrupt()` antes de llamarlos (RF-08). Acá
    # se asume que la decisión ya está tomada.

    async def crear_evento(self, usuario_id: UUID, evento: Evento) -> Evento:
        """Crea el evento en el calendario principal de la persona."""
        token = await self._credencial(usuario_id)
        cuerpo: dict[str, Any] = {
            "summary": evento.titulo,
            "start": {"dateTime": evento.inicio.isoformat()},
            "end": {
                "dateTime": (evento.fin if evento.fin is not None else evento.inicio).isoformat()
            },
        }

        datos = await self._mandar("POST", f"{BASE}/calendars/primary/events", token, json=cuerpo)
        creado = _a_evento(datos, "") or evento
        logger.info(
            "calendar.evento_creado",
            usuario_id=str(usuario_id),
            # Que se creó y cuándo sí; el título NUNCA (RF-18).
            con_id=creado.id is not None,
        )
        return creado

    async def modificar_evento(self, usuario_id: UUID, evento: Evento) -> Evento:
        """Aplica el estado deseado con un PATCH al calendario principal."""
        if evento.id is None:
            raise EntityNotFoundError("El evento a modificar no tiene identificador.")

        token = await self._credencial(usuario_id)
        cuerpo: dict[str, Any] = {"summary": evento.titulo}
        if evento.todo_el_dia:
            # Los de día completo van con `date`: mandar dateTime los convierte.
            cuerpo["start"] = {"date": evento.inicio.date().isoformat()}
            fin = evento.fin if evento.fin is not None else evento.inicio
            cuerpo["end"] = {"date": fin.date().isoformat()}
        else:
            cuerpo["start"] = {"dateTime": evento.inicio.isoformat()}
            cuerpo["end"] = {
                "dateTime": (evento.fin if evento.fin is not None else evento.inicio).isoformat()
            }

        datos = await self._mandar(
            "PATCH",
            f"{BASE}/calendars/primary/events/{httpx.URL(evento.id)}",
            token,
            json=cuerpo,
            # A diferencia de eliminar, acá un 404 SÍ es un error: modificar
            # algo que no existe no tiene un estado final equivalente.
            error_404=EntityNotFoundError("El evento ya no existe en el calendario."),
        )
        logger.info("calendar.evento_modificado", usuario_id=str(usuario_id))
        return _a_evento(datos, "") or evento

    async def eliminar_evento(self, usuario_id: UUID, evento_id: str) -> None:
        """Elimina un evento del calendario principal. Idempotente."""
        token = await self._credencial(usuario_id)
        await self._mandar(
            "DELETE",
            f"{BASE}/calendars/primary/events/{httpx.URL(evento_id)}",
            token,
            # Ya-borrado no es un error: el estado final es el mismo.
            tolerar=frozenset({httpx.codes.NOT_FOUND, httpx.codes.GONE}),
        )
        logger.info("calendar.evento_eliminado", usuario_id=str(usuario_id))

    async def _mandar(
        self,
        metodo: str,
        url: str,
        token: str,
        json: dict[str, Any] | None = None,
        tolerar: frozenset[int] = frozenset(),
        error_404: Exception | None = None,
    ) -> dict[str, Any]:
        """Petición de escritura autenticada, con la traducción de errores común.

        `error_404` permite que cada operación le dé su semántica al 404:
        eliminar lo tolera (ya-borrado es el mismo estado final), modificar lo
        convierte en "no existe".
        """
        try:
            respuesta = await self._cliente.request(
                metodo, url, json=json, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            logger.error("calendar.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Google Calendar.") from None

        if respuesta.status_code in tolerar:
            return {}
        if respuesta.status_code == httpx.codes.NOT_FOUND and error_404 is not None:
            raise error_404
        self._traducir_rechazo(respuesta)
        return _json_o_vacio(respuesta)

    def _traducir_rechazo(self, respuesta: httpx.Response) -> None:
        """Convierte un status de error en la excepción del dominio que toca."""
        traducir_rechazo(respuesta, "calendar")

    # --- Credencial ------------------------------------------------------

    async def _credencial(self, usuario_id: UUID) -> str:
        """Consigue un access_token vigente, renovándolo si hace falta."""
        token = await self._conectar.credencial_vigente(usuario_id, datetime.now(UTC))
        if token is None:
            raise CuentaNoConectadaError
        return token.access_token

    # --- Lista de calendarios --------------------------------------------

    async def _calendarios_de(self, usuario_id: UUID, token: str) -> tuple[tuple[str, str], ...]:
        """Devuelve los (id, nombre) de los calendarios que hay que leer."""
        cacheados = self._cache.obtener(usuario_id)
        if cacheados is not None:
            return cacheados

        datos = await self._pedir(
            f"{BASE}/users/me/calendarList",
            token,
            {"minAccessRole": "reader"},
            usuario_id=usuario_id,
        )
        calendarios = tuple(
            (item["id"], item.get("summary") or "")
            for item in datos.get("items", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str) and _hay_que_leerlo(item)
        )
        self._cache.guardar(usuario_id, calendarios)
        logger.info(
            "calendar.calendarios_listados",
            usuario_id=str(usuario_id),
            total=len(datos.get("items", [])),
            a_leer=len(calendarios),
        )
        return calendarios

    # --- Eventos de un calendario ----------------------------------------

    async def _eventos_de(
        self, calendario_id: str, nombre: str, token: str, desde: datetime, hasta: datetime
    ) -> list[Evento]:
        """Trae y normaliza los eventos de un calendario."""
        datos = await self._pedir(
            f"{BASE}/calendars/{httpx.URL(calendario_id)}/events",
            token,
            {
                "timeMin": desde.isoformat(),
                "timeMax": hasta.isoformat(),
                # NO es opcional: sin esto un evento semanal devuelve la regla
                # de recurrencia en vez de las ocurrencias, y "¿qué tengo hoy?"
                # no encontraría la reunión de todos los lunes.
                "singleEvents": "true",
                # Sólo válido junto con singleEvents.
                "orderBy": "startTime",
                "maxResults": str(MAX_EVENTOS_POR_CALENDARIO),
            },
        )
        return [
            evento
            for item in datos.get("items", [])
            if isinstance(item, dict) and (evento := _a_evento(item, nombre)) is not None
        ]

    # --- HTTP -------------------------------------------------------------

    async def _pedir(
        self,
        url: str,
        token: str,
        parametros: dict[str, str],
        usuario_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Hace un GET autenticado y traduce lo que salga mal."""
        try:
            respuesta = await self._cliente.get(
                url, params=parametros, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            logger.error("calendar.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Google Calendar.") from None

        if respuesta.status_code == httpx.codes.UNAUTHORIZED:
            # El token dejó de servir aunque no estuviera vencido: lo más común
            # es que la persona revocara el acceso desde su cuenta de Google.
            if usuario_id is not None:
                self._cache.olvidar(usuario_id)
            logger.warning("calendar.credencial_rechazada")
            raise AutorizacionFallidaError(
                "Google rechazó la credencial. Hay que volver a conectar la cuenta."
            )

        if respuesta.status_code >= httpx.codes.BAD_REQUEST:
            logger.error(
                "calendar.rechazado",
                status_code=respuesta.status_code,
                # Diagnóstico de la API, no dato de la persona.
                motivo=_motivo_de(respuesta),
            )
            raise ServiceUnavailableError("Google Calendar no pudo responder la consulta.")

        cuerpo = _json_o_vacio(respuesta)
        return cuerpo


# --- Funciones puras ---------------------------------------------------------


def _hay_que_leerlo(item: dict[str, Any]) -> bool:
    """Decide si un calendario aporta eventos de la persona.

    Dos señales, las dos con significado propio:

    - Los calendarios de grupo de Google —feriados, cumpleaños— no son cosas
      que la persona agendó.
    - `selected: false` significa que lo escondió en su propia interfaz de
      Google Calendar. Si no lo quiere ver ahí, tampoco acá.
    """
    identificador = item.get("id", "")
    if any(identificador.endswith(sufijo) for sufijo in SUFIJOS_DE_RUIDO):
        return False
    return item.get("selected", False) is True


def _a_evento(item: dict[str, Any], calendario: str) -> Evento | None:
    """Convierte un evento de la API en la entidad, o None si no se puede."""
    inicio, todo_el_dia = _momento(item.get("start"))
    if inicio is None:
        return None
    fin, _ = _momento(item.get("end"))

    identificador = item.get("id")
    return Evento(
        titulo=item.get("summary") or "",
        inicio=inicio,
        # En los de día completo Google marca el fin al día siguiente a las 00:00.
        fin=None if todo_el_dia else fin,
        todo_el_dia=todo_el_dia,
        calendario=calendario or None,
        id=identificador if isinstance(identificador, str) else None,
    )


def _momento(borde: Any) -> tuple[datetime | None, bool]:
    """Lee un `start`/`end` de Google, que tiene dos formas posibles.

    Los eventos con horario traen `dateTime`; los de día completo traen `date`
    a secas. Leer sólo `dateTime` saltea los de día completo **en silencio**.

    Returns:
        (momento, es_de_dia_completo). El momento siempre con zona horaria.
    """
    if not isinstance(borde, dict):
        return None, False

    con_hora = borde.get("dateTime")
    if isinstance(con_hora, str):
        try:
            momento = datetime.fromisoformat(con_hora)
        except ValueError:
            return None, False
        # Google siempre manda offset, pero si faltara, asumir UTC es mejor que
        # construir un Evento naive: la entidad lo rechazaría.
        return (momento if momento.tzinfo else momento.replace(tzinfo=UTC)), False

    solo_fecha = borde.get("date")
    if isinstance(solo_fecha, str):
        try:
            dia = date.fromisoformat(solo_fecha)
        except ValueError:
            return None, False
        return datetime.combine(dia, datetime.min.time(), tzinfo=UTC), True

    return None, False


def _motivo_de(respuesta: httpx.Response) -> str | None:
    """Saca el mensaje de error que explica el rechazo."""
    error = _json_o_vacio(respuesta).get("error")
    if not isinstance(error, dict):
        return None
    mensaje = error.get("message")
    return mensaje if isinstance(mensaje, str) else None


def _json_o_vacio(respuesta: httpx.Response) -> dict[str, Any]:
    """Devuelve el cuerpo como dict, o vacío si no tiene esa forma."""
    try:
        datos = respuesta.json()
    except ValueError:
        return {}
    return datos if isinstance(datos, dict) else {}


def create_calendario_google(
    cliente: httpx.AsyncClient, conectar: ConectarGoogle
) -> CalendarioGoogle:
    """Arma el adaptador de calendario. Se llama una vez, en el `lifespan`."""
    return CalendarioGoogle(cliente, conectar)


def rango_del_dia(dia: date, zona: Any) -> tuple[datetime, datetime]:
    """Devuelve (00:00, 24:00) de ese día en la zona indicada.

    Se usa para traducir "hoy" o "mañana" a un rango que Google entienda.
    """
    inicio = datetime.combine(dia, datetime.min.time(), tzinfo=zona)
    return inicio, inicio + timedelta(days=1)
