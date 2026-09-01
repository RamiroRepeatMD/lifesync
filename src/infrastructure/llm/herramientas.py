"""Herramientas que el agente puede invocar (PB-005 · PB-015 · PB-016).

El docstring de cada herramienta **es la documentación que ve el modelo**: es
lo que lee para decidir cuándo llamarla y con qué. Se escribe para él, no para
nosotros.

Sobre el parámetro `runtime`: LangGraph lo inyecta solo y **lo excluye del
esquema que viaja al modelo**, así que el LLM ni siquiera sabe que existe. Es
lo que hace imposible que un mensaje elija de quién es la agenda que se
consulta. Ver `contexto.py`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

import structlog
from langchain_core.tools import BaseTool, tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolRuntime
from langgraph.types import interrupt

from src.application.ports.calendario import Calendario
from src.domain.entities.evento import Evento
from src.domain.exceptions import (
    AutorizacionFallidaError,
    CuentaNoConectadaError,
    PermisoInsuficienteError,
    ServiceUnavailableError,
)
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.llm.contexto import ContextoDeAgente

logger = structlog.get_logger(__name__)

_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MESES = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)

# Tope de días que se pueden consultar de una. Sin esto, un "¿qué tengo este
# año?" traería cientos de eventos al prompt: caro e ilegible.
MAX_DIAS_DE_RANGO = 31

Runtime = ToolRuntime[ContextoDeAgente, MessagesState]


def fecha_en_palabras(momento: datetime) -> str:
    """Formatea una fecha en español.

    A mano y no con `strftime("%A")`: el nombre del día que devuelve strftime
    depende del locale del sistema, y en el contenedor de Railway ese locale es
    C, o sea inglés.
    """
    return f"{_DIAS[momento.weekday()]} {momento.day} de {_MESES[momento.month - 1]}"


@tool
def fecha_y_hora_actual() -> str:
    """Devuelve la fecha y la hora de ahora en la zona horaria de Argentina.

    Sólo hace falta si te preguntan la hora exacta: la fecha de hoy ya la
    tenés en tus instrucciones.
    """
    ahora = datetime.now(UTC).astimezone(ZONA_HORARIA)
    logger.info("agente.herramienta.invocada", herramienta="fecha_y_hora_actual")
    return f"{fecha_en_palabras(ahora)} de {ahora.year}, {ahora:%H:%M} (hora de Argentina)"


def construir_herramientas(calendario: Calendario | None) -> tuple[BaseTool, ...]:
    """Arma la lista de herramientas del agente.

    El `Calendario` entra por acá y no por el contexto de la invocación porque
    vive todo el proceso: lo único que cambia entre mensajes es de quién es la
    agenda, y eso viaja en `ContextoDeAgente`.

    Si no hay calendario —falta la configuración de OAuth— la herramienta
    simplemente no se ofrece, en vez de ofrecerse y fallar siempre.
    """
    if calendario is None:
        return (fecha_y_hora_actual,)

    @tool
    async def eventos_del_calendario(desde: str, hasta: str, runtime: Runtime) -> str:
        """Consulta los eventos del calendario de la persona entre dos fechas.

        Usala para cualquier pregunta sobre su agenda: qué tiene hoy, mañana,
        el viernes o esta semana.

        Args:
            desde: Primer día del rango, en formato AAAA-MM-DD. Incluido.
            hasta: Último día del rango, en formato AAAA-MM-DD. Incluido.
                Para un solo día, poné la misma fecha en los dos.
        """
        return await _consultar_agenda(calendario, runtime, desde, hasta)

    @tool
    async def crear_evento_en_calendario(
        titulo: str,
        fecha: str,
        hora_inicio: str,
        duracion_minutos: int = 60,
        *,
        runtime: Runtime,
    ) -> str:
        """Crea un evento en el calendario de la persona, previa confirmación.

        La confirmación la maneja el sistema: vos sólo llamá a la herramienta
        con los datos. Nunca digas que el evento ya se creó hasta que la
        herramienta te lo confirme.

        Args:
            titulo: Nombre del evento, corto y claro.
            fecha: Día del evento, en formato AAAA-MM-DD.
            hora_inicio: Hora de comienzo, en formato HH:MM de 24 horas.
            duracion_minutos: Cuánto dura, en minutos. Si la persona no lo
                dijo, dejá el valor por defecto.
        """
        return await _crear_evento(
            calendario, runtime, titulo, fecha, hora_inicio, duracion_minutos
        )

    @tool
    async def eliminar_evento_del_calendario(fecha: str, titulo: str, *, runtime: Runtime) -> str:
        """Elimina un evento del calendario principal, previa confirmación.

        Buscá siempre por el día y el nombre aproximado: el sistema encuentra
        el evento exacto y pide la confirmación. Nunca digas que se eliminó
        hasta que la herramienta te lo confirme.

        Args:
            fecha: Día en que está el evento, en formato AAAA-MM-DD.
            titulo: Nombre (o parte del nombre) del evento a eliminar.
        """
        return await _eliminar_evento(calendario, runtime, fecha, titulo)

    return (
        fecha_y_hora_actual,
        eventos_del_calendario,
        crear_evento_en_calendario,
        eliminar_evento_del_calendario,
    )


async def _consultar_agenda(
    calendario: Calendario, runtime: Runtime, desde: str, hasta: str
) -> str:
    """Resuelve la consulta y la traduce a algo que el modelo pueda relatar.

    **No lanza nunca.** Una excepción cortaría el turno y dejaría a la persona
    sin respuesta; en cambio, un texto explicando qué pasó el modelo lo puede
    contar con sus palabras (RF-19).
    """
    usuario_id = runtime.context.usuario_id

    rango = _rango(desde, hasta)
    if rango is None:
        return (
            "No entendí las fechas. Necesito el rango en formato AAAA-MM-DD, "
            f"y como mucho {MAX_DIAS_DE_RANGO} días."
        )

    comenzo = time.perf_counter()
    try:
        eventos = await calendario.eventos_entre(usuario_id, *rango)
    except CuentaNoConectadaError as exc:
        return exc.mensaje_usuario
    except AutorizacionFallidaError:
        return (
            "Se venció el permiso para ver tu calendario. Escribí /conectar "
            "para volver a autorizarlo."
        )
    except ServiceUnavailableError:
        return "No pude consultar tu calendario ahora mismo. Probá de nuevo en un minuto."

    logger.info(
        "agente.herramienta.invocada",
        herramienta="eventos_del_calendario",
        usuario_id=str(usuario_id),
        # Cantidad y rango sí; los títulos NUNCA (RF-18).
        eventos=len(eventos),
        dias=(rango[1] - rango[0]).days,
        duracion_ms=round((time.perf_counter() - comenzo) * 1000),
    )
    return _redactar(eventos, rango[0])


def _rango(desde: str, hasta: str) -> tuple[datetime, datetime] | None:
    """Convierte el rango del modelo en instantes con zona horaria.

    Devuelve None si las fechas no se entienden o el rango es absurdo: es
    entrada generada por un modelo, así que se valida como cualquier otra.
    """
    try:
        primero = date.fromisoformat(desde)
        ultimo = date.fromisoformat(hasta)
    except ValueError:
        return None

    if ultimo < primero or (ultimo - primero).days > MAX_DIAS_DE_RANGO:
        return None

    inicio = datetime.combine(primero, datetime.min.time(), tzinfo=ZONA_HORARIA)
    # `hasta` es inclusivo para la persona, así que el corte va al día siguiente.
    fin = datetime.combine(ultimo, datetime.min.time(), tzinfo=ZONA_HORARIA) + timedelta(days=1)
    return inicio, fin


def _redactar(eventos: Sequence[Evento], desde: datetime) -> str:
    """Arma el texto que lee el modelo.

    Se agrupa por día porque "¿qué tengo esta semana?" en una lista plana es
    ilegible, y el modelo termina reordenándola por su cuenta.
    """
    if not eventos:
        return f"No hay eventos agendados desde el {fecha_en_palabras(desde)}."

    lineas: list[str] = []
    dia_actual: date | None = None
    for evento in eventos:
        local = _en_hora_local(evento)
        if local.date() != dia_actual:
            dia_actual = local.date()
            lineas.append(f"\n{fecha_en_palabras(local)}:")
        lineas.append(f"- {_linea(evento, local)}")

    return "\n".join(lineas).strip()


def _en_hora_local(evento: Evento) -> datetime:
    """Pasa el evento a hora local, salvo los de jornada completa.

    **Un evento de día completo no se convierte de zona.** Google los entrega
    como una fecha sin hora, y convertir esa medianoche a otro huso lo corre un
    día: un cumpleaños del 27 aparecía el 26. La fecha de un evento de todo el
    día es la fecha, sin importar el huso desde el que se mire.
    """
    if evento.todo_el_dia:
        return evento.inicio
    return evento.inicio.astimezone(ZONA_HORARIA)


def _linea(evento: Evento, local: datetime) -> str:
    """Una línea por evento, con la hora sólo si la tiene."""
    partes = [evento.titulo_visible]
    if evento.todo_el_dia:
        partes.append("(todo el día)")
    else:
        horario = f"{local:%H:%M}"
        if evento.fin is not None:
            horario += f" a {evento.fin.astimezone(ZONA_HORARIA):%H:%M}"
        partes.insert(0, f"{horario} —")
    if evento.calendario:
        partes.append(f"[{evento.calendario}]")
    return " ".join(partes)


# --- Escritura con confirmación obligatoria (PB-016, RF-08) ------------------
#
# La estructura de las dos funciones es la garantía de RF-08, así que vale
# dejarla explícita:
#
#   1. Validar y armar el resumen        ← puro; se RE-EJECUTA al reanudar
#   2. interrupt({"resumen": ...})        ← el grafo se PAUSA acá
#   3. La llamada que escribe             ← corre UNA vez, sólo con aprobación
#
# El punto 1 se re-ejecuta porque LangGraph reanuda la tool desde el principio
# (verificado con una sonda antes de diseñar esto): por eso ahí no puede haber
# nada con efectos. Y no existe ningún camino hacia el punto 3 que no pase por
# el 2 — eso es lo que convierte la confirmación de instrucción en estructura.

MINUTOS_MINIMOS = 5
MINUTOS_MAXIMOS = 12 * 60


async def _crear_evento(
    calendario: Calendario,
    runtime: Runtime,
    titulo: str,
    fecha: str,
    hora_inicio: str,
    duracion_minutos: int,
) -> str:
    """Crea un evento en el calendario principal, con confirmación en el medio."""
    usuario_id = runtime.context.usuario_id

    titulo = titulo.strip()
    inicio = _momento_local(fecha, hora_inicio)
    if not titulo or inicio is None:
        return (
            "No entendí los datos del evento. Necesito un título, la fecha en "
            "AAAA-MM-DD y la hora en HH:MM."
        )
    if not MINUTOS_MINIMOS <= duracion_minutos <= MINUTOS_MAXIMOS:
        return f"La duración tiene que estar entre {MINUTOS_MINIMOS} y {MINUTOS_MAXIMOS} minutos."

    fin = inicio + timedelta(minutes=duracion_minutos)
    resumen = f'Crear "{titulo}" el {fecha_en_palabras(inicio)} de {inicio:%H:%M} a {fin:%H:%M}'

    decision = interrupt({"resumen": resumen})
    if not (isinstance(decision, dict) and decision.get("aprobado") is True):
        return "La persona lo canceló. No se creó nada."

    try:
        await calendario.crear_evento(usuario_id, Evento(titulo=titulo, inicio=inicio, fin=fin))
    except (CuentaNoConectadaError, PermisoInsuficienteError) as exc:
        return exc.mensaje_usuario
    except AutorizacionFallidaError:
        return "Se venció el permiso del calendario. Escribí /conectar para renovarlo."
    except ServiceUnavailableError:
        return "No pude guardar el evento ahora mismo. Probá de nuevo en un minuto."

    logger.info(
        "agente.herramienta.invocada",
        herramienta="crear_evento_en_calendario",
        usuario_id=str(usuario_id),
        # Que se creó sí; el título NUNCA (RF-18).
        duracion_minutos=duracion_minutos,
    )
    return f"Evento creado: {resumen[len('Crear ') :]}"


async def _eliminar_evento(
    calendario: Calendario, runtime: Runtime, fecha: str, titulo: str
) -> str:
    """Elimina un evento del principal, desambiguando ANTES de confirmar.

    La desambiguación es determinística y nuestra, nunca del modelo: con cero
    o varias coincidencias no hay interrupt, porque no hay una acción concreta
    que confirmar. Sólo se pausa cuando el evento a borrar es exactamente uno.
    """
    usuario_id = runtime.context.usuario_id

    rango = _rango(fecha, fecha)
    if rango is None or not titulo.strip():
        return "Necesito el día (AAAA-MM-DD) y el nombre del evento a eliminar."

    # Esta lectura se re-ejecuta al reanudar (la tool arranca de cero): es
    # idempotente, y de paso re-verifica que el evento siga existiendo.
    try:
        eventos = await calendario.eventos_del_principal(usuario_id, *rango)
    except (CuentaNoConectadaError, PermisoInsuficienteError) as exc:
        return exc.mensaje_usuario
    except AutorizacionFallidaError:
        return "Se venció el permiso del calendario. Escribí /conectar para renovarlo."
    except ServiceUnavailableError:
        return "No pude consultar tu calendario ahora mismo. Probá de nuevo en un minuto."

    candidatos = [
        e for e in eventos if _para_buscar(titulo) in _para_buscar(e.titulo) and e.id is not None
    ]

    if not candidatos:
        return (
            f"No encontré ningún evento que se llame algo como eso el {fecha}. "
            "Sólo busco en tu calendario principal."
        )
    if len(candidatos) > 1:
        lista = "\n".join(f"- {_linea(e, _en_hora_local(e))}" for e in candidatos)
        return f"Hay varios eventos que coinciden ese día:\n{lista}\n¿Cuál de estos?"

    unico = candidatos[0]
    resumen = (
        f"Eliminar {_linea(unico, _en_hora_local(unico))} del {fecha_en_palabras(unico.inicio)}"
    )

    decision = interrupt({"resumen": resumen})
    if not (isinstance(decision, dict) and decision.get("aprobado") is True):
        return "La persona lo canceló. No se eliminó nada."

    assert unico.id is not None  # noqa: S101 - ya filtrado arriba; para mypy
    try:
        await calendario.eliminar_evento(usuario_id, unico.id)
    except (CuentaNoConectadaError, PermisoInsuficienteError) as exc:
        return exc.mensaje_usuario
    except AutorizacionFallidaError:
        return "Se venció el permiso del calendario. Escribí /conectar para renovarlo."
    except ServiceUnavailableError:
        return "No pude eliminar el evento ahora mismo. Probá de nuevo en un minuto."

    logger.info(
        "agente.herramienta.invocada",
        herramienta="eliminar_evento_del_calendario",
        usuario_id=str(usuario_id),
    )
    return "Evento eliminado."


def _momento_local(fecha: str, hora: str) -> datetime | None:
    """Convierte fecha y hora del modelo en un instante local, o None si no valen."""
    try:
        dia = date.fromisoformat(fecha)
        partes = hora.strip().split(":")
        hh, mm = int(partes[0]), int(partes[1])
        if len(partes) != 2 or not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
    except (ValueError, IndexError):
        return None
    return datetime(dia.year, dia.month, dia.day, hh, mm, tzinfo=ZONA_HORARIA)


def _para_buscar(texto: str) -> str:
    """Normaliza un título para el matching: minúsculas y sin tildes."""
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    ).strip()
