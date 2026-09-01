"""Tests del adaptador de lectura de Google Calendar (PB-015).

Se usa `httpx.MockTransport`, igual que en el cliente de WhatsApp y en el de
OAuth: deja ejercitar el `AsyncClient` real e inspeccionar las peticiones que
habrían salido, que es donde viven las trampas de esta API.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
import structlog

from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import (
    AutorizacionFallidaError,
    CuentaNoConectadaError,
    ServiceUnavailableError,
)
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.external.google.calendario import CalendarioGoogle
from tests.dobles import AutorizadorFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
DESDE = datetime(2026, 9, 1, tzinfo=UTC)
HASTA = DESDE + timedelta(days=1)

Manejador = Callable[[httpx.Request], httpx.Response]

# Los cuatro calendarios de una cuenta real, en su forma exacta.
CALENDARIOS_REALES = {
    "items": [
        {"id": "yo@gmail.com", "summary": "Principal", "primary": True, "selected": True},
        {
            "id": "es.ar#holiday@group.v.calendar.google.com",
            "summary": "Feriados de Argentina",
            "selected": True,
        },
        {"id": "otro@group.calendar.google.com", "summary": "Facultad", "selected": True},
        {"id": "oculto@group.calendar.google.com", "summary": "Viejo", "selected": False},
    ]
}


async def _con_token(pedidos: list[httpx.Request], manejador: Manejador) -> CalendarioGoogle:
    """Arma el adaptador con un usuario ya conectado y transporte simulado."""
    tokens = RepositorioOAuthTokenEnMemoria()
    await tokens.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=ProveedorOAuth.GOOGLE,
            access_token="access-vigente",
            refresh_token="refresh",
            expira_en=datetime.now(UTC) + timedelta(hours=1),
        )
    )

    def interceptar(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        return manejador(pedido)

    http = httpx.AsyncClient(transport=httpx.MockTransport(interceptar))
    return CalendarioGoogle(http, ConectarGoogle(tokens, AutorizadorFalso(usuario_fijo=USUARIO)))


def _respondedor(
    calendarios: dict[str, Any] | None = None,
    eventos_por_calendario: dict[str, list[dict[str, Any]]] | None = None,
) -> Manejador:
    """Contesta la lista de calendarios y los eventos de cada uno."""
    lista = calendarios if calendarios is not None else CALENDARIOS_REALES
    eventos = eventos_por_calendario or {}

    def responder(pedido: httpx.Request) -> httpx.Response:
        if "calendarList" in pedido.url.path:
            return httpx.Response(200, json=lista)
        # /calendars/<id>/events
        cal_id = pedido.url.path.split("/calendars/")[1].split("/events")[0]
        from urllib.parse import unquote

        return httpx.Response(200, json={"items": eventos.get(unquote(cal_id), [])})

    return responder


def _evento(titulo: str, hora: int = 10, **extra: Any) -> dict[str, Any]:
    return {
        "summary": titulo,
        "start": {"dateTime": f"2026-09-01T{hora:02d}:00:00-03:00"},
        "end": {"dateTime": f"2026-09-01T{hora + 1:02d}:00:00-03:00"},
        **extra,
    }


# --- Qué calendarios se leen -------------------------------------------------


async def test_descarta_los_feriados_y_lee_los_propios() -> None:
    """Los feriados los agrega Google, no los agendó la persona."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, _respondedor())

    await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    consultados = [
        p.url.path.split("/calendars/")[1].split("/events")[0]
        for p in pedidos
        if "/events" in p.url.path
    ]
    assert len(consultados) == 2
    assert not any("holiday" in c for c in consultados)


async def test_respeta_los_calendarios_que_la_persona_escondio() -> None:
    """`selected: false` significa que no lo quiere ver ni en su propia UI."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, _respondedor())

    await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    urls = " ".join(str(p.url) for p in pedidos)
    assert "oculto" not in urls


async def test_los_calendarios_se_consultan_en_paralelo() -> None:
    """En serie, N calendarios serían N veces la latencia (RNF de 3 s)."""
    import asyncio

    en_vuelo = 0
    maximo = 0

    async def contar() -> None:
        nonlocal en_vuelo, maximo
        en_vuelo += 1
        maximo = max(maximo, en_vuelo)
        await asyncio.sleep(0.01)
        en_vuelo -= 1

    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, _respondedor())
    original = calendario._eventos_de

    async def espiado(*args: Any, **kwargs: Any) -> Any:
        await contar()
        return await original(*args, **kwargs)

    calendario._eventos_de = espiado  # type: ignore[method-assign]
    await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    assert maximo == 2  # los dos calendarios a la vez, no uno tras otro


# --- La query que sale -------------------------------------------------------


def _params_de_eventos(pedidos: list[httpx.Request]) -> dict[str, list[str]]:
    evento = next(p for p in pedidos if "/events" in p.url.path)
    return parse_qs(urlparse(str(evento.url)).query)


async def test_pide_las_ocurrencias_y_no_las_reglas_de_recurrencia() -> None:
    """Sin singleEvents, una reunión semanal no aparece en "¿qué tengo hoy?"."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, _respondedor())

    await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    assert _params_de_eventos(pedidos)["singleEvents"] == ["true"]


async def test_el_rango_viaja_con_zona_horaria() -> None:
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, _respondedor())

    await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    params = _params_de_eventos(pedidos)
    assert "+00:00" in params["timeMin"][0] or params["timeMin"][0].endswith("Z")


# --- Normalización de eventos -----------------------------------------------


async def test_lee_los_eventos_con_horario() -> None:
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos, _respondedor(eventos_por_calendario={"yo@gmail.com": [_evento("Dentista")]})
    )

    eventos = await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    assert [e.titulo for e in eventos] == ["Dentista"]
    assert eventos[0].todo_el_dia is False
    assert eventos[0].inicio.tzinfo is not None


async def test_lee_los_eventos_de_dia_completo() -> None:
    """Traen `date` en vez de `dateTime`: leer sólo dateTime los saltea callado."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos,
        _respondedor(
            eventos_por_calendario={
                "yo@gmail.com": [{"summary": "Vacaciones", "start": {"date": "2026-09-01"}}]
            }
        ),
    )

    eventos = await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    assert len(eventos) == 1
    assert eventos[0].todo_el_dia is True


async def test_mezcla_y_ordena_los_de_varios_calendarios() -> None:
    """Cada calendario ordena lo suyo; el orden global hay que hacerlo en casa."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos,
        _respondedor(
            eventos_por_calendario={
                "yo@gmail.com": [_evento("Tarde", hora=17)],
                "otro@group.calendar.google.com": [_evento("Temprano", hora=8)],
            }
        ),
    )

    eventos = await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    assert [e.titulo for e in eventos] == ["Temprano", "Tarde"]


async def test_cada_evento_sabe_de_que_calendario_salio() -> None:
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos,
        _respondedor(
            eventos_por_calendario={"otro@group.calendar.google.com": [_evento("Parcial")]}
        ),
    )

    eventos = await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    assert eventos[0].calendario == "Facultad"


async def test_un_evento_deforme_no_tira_toda_la_consulta() -> None:
    """Un item sin `start` no puede hacer que la persona se quede sin agenda."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos,
        _respondedor(
            eventos_por_calendario={"yo@gmail.com": [{"summary": "Roto"}, _evento("Sano")]}
        ),
    )

    eventos = await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    assert [e.titulo for e in eventos] == ["Sano"]


# --- Caché de la lista de calendarios ---------------------------------------


async def test_la_lista_de_calendarios_se_cachea() -> None:
    """Es un viaje a Google menos por pregunta, sobre un presupuesto ajustado."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, _respondedor())

    await calendario.eventos_entre(USUARIO, DESDE, HASTA)
    await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    listados = [p for p in pedidos if "calendarList" in p.url.path]
    assert len(listados) == 1


# --- Errores -----------------------------------------------------------------


async def test_sin_cuenta_conectada_avisa_que_falta_conectar() -> None:
    tokens = RepositorioOAuthTokenEnMemoria()
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    calendario = CalendarioGoogle(http, ConectarGoogle(tokens, AutorizadorFalso()))

    with pytest.raises(CuentaNoConectadaError):
        await calendario.eventos_entre(USUARIO, DESDE, HASTA)


async def test_un_401_pide_volver_a_conectar() -> None:
    """Pasa cuando la persona revoca el acceso desde su cuenta de Google."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, lambda _: httpx.Response(401, json={}))

    with pytest.raises(AutorizacionFallidaError):
        await calendario.eventos_entre(USUARIO, DESDE, HASTA)


async def test_un_error_de_google_no_se_confunde_con_uno_nuestro() -> None:
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, lambda _: httpx.Response(500, json={}))

    with pytest.raises(ServiceUnavailableError):
        await calendario.eventos_entre(USUARIO, DESDE, HASTA)


async def test_un_error_de_red_es_servicio_no_disponible() -> None:
    def caerse(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin conexión", request=pedido)

    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, caerse)

    with pytest.raises(ServiceUnavailableError):
        await calendario.eventos_entre(USUARIO, DESDE, HASTA)


async def test_una_cuenta_sin_calendarios_utiles_devuelve_vacio() -> None:
    """Sólo feriados: no es un error, es que no hay nada que leer."""
    solo_feriados = {
        "items": [{"id": "es.ar#holiday@group.v.calendar.google.com", "selected": True}]
    }
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, _respondedor(calendarios=solo_feriados))

    assert await calendario.eventos_entre(USUARIO, DESDE, HASTA) == ()


# --- Privacidad (RF-18) ------------------------------------------------------


async def test_no_se_loguea_ningun_titulo() -> None:
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos,
        _respondedor(eventos_por_calendario={"yo@gmail.com": [_evento("Análisis clínicos")]}),
    )

    with structlog.testing.capture_logs() as eventos:
        await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "Análisis" not in registrado
    assert "clínicos" not in registrado


async def test_si_se_loguea_cuantos_calendarios_y_cuantos_eventos() -> None:
    """Sin contadores no hay forma de diagnosticar "no me trae nada"."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos, _respondedor(eventos_por_calendario={"yo@gmail.com": [_evento("X")]})
    )

    with structlog.testing.capture_logs() as capturados:
        await calendario.eventos_entre(USUARIO, DESDE, HASTA)

    leidos = next(e for e in capturados if e["event"] == "calendar.eventos_leidos")
    assert leidos["calendarios"] == 2
    assert leidos["eventos"] == 1


# --- Escritura (PB-016) ------------------------------------------------------
#
# Estos métodos no piden confirmación: eso es del grafo (RF-08). Acá se prueba
# la mecánica HTTP y la traducción de errores.


async def test_crear_postea_al_calendario_principal() -> None:
    pedidos: list[httpx.Request] = []

    def responder(pedido: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "id-nuevo",
                "summary": "Dentista",
                "start": {"dateTime": "2026-09-05T10:00:00-03:00"},
                "end": {"dateTime": "2026-09-05T11:00:00-03:00"},
            },
        )

    calendario = await _con_token(pedidos, responder)
    from src.domain.entities.evento import Evento

    inicio = datetime(2026, 9, 5, 13, 0, tzinfo=UTC)
    creado = await calendario.crear_evento(
        USUARIO, Evento(titulo="Dentista", inicio=inicio, fin=inicio + timedelta(hours=1))
    )

    assert pedidos[0].method == "POST"
    assert "/calendars/primary/events" in str(pedidos[0].url)
    cuerpo = json.loads(pedidos[0].content)
    assert cuerpo["summary"] == "Dentista"
    assert "+00:00" in cuerpo["start"]["dateTime"] or cuerpo["start"]["dateTime"].endswith("Z")
    assert creado.id == "id-nuevo"


async def test_eliminar_manda_el_delete_correcto() -> None:
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, lambda _: httpx.Response(204))

    await calendario.eliminar_evento(USUARIO, "id-a-borrar")

    assert pedidos[0].method == "DELETE"
    assert str(pedidos[0].url).endswith("/calendars/primary/events/id-a-borrar")


@pytest.mark.parametrize("codigo", [404, 410], ids=["not_found", "gone"])
async def test_eliminar_algo_que_ya_no_existe_no_es_un_error(codigo: int) -> None:
    """Idempotente: el estado final es el mismo que se pedía."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, lambda _: httpx.Response(codigo, json={}))

    await calendario.eliminar_evento(USUARIO, "id-fantasma")  # no lanza


async def test_un_403_al_escribir_pide_reconectar() -> None:
    """El caso real: cuenta conectada antes de PB-016, con scope de sólo lectura."""
    from src.domain.entities.evento import Evento
    from src.domain.exceptions import PermisoInsuficienteError

    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos, lambda _: httpx.Response(403, json={"error": {"message": "insufficient"}})
    )

    inicio = datetime(2026, 9, 5, 13, 0, tzinfo=UTC)
    with pytest.raises(PermisoInsuficienteError) as capturado:
        await calendario.crear_evento(USUARIO, Evento(titulo="X", inicio=inicio))

    assert "/conectar" in capturado.value.mensaje_usuario


async def test_un_401_al_escribir_es_credencial_rechazada() -> None:
    from src.domain.entities.evento import Evento

    pedidos: list[httpx.Request] = []
    calendario = await _con_token(pedidos, lambda _: httpx.Response(401, json={}))

    inicio = datetime(2026, 9, 5, 13, 0, tzinfo=UTC)
    with pytest.raises(AutorizacionFallidaError):
        await calendario.crear_evento(USUARIO, Evento(titulo="X", inicio=inicio))


async def test_eventos_del_principal_solo_consulta_primary() -> None:
    """La búsqueda previa a borrar mira el mismo lugar donde se va a borrar."""
    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos, _respondedor(eventos_por_calendario={"primary": [_evento("Turno")]})
    )

    eventos = await calendario.eventos_del_principal(USUARIO, DESDE, HASTA)

    urls = [str(p.url) for p in pedidos]
    assert all("/calendars/primary/events" in u for u in urls)
    assert "calendarList" not in " ".join(urls)  # ni siquiera lista calendarios
    assert [e.titulo for e in eventos] == ["Turno"]


async def test_al_crear_no_se_loguea_el_titulo() -> None:
    from src.domain.entities.evento import Evento

    pedidos: list[httpx.Request] = []
    calendario = await _con_token(
        pedidos,
        lambda _: httpx.Response(
            200,
            json={
                "id": "x",
                "summary": "Cita médica privada",
                "start": {"dateTime": "2026-09-05T10:00:00-03:00"},
            },
        ),
    )

    inicio = datetime(2026, 9, 5, 13, 0, tzinfo=UTC)
    with structlog.testing.capture_logs() as eventos:
        await calendario.crear_evento(USUARIO, Evento(titulo="Cita médica privada", inicio=inicio))

    assert eventos
    assert "Cita médica" not in json.dumps(eventos, default=str)
