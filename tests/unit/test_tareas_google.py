"""Tests del adaptador de Google Tasks (PB-028). Sin red: MockTransport."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest

from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.entities.tarea import Tarea
from src.domain.exceptions import (
    CuentaNoConectadaError,
    EntityNotFoundError,
    PermisoInsuficienteError,
)
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.external.google.tareas import TareasGoogle
from tests.dobles import AutorizadorFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
Manejador = Callable[[httpx.Request], httpx.Response]


async def _con_token(pedidos: list[httpx.Request], manejador: Manejador) -> TareasGoogle:
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
    return TareasGoogle(http, ConectarGoogle(tokens, AutorizadorFalso(usuario_fijo=USUARIO)))


def _items(*items: dict[str, Any]) -> Manejador:
    return lambda _: httpx.Response(200, json={"items": list(items)})


# --- Listar ------------------------------------------------------------------


async def test_lista_solo_pendientes_y_de_la_lista_default() -> None:
    pedidos: list[httpx.Request] = []
    tareas = await _con_token(pedidos, _items({"id": "t1", "title": "Comprar regalo"}))

    resultado = await tareas.pendientes(USUARIO)

    assert [t.titulo for t in resultado] == ["Comprar regalo"]
    url = urlparse(str(pedidos[0].url))
    assert url.path.endswith("/lists/@default/tasks")
    assert parse_qs(url.query)["showCompleted"] == ["false"]


async def test_ordena_por_vencimiento_y_las_sin_fecha_al_final() -> None:
    tareas = await _con_token(
        [],
        _items(
            {"id": "a", "title": "Sin fecha"},
            {"id": "b", "title": "Vence después", "due": "2026-09-20T00:00:00.000Z"},
            {"id": "c", "title": "Vence antes", "due": "2026-09-16T00:00:00.000Z"},
        ),
    )

    resultado = await tareas.pendientes(USUARIO)

    assert [t.titulo for t in resultado] == ["Vence antes", "Vence después", "Sin fecha"]


async def test_el_due_se_lee_como_fecha_sin_hora() -> None:
    """La API guarda sólo la fecha: leer más que eso sería inventar."""
    tareas = await _con_token(
        [], _items({"id": "a", "title": "X", "due": "2026-09-16T00:00:00.000Z"})
    )

    (tarea,) = await tareas.pendientes(USUARIO)

    assert tarea.vencimiento == date(2026, 9, 16)


async def test_sin_cuenta_conectada_avisa() -> None:
    tareas = TareasGoogle(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))),
        ConectarGoogle(RepositorioOAuthTokenEnMemoria(), AutorizadorFalso()),
    )

    with pytest.raises(CuentaNoConectadaError):
        await tareas.pendientes(USUARIO)


async def test_403_es_permiso_insuficiente() -> None:
    """Una cuenta conectada antes de PB-028 no tiene el scope de tareas."""
    tareas = await _con_token([], lambda _: httpx.Response(403, json={}))

    with pytest.raises(PermisoInsuficienteError):
        await tareas.pendientes(USUARIO)


# --- Crear -------------------------------------------------------------------


async def test_crear_postea_titulo_y_solo_fecha() -> None:
    pedidos: list[httpx.Request] = []
    tareas = await _con_token(pedidos, lambda _: httpx.Response(200, json={"id": "nueva"}))

    await tareas.crear(USUARIO, Tarea(titulo="Llamar al banco", vencimiento=date(2026, 9, 17)))

    assert pedidos[0].method == "POST"
    cuerpo = json.loads(pedidos[0].content)
    assert cuerpo["title"] == "Llamar al banco"
    assert cuerpo["due"] == "2026-09-17T00:00:00.000Z"  # medianoche: sin hora real


async def test_crear_sin_vencimiento_no_manda_due() -> None:
    pedidos: list[httpx.Request] = []
    tareas = await _con_token(pedidos, lambda _: httpx.Response(200, json={"id": "nueva"}))

    await tareas.crear(USUARIO, Tarea(titulo="Algún día"))

    assert "due" not in json.loads(pedidos[0].content)


async def test_crear_devuelve_la_tarea_con_su_id() -> None:
    tareas = await _con_token(
        [], lambda _: httpx.Response(200, json={"id": "id-nuevo", "title": "X"})
    )

    creada = await tareas.crear(USUARIO, Tarea(titulo="X"))

    assert creada.id == "id-nuevo"


# --- Completar ---------------------------------------------------------------


async def test_completar_patchea_el_estado() -> None:
    pedidos: list[httpx.Request] = []
    tareas = await _con_token(pedidos, lambda _: httpx.Response(200, json={}))

    await tareas.completar(USUARIO, "id-x")

    assert pedidos[0].method == "PATCH"
    assert str(pedidos[0].url).endswith("/lists/@default/tasks/id-x")
    assert json.loads(pedidos[0].content) == {"status": "completed"}


async def test_completar_algo_inexistente_es_error_con_nombre() -> None:
    tareas = await _con_token([], lambda _: httpx.Response(404, json={}))

    with pytest.raises(EntityNotFoundError):
        await tareas.completar(USUARIO, "fantasma")
