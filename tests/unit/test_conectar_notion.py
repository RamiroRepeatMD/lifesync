"""Tests del caso de uso que conecta Notion (PB-014).

Lo delicado es `renovar`: Notion no informa cuándo vence el token (se renueva
ante un 401) y puede rotar el par. Dos renovaciones a la vez, o una que termina
después de un `/desconectar notion`, no se pueden pisar: por eso las pruebas de
concurrencia frenan el refresco falso con un evento, y sin el candado fallan.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import structlog

from src.application.ports.autorizador_notion import CredencialesNotion
from src.application.use_cases.conectar_notion import ConectarNotion
from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import (
    AutorizacionFallidaError,
    InvalidValueError,
    NotionNoConectadoError,
    RepositoryError,
)
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from tests.dobles import AutorizadorNotionFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
AHORA = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
NOTION = ProveedorOAuth.NOTION


class _RepoQueFalla(RepositorioOAuthTokenEnMemoria):
    """Guarda bien, salvo las próximas `fallas` veces: la base que parpadea."""

    def __init__(self) -> None:
        super().__init__()
        self.fallas = 0

    async def guardar(self, token: OAuthToken) -> OAuthToken:
        if self.fallas > 0:
            self.fallas -= 1
            raise RepositoryError("se cortó la base")
        return await super().guardar(token)


def _caso(
    repo: RepositorioOAuthTokenEnMemoria | None = None,
    auth: AutorizadorNotionFalso | None = None,
) -> tuple[ConectarNotion, RepositorioOAuthTokenEnMemoria, AutorizadorNotionFalso]:
    repo = repo if repo is not None else RepositorioOAuthTokenEnMemoria()
    auth = auth if auth is not None else AutorizadorNotionFalso(usuario_fijo=USUARIO)
    return ConectarNotion(repo, auth), repo, auth


async def _conectado(caso: ConectarNotion) -> OAuthToken:
    await caso.completar("codigo", "state", AHORA)
    return await caso.credencial(USUARIO)


async def _ceder() -> None:
    """Deja correr a las otras tareas hasta que todas queden esperando algo."""
    for _ in range(10):
        await asyncio.sleep(0)


# --- Conectar ---------------------------------------------------------------------


async def test_guarda_el_par_del_usuario_del_estado() -> None:
    caso, repo, auth = _caso()

    assert await caso.completar("codigo-1", "state", AHORA) == USUARIO

    token = await repo.obtener(USUARIO, NOTION)
    assert token is not None
    assert (token.access_token, token.refresh_token) == ("notion-access-1", "notion-refresh-1")
    assert token.scopes == ()  # Notion no tiene scopes: los permisos son de la conexión
    assert auth.codigos_canjeados == ["codigo-1"]


async def test_un_state_invalido_ni_siquiera_gasta_el_codigo() -> None:
    caso, repo, auth = _caso()
    auth.estado_invalido = InvalidValueError("falsificado")

    with pytest.raises(InvalidValueError):
        await caso.completar("codigo", "falso", AHORA)

    assert auth.codigos_canjeados == []
    assert await repo.obtener(USUARIO, NOTION) is None


async def test_si_notion_rechaza_el_canje_no_se_guarda_nada() -> None:
    caso, repo, auth = _caso()
    auth.fallar_canje = AutorizacionFallidaError("rechazado")

    with pytest.raises(AutorizacionFallidaError):
        await caso.completar("codigo", "state", AHORA)

    assert await repo.obtener(USUARIO, NOTION) is None


async def test_reconectar_no_conserva_el_refresh_del_consentimiento_anterior() -> None:
    """Una autorización nueva trae su propio par: el refresh viejo es de otra."""
    caso, repo, auth = _caso()
    await caso.completar("codigo", "state", AHORA)
    auth.credenciales = CredencialesNotion(access_token="otro-access", refresh_token=None)

    await caso.completar("codigo-2", "state", AHORA)

    token = await repo.obtener(USUARIO, NOTION)
    assert token is not None
    assert (token.access_token, token.refresh_token) == ("otro-access", None)


async def test_esta_conectado_y_la_credencial_reflejan_lo_guardado() -> None:
    caso, _, _ = _caso()

    assert await caso.esta_conectado(USUARIO) is False
    with pytest.raises(NotionNoConectadoError):
        await caso.credencial(USUARIO)

    await caso.completar("codigo", "state", AHORA)

    assert await caso.esta_conectado(USUARIO) is True
    assert (await caso.credencial(USUARIO)).access_token == "notion-access-1"


# --- Desconectar -------------------------------------------------------------------


async def test_desconectar_revoca_las_dos_credenciales_y_borra() -> None:
    caso, repo, auth = _caso()
    await caso.completar("codigo", "state", AHORA)

    assert await caso.desconectar(USUARIO) is True

    assert auth.revocados == ["notion-access-1", "notion-refresh-1"]
    assert await repo.obtener(USUARIO, NOTION) is None


async def test_si_la_revocacion_falla_se_borra_igual() -> None:
    """Quedarse el token "para reintentar" sería conservar lo que pidió borrar."""
    caso, repo, auth = _caso()
    await caso.completar("codigo", "state", AHORA)
    auth.revocacion_exitosa = False

    assert await caso.desconectar(USUARIO) is False
    assert await repo.obtener(USUARIO, NOTION) is None


async def test_desconectar_sin_conexion_no_revoca_nada() -> None:
    caso, _, auth = _caso()

    assert await caso.desconectar(USUARIO) is True
    assert auth.revocados == []


# --- Renovar ante un 401 --------------------------------------------------------


async def test_renovar_guarda_el_par_nuevo() -> None:
    caso, repo, auth = _caso()
    vencido = await _conectado(caso)

    renovado = await caso.renovar(USUARIO, vencido.access_token)

    assert auth.refresh_usados == ["notion-refresh-1"]
    assert (renovado.access_token, renovado.refresh_token) == (
        "notion-access-2",
        "notion-refresh-2",
    )
    assert await repo.obtener(USUARIO, NOTION) == renovado


async def test_un_refresh_vacio_no_pisa_el_guardado() -> None:
    """La lección de Google: escribir None sobre el único refresh mata la conexión."""
    caso, _, auth = _caso()
    vencido = await _conectado(caso)
    auth.refresh_nulo = True

    renovado = await caso.renovar(USUARIO, vencido.access_token)

    assert renovado.access_token == "notion-access-2"
    assert renovado.refresh_token == "notion-refresh-1"


async def test_si_otro_pedido_ya_lo_renovo_no_se_gasta_el_refresh() -> None:
    caso, _, auth = _caso()
    vencido = await _conectado(caso)
    primero = await caso.renovar(USUARIO, vencido.access_token)

    segundo = await caso.renovar(USUARIO, vencido.access_token)

    assert segundo == primero
    assert auth.refresh_usados == ["notion-refresh-1"]


async def test_dos_renovaciones_a_la_vez_hacen_un_solo_refresco() -> None:
    """Sin el candado, las dos verían el token vencido y gastarían el mismo refresh.

    Son dos instancias del caso de uso, como dos pedidos: el candado es del
    módulo, no de la instancia.
    """
    caso, repo, auth = _caso()
    vencido = await _conectado(caso)
    auth.esperar = asyncio.Event()

    tareas = [
        asyncio.create_task(ConectarNotion(repo, auth).renovar(USUARIO, vencido.access_token))
        for _ in range(2)
    ]
    await _ceder()
    auth.esperar.set()
    uno, otro = await asyncio.gather(*tareas)

    assert auth.refresh_usados == ["notion-refresh-1"]
    assert uno == otro


async def test_desconectar_en_medio_de_una_renovacion_no_la_revive() -> None:
    """Sin el candado, la renovación guardaría el par nuevo después del borrado."""
    caso, repo, auth = _caso()
    vencido = await _conectado(caso)
    auth.esperar = asyncio.Event()

    renovacion = asyncio.create_task(caso.renovar(USUARIO, vencido.access_token))
    await _ceder()
    desconexion = asyncio.create_task(ConectarNotion(repo, auth).desconectar(USUARIO))
    await _ceder()
    auth.esperar.set()
    await renovacion
    await desconexion

    assert await repo.obtener(USUARIO, NOTION) is None
    assert "notion-access-2" in auth.revocados  # se revocó el par nuevo, no sólo el viejo


async def test_con_la_conexion_borrada_no_se_revive() -> None:
    caso, repo, auth = _caso()
    vencido = await _conectado(caso)
    await repo.eliminar(USUARIO, NOTION)

    with pytest.raises(NotionNoConectadoError):
        await caso.renovar(USUARIO, vencido.access_token)

    assert auth.refresh_usados == []
    assert await repo.obtener(USUARIO, NOTION) is None


async def test_sin_refresh_no_se_puede_renovar() -> None:
    caso, _, auth = _caso()
    auth.credenciales = CredencialesNotion(access_token="solo-access", refresh_token=None)
    vencido = await _conectado(caso)

    with pytest.raises(AutorizacionFallidaError):
        await caso.renovar(USUARIO, vencido.access_token)


async def test_si_notion_rechaza_el_refresco_se_propaga() -> None:
    caso, _, auth = _caso()
    vencido = await _conectado(caso)
    auth.fallar_refresco = AutorizacionFallidaError("quitaron la conexión")

    with pytest.raises(AutorizacionFallidaError):
        await caso.renovar(USUARIO, vencido.access_token)


async def test_si_otro_contenedor_ya_renovo_vale_lo_que_guardo() -> None:
    """Durante un deploy conviven dos procesos, y el candado es de cada uno."""
    caso, repo, auth = _caso()
    vencido = await _conectado(caso)

    async def el_otro_contenedor_renueva() -> None:
        await repo.guardar(
            OAuthToken(
                usuario_id=USUARIO,
                proveedor=NOTION,
                access_token="del-otro",
                refresh_token="refresh-del-otro",
            )
        )

    auth.al_refrescar = el_otro_contenedor_renueva
    auth.fallar_refresco = AutorizacionFallidaError("ese refresh ya se usó")

    renovado = await caso.renovar(USUARIO, vencido.access_token)

    assert renovado.access_token == "del-otro"


async def test_si_el_guardado_falla_se_reintenta_una_vez() -> None:
    """Notion pudo invalidar el refresh viejo: perder el nuevo obligaría a reconectar."""
    repo = _RepoQueFalla()
    caso, _, _ = _caso(repo=repo)
    vencido = await _conectado(caso)
    repo.fallas = 1

    with structlog.testing.capture_logs() as eventos:
        renovado = await caso.renovar(USUARIO, vencido.access_token)

    assert renovado.access_token == "notion-access-2"
    assert await repo.obtener(USUARIO, NOTION) == renovado
    assert "notion.renovacion_sin_guardar" in [e["event"] for e in eventos]


async def test_si_el_guardado_falla_dos_veces_se_propaga() -> None:
    repo = _RepoQueFalla()
    caso, _, _ = _caso(repo=repo)
    vencido = await _conectado(caso)
    repo.fallas = 2

    with pytest.raises(RepositoryError):
        await caso.renovar(USUARIO, vencido.access_token)


# --- Privacidad (RF-18) -------------------------------------------------------


async def test_no_se_loguea_ninguna_credencial() -> None:
    caso, _, _ = _caso()

    with structlog.testing.capture_logs() as eventos:
        await caso.completar("codigo-secreto", "state", AHORA)
        await caso.renovar(USUARIO, "notion-access-1")
        await caso.desconectar(USUARIO)

    assert eventos
    registrado = json.dumps(eventos, default=str)
    for secreto in ("codigo-secreto", "notion-access-1", "notion-refresh-1", "notion-access-2"):
        assert secreto not in registrado
