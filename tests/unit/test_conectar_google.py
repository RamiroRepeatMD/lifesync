"""Tests del caso de uso que conecta la cuenta de Google (PB-009).

El grupo que importa es el del `refresh_token`: Google lo emite una sola vez y
perderlo deja la conexión muerta hasta que la persona vuelva a autorizar a mano.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import structlog

from src.application.ports.autorizador_google import CredencialesGoogle
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import AutorizacionFallidaError, InvalidValueError
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from tests.dobles import AutorizadorFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
AHORA = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
GOOGLE = ProveedorOAuth.GOOGLE


def _caso(
    autorizador: AutorizadorFalso | None = None,
) -> tuple[ConectarGoogle, RepositorioOAuthTokenEnMemoria, AutorizadorFalso]:
    repo = RepositorioOAuthTokenEnMemoria()
    auth = autorizador or AutorizadorFalso(usuario_fijo=USUARIO)
    return ConectarGoogle(repo, auth), repo, auth


def _credenciales(refresh: str | None = "refresh-nuevo") -> CredencialesGoogle:
    return CredencialesGoogle(
        access_token="access-nuevo",
        refresh_token=refresh,
        expira_en=AHORA + timedelta(hours=1),
        scopes=("https://www.googleapis.com/auth/calendar.readonly",),
    )


# --- El camino feliz --------------------------------------------------------


async def test_guarda_las_credenciales_del_usuario_del_estado() -> None:
    caso, repo, _ = _caso()

    devuelto = await caso.completar("4/codigo", "state-valido", AHORA)

    assert devuelto == USUARIO
    token = await repo.obtener(USUARIO, GOOGLE)
    assert token is not None
    assert token.access_token == "access-de-prueba"


async def test_canjea_el_codigo_recibido() -> None:
    caso, _, auth = _caso()

    await caso.completar("4/codigo-particular", "state", AHORA)

    assert auth.codigos_canjeados == ["4/codigo-particular"]


async def test_esta_conectado_refleja_lo_guardado() -> None:
    caso, _, _ = _caso()

    assert await caso.esta_conectado(USUARIO) is False
    await caso.completar("4/codigo", "state", AHORA)
    assert await caso.esta_conectado(USUARIO) is True


# --- El refresh_token: el punto delicado del PB -----------------------------


async def test_no_pisa_el_refresh_token_guardado_cuando_google_no_lo_manda() -> None:
    """Google lo emite una sola vez. Escribirle None encima mata la conexión."""
    auth = AutorizadorFalso(usuario_fijo=USUARIO, credenciales=_credenciales(refresh="el-bueno"))
    caso, repo, _ = _caso(auth)
    await caso.completar("4/primero", "state", AHORA)

    # Segunda vuelta: Google ya no manda refresh_token.
    auth.credenciales = _credenciales(refresh=None)
    await caso.completar("4/segundo", "state", AHORA)

    token = await repo.obtener(USUARIO, GOOGLE)
    assert token is not None
    assert token.refresh_token == "el-bueno"


async def test_un_refresh_token_nuevo_si_reemplaza_al_viejo() -> None:
    """Conservar no es ignorar: si Google manda uno nuevo, ese vale."""
    auth = AutorizadorFalso(usuario_fijo=USUARIO, credenciales=_credenciales(refresh="viejo"))
    caso, repo, _ = _caso(auth)
    await caso.completar("4/primero", "state", AHORA)

    auth.credenciales = _credenciales(refresh="flamante")
    await caso.completar("4/segundo", "state", AHORA)

    token = await repo.obtener(USUARIO, GOOGLE)
    assert token is not None
    assert token.refresh_token == "flamante"


async def test_se_deja_rastro_de_que_se_conservo() -> None:
    auth = AutorizadorFalso(usuario_fijo=USUARIO, credenciales=_credenciales(refresh="el-bueno"))
    caso, _, _ = _caso(auth)
    await caso.completar("4/primero", "state", AHORA)
    auth.credenciales = _credenciales(refresh=None)

    with structlog.testing.capture_logs() as eventos:
        await caso.completar("4/segundo", "state", AHORA)

    assert any(e["event"] == "google.refresh_token_conservado" for e in eventos)


# --- El state inválido ------------------------------------------------------


async def test_un_state_invalido_no_guarda_nada() -> None:
    auth = AutorizadorFalso(usuario_fijo=USUARIO)
    auth.estado_invalido = InvalidValueError("state falsificado")
    caso, repo, _ = _caso(auth)

    with pytest.raises(InvalidValueError):
        await caso.completar("4/codigo", "state-falso", AHORA)

    assert await repo.obtener(USUARIO, GOOGLE) is None


async def test_un_state_invalido_ni_siquiera_gasta_el_codigo() -> None:
    """Verificar primero evita quemar un canje contra Google por nada."""
    auth = AutorizadorFalso(usuario_fijo=USUARIO)
    auth.estado_invalido = InvalidValueError("state falsificado")
    caso, _, _ = _caso(auth)

    with pytest.raises(InvalidValueError):
        await caso.completar("4/codigo", "state-falso", AHORA)

    assert auth.codigos_canjeados == []


async def test_si_google_rechaza_el_canje_no_se_guarda_nada() -> None:
    auth = AutorizadorFalso(usuario_fijo=USUARIO)
    auth.fallar_canje = AutorizacionFallidaError()
    caso, repo, _ = _caso(auth)

    with pytest.raises(AutorizacionFallidaError):
        await caso.completar("4/codigo", "state", AHORA)

    assert await repo.obtener(USUARIO, GOOGLE) is None


# --- Refresco perezoso -------------------------------------------------------


async def test_sin_conexion_devuelve_none() -> None:
    caso, _, _ = _caso()

    assert await caso.credencial_vigente(USUARIO, AHORA) is None


async def test_un_token_vigente_se_devuelve_tal_cual() -> None:
    caso, repo, auth = _caso()
    await repo.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=GOOGLE,
            access_token="todavia-sirve",
            refresh_token="refresh",
            expira_en=AHORA + timedelta(hours=1),
        )
    )

    token = await caso.credencial_vigente(USUARIO, AHORA)

    assert token is not None
    assert token.access_token == "todavia-sirve"
    assert auth.refresh_usados == []  # no se molestó a Google


async def test_un_token_vencido_se_renueva_solo() -> None:
    """Es lo que hace que la conexión siga viva pasada la hora."""
    caso, repo, auth = _caso()
    await repo.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=GOOGLE,
            access_token="vencido",
            refresh_token="el-refresh",
            expira_en=AHORA - timedelta(minutes=1),
        )
    )

    token = await caso.credencial_vigente(USUARIO, AHORA)

    assert token is not None
    assert token.access_token == "access-de-prueba"
    assert auth.refresh_usados == ["el-refresh"]


async def test_el_refresco_conserva_el_refresh_token() -> None:
    """Google no lo repite al renovar: perderlo acá sería perder la conexión."""
    auth = AutorizadorFalso(usuario_fijo=USUARIO, credenciales=_credenciales(refresh=None))
    caso, repo, _ = _caso(auth)
    await repo.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=GOOGLE,
            access_token="vencido",
            refresh_token="hay-que-conservarlo",
            expira_en=AHORA - timedelta(minutes=1),
        )
    )

    token = await caso.credencial_vigente(USUARIO, AHORA)

    assert token is not None
    assert token.refresh_token == "hay-que-conservarlo"


async def test_un_token_vencido_sin_refresh_no_se_puede_renovar() -> None:
    caso, repo, _ = _caso()
    await repo.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=GOOGLE,
            access_token="vencido",
            refresh_token=None,
            expira_en=AHORA - timedelta(minutes=1),
        )
    )

    with pytest.raises(AutorizacionFallidaError):
        await caso.credencial_vigente(USUARIO, AHORA)


async def test_si_la_persona_revoco_el_acceso_se_propaga() -> None:
    auth = AutorizadorFalso(usuario_fijo=USUARIO)
    auth.fallar_refresco = AutorizacionFallidaError()
    caso, repo, _ = _caso(auth)
    await repo.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=GOOGLE,
            access_token="vencido",
            refresh_token="revocado",
            expira_en=AHORA - timedelta(minutes=1),
        )
    )

    with pytest.raises(AutorizacionFallidaError):
        await caso.credencial_vigente(USUARIO, AHORA)


# --- Privacidad (RF-18) ------------------------------------------------------


async def test_no_se_loguea_ninguna_credencial() -> None:
    caso, _, _ = _caso()

    with structlog.testing.capture_logs() as eventos:
        await caso.completar("4/codigo-secreto", "state", AHORA)

    # Sin esto la aserción negativa de abajo pasaría aunque `capture_logs`
    # no hubiera capturado nada. Ver `test_logging.py`, sección de la trampa.
    assert eventos
    registrado = json.dumps(eventos, default=str)
    assert "4/codigo-secreto" not in registrado
    assert "access-de-prueba" not in registrado
    assert "refresh-de-prueba" not in registrado


# --- Desconexión (PB-011, RF-12) ---------------------------------------------


async def test_desconectar_revoca_primero_y_borra_despues() -> None:
    """El orden importa: revocar necesita el token, así que va antes del borrado."""
    caso, repo, auth = _caso()
    await caso.completar("4/codigo", "state", AHORA)

    revocado = await caso.desconectar(USUARIO)

    assert revocado is True
    assert auth.revocados == ["refresh-de-prueba"]  # se revoca el refresh
    assert await repo.obtener(USUARIO, GOOGLE) is None


async def test_desconectar_sin_cuenta_es_idempotente() -> None:
    """El estado final es el pedido: no había nada y no hay nada."""
    caso, _, auth = _caso()

    assert await caso.desconectar(USUARIO) is True
    assert auth.revocados == []  # no hubo qué revocar


async def test_si_la_revocacion_falla_el_token_se_borra_igual() -> None:
    """Conservarlo "para reintentar" sería quedarse una credencial que pidieron eliminar."""
    auth = AutorizadorFalso(usuario_fijo=USUARIO)
    auth.revocacion_exitosa = False
    caso, repo, _ = _caso(auth)
    await caso.completar("4/codigo", "state", AHORA)

    revocado = await caso.desconectar(USUARIO)

    assert revocado is False
    assert await repo.obtener(USUARIO, GOOGLE) is None
