"""Tests del doble en memoria del puerto de tokens.

Validan que el contrato de `OAuthTokenRepository` es implementable sin base de
datos. El doble queda listo para los casos de uso de PB-009 y del Sprint 2,
que van a querer probar su lógica sin tocar Supabase.
"""

from __future__ import annotations

from uuid import uuid4

from src.domain.entities.oauth_token import OAuthToken
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from tests.dobles import RepositorioOAuthTokenEnMemoria

USUARIO_ID = uuid4()


def _token(proveedor: ProveedorOAuth = ProveedorOAuth.GOOGLE, **overrides: object) -> OAuthToken:
    campos: dict[str, object] = {
        "usuario_id": USUARIO_ID,
        "proveedor": proveedor,
        "access_token": "token-de-prueba",
    }
    campos.update(overrides)
    return OAuthToken(**campos)  # type: ignore[arg-type]


async def test_guardar_asigna_un_id() -> None:
    repo = RepositorioOAuthTokenEnMemoria()

    guardado = await repo.guardar(_token())

    assert guardado.id is not None


async def test_obtener_devuelve_lo_guardado() -> None:
    repo = RepositorioOAuthTokenEnMemoria()
    await repo.guardar(_token())

    recuperado = await repo.obtener(USUARIO_ID, ProveedorOAuth.GOOGLE)

    assert recuperado is not None
    assert recuperado.access_token == "token-de-prueba"


async def test_obtener_devuelve_none_para_un_proveedor_no_conectado() -> None:
    repo = RepositorioOAuthTokenEnMemoria()
    await repo.guardar(_token(ProveedorOAuth.GOOGLE))

    assert await repo.obtener(USUARIO_ID, ProveedorOAuth.NOTION) is None


async def test_guardar_dos_veces_el_mismo_proveedor_reemplaza() -> None:
    """Un usuario tiene a lo sumo un token vigente por integración."""
    repo = RepositorioOAuthTokenEnMemoria()
    await repo.guardar(_token(access_token="viejo"))
    await repo.guardar(_token(access_token="nuevo"))

    tokens = await repo.listar_por_usuario(USUARIO_ID)

    assert len(tokens) == 1
    assert tokens[0].access_token == "nuevo"


async def test_listar_devuelve_un_token_por_proveedor() -> None:
    repo = RepositorioOAuthTokenEnMemoria()
    await repo.guardar(_token(ProveedorOAuth.GOOGLE))
    await repo.guardar(_token(ProveedorOAuth.NOTION))

    tokens = await repo.listar_por_usuario(USUARIO_ID)

    assert {token.proveedor for token in tokens} == {
        ProveedorOAuth.GOOGLE,
        ProveedorOAuth.NOTION,
    }


async def test_listar_no_mezcla_usuarios() -> None:
    repo = RepositorioOAuthTokenEnMemoria()
    await repo.guardar(_token())
    await repo.guardar(_token(usuario_id=uuid4()))

    assert len(await repo.listar_por_usuario(USUARIO_ID)) == 1


async def test_eliminar_desconecta_la_integracion() -> None:
    repo = RepositorioOAuthTokenEnMemoria()
    await repo.guardar(_token())

    await repo.eliminar(USUARIO_ID, ProveedorOAuth.GOOGLE)

    assert await repo.obtener(USUARIO_ID, ProveedorOAuth.GOOGLE) is None


async def test_eliminar_lo_que_no_existe_es_idempotente() -> None:
    repo = RepositorioOAuthTokenEnMemoria()

    await repo.eliminar(USUARIO_ID, ProveedorOAuth.NOTION)  # no debe lanzar
