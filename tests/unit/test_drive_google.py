"""Tests del adaptador de Google Drive (PB-037). Sin red: MockTransport.

Los JSON respetan la forma real de `files.list`: sólo los campos pedidos en
`fields`, `modifiedTime` en RFC 3339 con milisegundos y `owners` con la marca
`me` del dueño que es la persona.
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
    PermisoInsuficienteError,
    ServiceUnavailableError,
)
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.external.google.drive import (
    CAMPOS,
    MAX_CARACTERES_DE_TEXTO,
    DriveGoogle,
)
from tests.dobles import AutorizadorFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
Manejador = Callable[[httpx.Request], httpx.Response]
PLANILLA = "application/vnd.google-apps.spreadsheet"


def _item(**cambios: Any) -> dict[str, Any]:
    datos: dict[str, Any] = {
        "id": "f1",
        "name": "Presupuesto 2026",
        "mimeType": PLANILLA,
        "modifiedTime": "2026-10-12T21:30:00.000Z",
        "webViewLink": "https://docs.google.com/spreadsheets/d/f1/edit",
        "owners": [{"displayName": "Vos", "me": True}],
    }
    datos.update(cambios)
    return datos


def _archivos(*items: Any) -> Manejador:
    return lambda _: httpx.Response(200, json={"files": list(items)})


async def _drive(pedidos: list[httpx.Request], manejador: Manejador) -> DriveGoogle:
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
    return DriveGoogle(http, ConectarGoogle(tokens, AutorizadorFalso(usuario_fijo=USUARIO)))


async def _parametros(
    texto: str = "",
    tipo: TipoDeArchivo | None = None,
    desde: datetime | None = None,
    cantidad: int = 5,
) -> dict[str, list[str]]:
    """Busca con un Drive vacío y devuelve los parámetros del pedido."""
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _archivos())
    await drive.buscar(USUARIO, texto, tipo, desde, cantidad)
    [pedido] = pedidos
    return parse_qs(urlparse(str(pedido.url)).query)


def _con_texto(literal: str) -> str:
    return f"trashed = false and (name contains '{literal}' or fullText contains '{literal}')"


# --- La consulta -----------------------------------------------------------------


async def test_sin_texto_lista_los_ultimos_modificados() -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _archivos())

    await drive.buscar(USUARIO, "", None, None, 5)

    [pedido] = pedidos
    assert pedido.url.path == "/drive/v3/files"
    assert pedido.headers["Authorization"] == "Bearer access-vigente"
    parametros = parse_qs(urlparse(str(pedido.url)).query)
    assert parametros["q"] == ["trashed = false"]  # nunca lo que está en la papelera
    assert parametros["orderBy"] == ["modifiedTime desc"]
    assert parametros["pageSize"] == ["5"]
    assert parametros["fields"] == [CAMPOS]


async def test_con_texto_busca_en_el_nombre_y_en_el_contenido_sin_ordenar() -> None:
    """Por relevancia: es lo que sirve para "¿dónde está…?"."""
    parametros = await _parametros("presupuesto")

    assert parametros["q"] == [_con_texto("presupuesto")]
    assert "orderBy" not in parametros


async def test_un_texto_de_puros_espacios_es_listar() -> None:
    parametros = await _parametros(" \n\t ")

    assert parametros["q"] == ["trashed = false"]
    assert parametros["orderBy"] == ["modifiedTime desc"]


async def test_cada_tipo_suma_los_formatos_de_google_y_de_office() -> None:
    """ "La planilla" puede ser una de Google o un Excel subido."""
    parametros = await _parametros(tipo=TipoDeArchivo.PLANILLA)

    assert parametros["q"] == [
        "trashed = false and ("
        f"mimeType = '{PLANILLA}' or "
        "mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or "
        "mimeType = 'application/vnd.ms-excel')"
    ]


async def test_las_imagenes_son_cualquier_formato_de_imagen() -> None:
    parametros = await _parametros(tipo=TipoDeArchivo.IMAGEN)

    assert parametros["q"] == ["trashed = false and mimeType contains 'image/'"]


async def test_desde_viaja_en_utc() -> None:
    """La medianoche del 12/10 en Buenos Aires son las 3 de la mañana en UTC."""
    desde = datetime(2026, 10, 12, tzinfo=ZONA_HORARIA)

    parametros = await _parametros(desde=desde)

    assert parametros["q"] == ["trashed = false and modifiedTime >= '2026-10-12T03:00:00Z'"]


async def test_las_condiciones_se_combinan() -> None:
    desde = datetime(2026, 10, 12, tzinfo=ZONA_HORARIA)

    parametros = await _parametros("redes", TipoDeArchivo.PDF, desde)

    assert parametros["q"] == [
        _con_texto("redes")
        + " and (mimeType = 'application/pdf')"
        + " and modifiedTime >= '2026-10-12T03:00:00Z'"
    ]


@pytest.mark.parametrize(("pedida", "enviada"), [(50, "10"), (0, "1"), (-3, "1")])
async def test_la_cantidad_se_acota(pedida: int, enviada: str) -> None:
    parametros = await _parametros(cantidad=pedida)

    assert parametros["pageSize"] == [enviada]


# --- El texto entra sólo como literal ---------------------------------------------


async def test_un_texto_con_comillas_no_agrega_condiciones() -> None:
    """La inyección: cerrar el literal y sumar `or trashed = true`."""
    parametros = await _parametros("x' or trashed = true or name contains '")

    literal = "x\\' or trashed = true or name contains \\'"
    assert parametros["q"] == [_con_texto(literal)]


async def test_la_barra_se_escapa_antes_que_la_comilla() -> None:
    """Al revés, la barra que protege la comilla se duplicaría y la dejaría suelta."""
    parametros = await _parametros("a\\b'c")

    assert parametros["q"] == [_con_texto("a\\\\b\\'c")]


async def test_sin_caracteres_de_control_y_en_una_linea() -> None:
    parametros = await _parametros("  presu\npuesto\x00\t2026  ")

    assert parametros["q"] == [_con_texto("presu puesto 2026")]


async def test_un_texto_enorme_se_recorta() -> None:
    parametros = await _parametros("a" * 500)

    assert parametros["q"] == [_con_texto("a" * MAX_CARACTERES_DE_TEXTO)]


# --- De JSON a entidad -----------------------------------------------------------


async def test_cada_archivo_trae_lo_que_hay_que_mostrar() -> None:
    drive = await _drive([], _archivos(_item()))

    [archivo] = await drive.buscar(USUARIO, "", None, None, 5)

    assert archivo.id == "f1"
    assert archivo.nombre == "Presupuesto 2026"
    assert archivo.tipo is TipoDeArchivo.PLANILLA
    assert archivo.modificado == datetime(2026, 10, 12, 21, 30, tzinfo=UTC)
    assert archivo.enlace == "https://docs.google.com/spreadsheets/d/f1/edit"
    assert archivo.propietario is None  # es suyo


async def test_un_archivo_ajeno_dice_quien_lo_compartio() -> None:
    drive = await _drive([], _archivos(_item(owners=[{"displayName": "Ana Gómez", "me": False}])))

    [archivo] = await drive.buscar(USUARIO, "", None, None, 5)

    assert archivo.propietario == "Ana Gómez"


@pytest.mark.parametrize(
    ("formato", "tipo"),
    [
        ("application/vnd.google-apps.document", TipoDeArchivo.DOCUMENTO),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            TipoDeArchivo.DOCUMENTO,
        ),
        ("application/vnd.ms-excel", TipoDeArchivo.PLANILLA),
        ("application/vnd.google-apps.presentation", TipoDeArchivo.PRESENTACION),
        ("application/pdf", TipoDeArchivo.PDF),
        ("image/jpeg", TipoDeArchivo.IMAGEN),
        ("application/vnd.google-apps.folder", TipoDeArchivo.CARPETA),
        ("text/plain", None),
    ],
)
async def test_el_formato_se_traduce_a_un_tipo(formato: str, tipo: TipoDeArchivo | None) -> None:
    drive = await _drive([], _archivos(_item(mimeType=formato)))

    [archivo] = await drive.buscar(USUARIO, "", None, None, 5)

    assert archivo.tipo is tipo


async def test_lo_que_no_es_un_archivo_valido_se_descarta() -> None:
    drive = await _drive(
        [],
        _archivos(
            _item(id="bueno"),
            _item(id=None),
            _item(id="sin-fecha", modifiedTime=None),
            _item(id="fecha-rota", modifiedTime="ayer"),
            "no soy un archivo",
        ),
    )

    archivos = await drive.buscar(USUARIO, "", None, None, 5)

    assert [a.id for a in archivos] == ["bueno"]


async def test_sin_link_ni_duenos_no_inventa() -> None:
    sin_datos = _item()
    del sin_datos["webViewLink"], sin_datos["owners"]
    drive = await _drive([], _archivos(sin_datos))

    [archivo] = await drive.buscar(USUARIO, "", None, None, 5)

    assert archivo.enlace is None
    assert archivo.propietario is None


async def test_los_nombres_no_se_loguean() -> None:
    """RF-18: el log dice cuántos, nunca cuáles ni de quién."""
    ajeno = _item(name="Resultados de biopsia", owners=[{"displayName": "Clínica", "me": False}])
    drive = await _drive([], _archivos(ajeno))

    with structlog.testing.capture_logs() as eventos:
        await drive.buscar(USUARIO, "biopsia", None, None, 5)

    assert {"event": "drive.buscados", "cantidad": 1}.items() <= eventos[-1].items()
    registrado = json.dumps(eventos, default=str)
    assert "biopsia" not in registrado
    assert "Clínica" not in registrado


# --- Errores -----------------------------------------------------------------------


async def test_sin_cuenta_conectada_avisa() -> None:
    drive = DriveGoogle(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))),
        ConectarGoogle(RepositorioOAuthTokenEnMemoria(), AutorizadorFalso()),
    )

    with pytest.raises(CuentaNoConectadaError):
        await drive.buscar(USUARIO, "", None, None, 5)


async def test_api_deshabilitada_no_manda_a_reconectar() -> None:
    """La lección de Tasks: el permiso solo no alcanza, hay que habilitar la API."""
    cuerpo = {
        "error": {
            "code": 403,
            "message": "Google Drive API has not been used in project X before or it is disabled.",
            "details": [{"reason": "SERVICE_DISABLED"}],
        }
    }
    drive = await _drive([], lambda _: httpx.Response(403, json=cuerpo))

    with pytest.raises(ServiceUnavailableError):
        await drive.buscar(USUARIO, "", None, None, 5)


async def test_sin_el_permiso_de_drive_es_permiso_insuficiente() -> None:
    """Una cuenta conectada antes de PB-037 no tiene drive.metadata.readonly."""
    cuerpo = {"error": {"code": 403, "errors": [{"reason": "insufficientPermissions"}]}}
    drive = await _drive([], lambda _: httpx.Response(403, json=cuerpo))

    with pytest.raises(PermisoInsuficienteError):
        await drive.buscar(USUARIO, "", None, None, 5)


async def test_una_credencial_rechazada_pide_reconectar() -> None:
    drive = await _drive([], lambda _: httpx.Response(401, json={}))

    with pytest.raises(AutorizacionFallidaError):
        await drive.buscar(USUARIO, "", None, None, 5)


async def test_sin_red_es_servicio_no_disponible() -> None:
    def caido(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=pedido)

    drive = await _drive([], caido)

    with pytest.raises(ServiceUnavailableError):
        await drive.buscar(USUARIO, "", None, None, 5)
