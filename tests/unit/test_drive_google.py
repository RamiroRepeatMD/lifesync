"""Tests del adaptador de Google Drive (PB-037 · PB-038 · PB-039). Sin red: MockTransport.

Los JSON respetan la forma real de `files.list`: sólo los campos pedidos en
`fields`, `modifiedTime` en RFC 3339 con milisegundos y `owners` con la marca
`me` del dueño que es la persona.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
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
    DriveLlenoError,
    DriveRechazoError,
    EntityNotFoundError,
    PermisoInsuficienteError,
    ServiceUnavailableError,
)
from src.domain.value_objects.permiso_de_archivo import PermisoDeArchivo
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.external.google.drive import (
    CAMPOS,
    CAMPOS_DE_ARCHIVO,
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


# --- Un archivo y las carpetas (PB-038) -------------------------------------------------


async def test_obtener_lee_un_archivo_con_los_campos_sin_envoltorio() -> None:
    """`files(...)` sólo lo acepta la búsqueda: un get con eso da 400."""
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, lambda _: httpx.Response(200, json=_item()))

    archivo = await drive.obtener(USUARIO, "f1")

    [pedido] = pedidos
    assert pedido.url.path == "/drive/v3/files/f1"
    assert parse_qs(urlparse(str(pedido.url)).query)["fields"] == [CAMPOS_DE_ARCHIVO]
    assert archivo.nombre == "Presupuesto 2026"


@pytest.mark.parametrize("archivo_id", ["../about", "f1/permissions", "", "x" * 200])
async def test_obtener_con_un_id_que_no_es_de_drive_no_genera_pedido(archivo_id: str) -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, lambda _: httpx.Response(200, json=_item()))

    with pytest.raises(EntityNotFoundError):
        await drive.obtener(USUARIO, archivo_id)

    assert pedidos == []


async def test_obtener_uno_que_no_existe() -> None:
    drive = await _drive([], lambda _: httpx.Response(404, json={}))

    with pytest.raises(EntityNotFoundError):
        await drive.obtener(USUARIO, "borrado")


async def test_las_carpetas_son_solo_las_propias() -> None:
    """Subir a una carpeta que te compartieron le daría el archivo a su dueño."""
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _archivos())

    await drive.carpetas(USUARIO, "Facu'ltad")

    q = parse_qs(urlparse(str(pedidos[0].url)).query)["q"][0]
    assert "mimeType = 'application/vnd.google-apps.folder'" in q
    assert "'me' in owners" in q
    assert "name contains 'Facu\\'ltad'" in q  # el literal escapado de siempre
    assert "trashed = false" in q


# --- Subir (PB-038) ---------------------------------------------------------------------

SESION = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=abc"


def _subida(location: str = SESION, abrir: int = 200, cuerpo: Any = None) -> Manejador:
    """Abre la sesión (POST) y recibe los bytes (PUT), como la API reanudable."""

    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.method == "POST":
            if abrir != 200:
                return httpx.Response(abrir, json=cuerpo or {})
            return httpx.Response(200, headers={"Location": location})
        return httpx.Response(200, json=_item(id="nuevo", name="informe.pdf"))

    return responder


async def test_subir_abre_la_sesion_y_manda_los_bytes() -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _subida())

    archivo = await drive.subir(USUARIO, "informe.pdf", "application/pdf", b"%PDF", "carpeta1")

    abrir, mandar = pedidos
    assert abrir.url.path == "/upload/drive/v3/files"
    parametros = parse_qs(urlparse(str(abrir.url)).query)
    assert parametros["uploadType"] == ["resumable"]
    assert parametros["fields"] == [CAMPOS_DE_ARCHIVO]
    assert json.loads(abrir.content) == {
        "name": "informe.pdf",
        "mimeType": "application/pdf",  # tal cual: sin convertir a Docs
        "parents": ["carpeta1"],
    }
    assert abrir.headers["X-Upload-Content-Length"] == "4"
    assert mandar.method == "PUT"
    assert str(mandar.url) == SESION
    assert mandar.content == b"%PDF"
    assert archivo.id == "nuevo"


async def test_subir_sin_carpeta_va_a_mi_unidad() -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _subida())

    await drive.subir(USUARIO, "foto.jpg", "image/jpeg", b"x", None)

    assert "parents" not in json.loads(pedidos[0].content)


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example/upload/drive/v3/files?upload_id=abc",
        "http://www.googleapis.com/upload/drive/v3/files?upload_id=abc",
        "https://www.googleapis.com.evil.example/upload?upload_id=abc",
        "",
    ],
)
async def test_una_sesion_fuera_de_google_no_recibe_los_bytes(location: str) -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _subida(location=location))

    with pytest.raises(ServiceUnavailableError):
        await drive.subir(USUARIO, "informe.pdf", "application/pdf", b"%PDF", None)

    assert [p.method for p in pedidos] == ["POST"]


async def test_si_se_corta_al_mandar_los_bytes_es_servicio_no_disponible() -> None:
    """Pudo haber llegado: quien lo cuenta aclara que se fije en Drive."""

    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.method == "POST":
            return httpx.Response(200, headers={"Location": SESION})
        raise httpx.ReadTimeout("se cortó", request=pedido)

    drive = await _drive([], responder)

    with pytest.raises(ServiceUnavailableError):
        await drive.subir(USUARIO, "informe.pdf", "application/pdf", b"%PDF", None)


async def test_con_el_drive_lleno_lo_dice() -> None:
    cuerpo = {"error": {"code": 403, "errors": [{"reason": "storageQuotaExceeded"}]}}
    drive = await _drive([], _subida(abrir=403, cuerpo=cuerpo))

    with pytest.raises(DriveLlenoError):
        await drive.subir(USUARIO, "informe.pdf", "application/pdf", b"%PDF", None)


# --- Compartir (PB-038) -----------------------------------------------------------------


async def test_compartir_con_una_persona_le_avisa_google() -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, lambda _: httpx.Response(200, json={"id": "p1"}))

    await drive.compartir(USUARIO, "f1", PermisoDeArchivo.EDITOR, "ana@ejemplo.com")

    [pedido] = pedidos
    assert pedido.method == "POST"
    assert pedido.url.path == "/drive/v3/files/f1/permissions"
    assert json.loads(pedido.content) == {
        "type": "user",
        "role": "writer",
        "emailAddress": "ana@ejemplo.com",
    }
    # El aviso por correo queda en su valor por defecto: sin aviso, Drive
    # rechaza a quien no tiene cuenta de Google.
    assert "sendNotificationEmail" not in parse_qs(urlparse(str(pedido.url)).query)


async def test_compartir_por_link_es_cualquiera_que_lo_tenga() -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, lambda _: httpx.Response(200, json={"id": "p1"}))

    await drive.compartir(USUARIO, "f1", PermisoDeArchivo.LECTOR, None)

    assert json.loads(pedidos[0].content) == {"type": "anyone", "role": "reader"}


@pytest.mark.parametrize(
    ("status", "motivo", "esperado"),
    [
        (403, "insufficientFilePermissions", DriveRechazoError),
        (400, "invalidSharingRequest", DriveRechazoError),
        (403, "shareOutNotPermitted", DriveRechazoError),  # política de Workspace
        (403, "insufficientPermissions", PermisoInsuficienteError),  # falta el scope
        (403, "sharingRateLimitExceeded", ServiceUnavailableError),
        (404, "notFound", EntityNotFoundError),
    ],
)
async def test_los_rechazos_al_compartir_tienen_su_remedio(
    status: int, motivo: str, esperado: type[Exception]
) -> None:
    """Sólo la falta de scope manda a /conectar (el remedio equivocado de PB-031)."""
    cuerpo = {"error": {"code": status, "errors": [{"reason": motivo}], "message": "x"}}
    drive = await _drive([], lambda _: httpx.Response(status, json=cuerpo))

    with pytest.raises(esperado):
        await drive.compartir(USUARIO, "f1", PermisoDeArchivo.LECTOR, "ana@ejemplo.com")


async def test_al_compartir_la_direccion_no_se_loguea() -> None:
    """El mensaje de Google puede repetir la dirección invitada (RF-18)."""
    cuerpo = {
        "error": {
            "code": 400,
            "errors": [{"reason": "invalidSharingRequest"}],
            "message": "Bad Request. User message: could not notify ana@ejemplo.com",
        }
    }
    drive = await _drive([], lambda _: httpx.Response(400, json=cuerpo))

    with structlog.testing.capture_logs() as eventos, pytest.raises(DriveRechazoError):
        await drive.compartir(USUARIO, "f1", PermisoDeArchivo.LECTOR, "ana@ejemplo.com")

    assert eventos
    assert "ana@ejemplo.com" not in json.dumps(eventos, default=str)


async def test_comprobar_acceso_no_le_pide_nada_a_drive() -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, lambda _: httpx.Response(200, json={}))

    await drive.comprobar_acceso(USUARIO)

    assert pedidos == []


# --- Renombrar, mover y mandar a la papelera (PB-039) ---------------------------------

RAIZ = "0AraizRealDeMiUnidad"


def _consulta_de(pedido: httpx.Request) -> dict[str, list[str]]:
    return parse_qs(urlparse(str(pedido.url)).query)


def _con_padres(*padres: str) -> Manejador:
    """La raíz, los padres del archivo y el PATCH, como la API."""

    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.method == "GET" and pedido.url.path == "/drive/v3/files/root":
            return httpx.Response(200, json={"id": RAIZ})
        if pedido.method == "GET":
            return httpx.Response(200, json={"parents": list(padres)})
        return httpx.Response(200, json={"id": "f1"})

    return responder


async def test_los_campos_dicen_si_es_propio_y_si_esta_compartido() -> None:
    campos = CAMPOS_DE_ARCHIVO.split(",")

    assert "shared" in campos
    assert "permissionIds" in campos


@pytest.mark.parametrize(
    ("duenos", "propio"),
    [
        ([{"displayName": "Vos", "me": True}], True),
        ([{"displayName": "Ana", "me": False}], False),
        ([{"displayName": "Sin marca"}], False),
        (None, False),  # Drive no lo dijo: no se toca
    ],
)
async def test_es_propio_solo_con_la_marca_de_drive(duenos: Any, propio: bool) -> None:
    item = _item(owners=duenos)
    drive = await _drive([], lambda _: httpx.Response(200, json=item))

    archivo = await drive.obtener(USUARIO, "f1")

    assert archivo.propio is propio


@pytest.mark.parametrize(
    ("cambios", "compartido"),
    [
        ({"shared": True, "permissionIds": ["yo"]}, True),
        # Una subcarpeta de una compartida hereda el acceso sin la marca.
        ({"shared": False, "permissionIds": ["yo", "ana"]}, True),
        ({"shared": False, "permissionIds": ["yo"]}, False),
        ({}, False),
        ({"shared": "true", "permissionIds": "yo,ana"}, False),  # formas que no son
    ],
)
async def test_esta_compartido_por_la_marca_o_por_los_permisos(
    cambios: dict[str, Any], compartido: bool
) -> None:
    item = _item(**cambios)
    drive = await _drive([], lambda _: httpx.Response(200, json=item))

    archivo = await drive.obtener(USUARIO, "f1")

    assert archivo.compartido is compartido


async def test_renombrar_es_un_patch_con_el_nombre() -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, lambda _: httpx.Response(200, json={"id": "f1"}))

    await drive.renombrar(USUARIO, "f1", "Presupuesto final.xlsx")

    [pedido] = pedidos
    assert pedido.method == "PATCH"
    assert pedido.url.path == "/drive/v3/files/f1"
    assert pedido.headers["Authorization"] == "Bearer access-vigente"
    assert json.loads(pedido.content) == {"name": "Presupuesto final.xlsx"}


async def test_a_la_papelera_es_un_patch_nunca_un_delete() -> None:
    """DELETE lo borra para siempre; la papelera se recupera durante 30 días."""
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, lambda _: httpx.Response(200, json={"id": "f1"}))

    await drive.a_la_papelera(USUARIO, "f1")

    [pedido] = pedidos
    assert pedido.method == "PATCH"
    assert pedido.url.path == "/drive/v3/files/f1"
    assert json.loads(pedido.content) == {"trashed": True}


async def test_mover_a_una_carpeta_saca_al_padre_de_antes() -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _con_padres("vieja"))

    await drive.mover(USUARIO, "f1", "c1")

    leer, mover = pedidos
    assert leer.method == "GET"
    assert leer.url.path == "/drive/v3/files/f1"
    assert _consulta_de(leer)["fields"] == ["parents"]
    assert mover.method == "PATCH"
    assert mover.url.path == "/drive/v3/files/f1"
    assert _consulta_de(mover)["addParents"] == ["c1"]
    assert _consulta_de(mover)["removeParents"] == ["vieja"]
    assert json.loads(mover.content) == {}


async def test_mover_a_mi_unidad_usa_el_id_real_de_la_raiz() -> None:
    """`parents` trae el id real, nunca `root`: sin él no se sabe qué sacar."""
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _con_padres("c1"))

    await drive.mover(USUARIO, "f1", None)

    raiz, _, mover = pedidos
    assert raiz.url.path == "/drive/v3/files/root"
    assert _consulta_de(raiz)["fields"] == ["id"]
    assert _consulta_de(mover)["addParents"] == [RAIZ]
    assert _consulta_de(mover)["removeParents"] == ["c1"]


@pytest.mark.parametrize(("destino", "padre"), [("c1", "c1"), (None, RAIZ)])
async def test_mover_adonde_ya_esta_no_modifica_nada(destino: str | None, padre: str) -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _con_padres(padre))

    await drive.mover(USUARIO, "f1", destino)

    assert "PATCH" not in [p.method for p in pedidos]


async def test_con_varios_padres_quedan_solo_en_el_destino() -> None:
    """Los archivos viejos pueden tener más de un padre: quedan sólo en el destino."""
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _con_padres("c1", "otra", "tercera"))

    await drive.mover(USUARIO, "f1", "c1")

    parametros = _consulta_de(pedidos[-1])
    assert "addParents" not in parametros  # ya estaba ahí
    assert parametros["removeParents"] == ["otra,tercera"]


@pytest.mark.parametrize(
    "accion",
    [
        lambda d: d.renombrar(USUARIO, "../about", "x"),
        lambda d: d.a_la_papelera(USUARIO, "f1/permissions"),
        lambda d: d.mover(USUARIO, "f1?alt=media", "c1"),
        lambda d: d.mover(USUARIO, "f1", "../c1"),  # también el destino
    ],
)
async def test_un_id_que_no_es_de_drive_no_genera_pedido(
    accion: Callable[[DriveGoogle], Awaitable[None]],
) -> None:
    pedidos: list[httpx.Request] = []
    drive = await _drive(pedidos, _con_padres("vieja"))

    with pytest.raises(EntityNotFoundError):
        await accion(drive)

    assert pedidos == []


@pytest.mark.parametrize(
    ("status", "motivo", "esperado"),
    [
        (403, "insufficientFilePermissions", DriveRechazoError),  # sin permiso sobre el archivo
        (403, "insufficientPermissions", PermisoInsuficienteError),  # falta el scope
        (404, "notFound", EntityNotFoundError),
    ],
)
async def test_los_rechazos_al_modificar_tienen_su_remedio(
    status: int, motivo: str, esperado: type[Exception]
) -> None:
    cuerpo = {"error": {"code": status, "errors": [{"reason": motivo}], "message": "x"}}
    drive = await _drive([], lambda _: httpx.Response(status, json=cuerpo))

    with pytest.raises(esperado):
        await drive.renombrar(USUARIO, "f1", "Otro nombre")
    with pytest.raises(esperado):
        await drive.a_la_papelera(USUARIO, "f1")


async def test_mover_a_una_carpeta_que_ya_no_existe() -> None:
    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.method == "GET":
            return httpx.Response(200, json={"parents": ["vieja"]})
        return httpx.Response(
            404, json={"error": {"code": 404, "errors": [{"reason": "notFound"}]}}
        )

    drive = await _drive([], responder)

    with pytest.raises(EntityNotFoundError):
        await drive.mover(USUARIO, "f1", "borrada")


async def test_sin_red_al_modificar_es_servicio_no_disponible() -> None:
    def caido(pedido: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=pedido)

    drive = await _drive([], caido)

    with pytest.raises(ServiceUnavailableError):
        await drive.renombrar(USUARIO, "f1", "Otro nombre")


async def test_al_modificar_los_nombres_no_se_loguean() -> None:
    """RF-18: qué se hizo y adónde (Mi unidad o no), nunca cómo se llama."""
    drive = await _drive([], _con_padres("c1"))

    with structlog.testing.capture_logs() as eventos:
        await drive.renombrar(USUARIO, "f1", "Resultados de biopsia")
        await drive.mover(USUARIO, "f1", None)
        await drive.a_la_papelera(USUARIO, "f1")

    hechos = [e["event"] for e in eventos]
    assert hechos == ["drive.renombrado", "drive.movido", "drive.a_la_papelera"]
    assert eventos[1]["a_mi_unidad"] is True
    assert "biopsia" not in json.dumps(eventos, default=str)
