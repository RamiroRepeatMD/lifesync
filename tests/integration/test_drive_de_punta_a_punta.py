"""Integración de Drive de punta a punta, sin red (PB-039).

La receta de la de Gmail (PB-036): el camino entero con piezas reales
(`AgenteGemini`, grafo, tools, `DriveGoogle`, `ConectarGoogle` con su refresco
perezoso), y sólo tres cosas falsas: el modelo, que pide las acciones como las
pide Gemini; lo que la persona manda por WhatsApp, cuya descarga ya tiene sus
tests; y Drive, simulado con estado por HTTP. Se aserta el pedido EXACTO que
recibe Drive (la `q`, el cuerpo del PATCH, los padres que se agregan y se
sacan) y que nada se escribe antes del sí.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.application.dto.archivo_adjunto import ArchivoAdjunto
from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.external.google.drive import CAMPOS_DE_ARCHIVO, DriveGoogle
from src.infrastructure.llm.agente_gemini import AgenteGemini
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import (
    AutorizadorFalso,
    ModeloFalso,
    RecibidosFalsos,
    RepositorioOAuthTokenEnMemoria,
)

USUARIO = uuid4()
RUTA = "/drive/v3/files"
RUTA_DE_SUBIDA = "/upload/drive/v3/files"
SESION = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=s1"
RAIZ = "0AMiUnidadDeVerdad"
CARPETA = "application/vnd.google-apps.folder"
PLANILLA = "application/vnd.google-apps.spreadsheet"
CONFIRMAR = "¿Confirmás?"
INFORME = ArchivoAdjunto(
    referencia="1037543291543636", tipo_mime="application/pdf", nombre="informe.pdf"
)


def _archivo(
    id_: str,
    nombre: str,
    *,
    tipo: str = PLANILLA,
    padres: tuple[str, ...] = (RAIZ,),
    de: str | None = None,
    compartido: bool = False,
) -> dict[str, Any]:
    """Un archivo con la forma de Drive: es de la persona salvo que se diga de quién."""
    return {
        "id": id_,
        "name": nombre,
        "mimeType": tipo,
        "modifiedTime": "2026-10-12T21:30:00.000Z",
        "webViewLink": f"https://drive.google.com/file/d/{id_}/view",
        "owners": [{"displayName": de or "Vos", "me": de is None}],
        "shared": compartido,
        "permissionIds": ["yo", "alguien"] if compartido else ["yo"],
        "parents": list(padres),
        "trashed": False,
    }


def _parametros(pedido: httpx.Request) -> dict[str, list[str]]:
    return parse_qs(urlparse(str(pedido.url)).query)


def _no_esta() -> httpx.Response:
    return httpx.Response(404, json={"error": {"code": 404, "errors": [{"reason": "notFound"}]}})


class DriveFalso:
    """Contesta como la API de Drive v3, con estado, y registra cada pedido.

    Una ruta que no conoce da 404 y queda registrada: así se ve si un pedido
    salió del recurso de los archivos.
    """

    def __init__(self, *archivos: dict[str, Any]) -> None:
        self.archivos = {a["id"]: a for a in archivos}
        self.pedidos: list[httpx.Request] = []
        self.rechazo_al_escribir: httpx.Response | None = None
        self._subida: dict[str, Any] | None = None

    def __call__(self, pedido: httpx.Request) -> httpx.Response:
        self.pedidos.append(pedido)
        if pedido.method != "GET" and self.rechazo_al_escribir is not None:
            return self.rechazo_al_escribir
        ruta = pedido.url.path
        if ruta == RUTA_DE_SUBIDA:
            return self._subir(pedido)
        if pedido.method == "GET" and ruta == RUTA:
            return self._listar(pedido)
        if pedido.method == "GET" and ruta == f"{RUTA}/root":
            return httpx.Response(200, json={"id": RAIZ})
        archivo_id, _, recurso = ruta.removeprefix(f"{RUTA}/").partition("/")
        archivo = self.archivos.get(archivo_id) if ruta.startswith(f"{RUTA}/") else None
        if archivo is None:
            return _no_esta()
        if pedido.method == "GET" and not recurso:
            campos = _parametros(pedido)["fields"][0]
            if campos == "parents":
                return httpx.Response(200, json={"parents": archivo["parents"]})
            return httpx.Response(200, json=_vista(archivo))
        if pedido.method == "PATCH" and not recurso:
            return self._modificar(pedido, archivo)
        if pedido.method == "POST" and recurso == "permissions":
            archivo["permissionIds"].append(f"p{len(archivo['permissionIds'])}")
            return httpx.Response(200, json={"id": archivo["permissionIds"][-1]})
        return _no_esta()

    def _listar(self, pedido: httpx.Request) -> httpx.Response:
        q = _parametros(pedido)["q"][0]
        visibles = [a for a in self.archivos.values() if not a["trashed"]]
        if f"mimeType = '{CARPETA}'" in q:
            visibles = [a for a in visibles if a["mimeType"] == CARPETA]
        if "'me' in owners" in q:
            visibles = [a for a in visibles if a["owners"][0]["me"]]
        if texto := re.search(r"name contains '((?:[^'\\]|\\.)*)'", q):
            buscado = texto.group(1).replace("\\'", "'").replace("\\\\", "\\").lower()
            visibles = [a for a in visibles if buscado in a["name"].lower()]
        return httpx.Response(200, json={"files": [_vista(a) for a in visibles]})

    def _modificar(self, pedido: httpx.Request, archivo: dict[str, Any]) -> httpx.Response:
        cuerpo: dict[str, Any] = json.loads(pedido.content or b"{}")
        parametros = _parametros(pedido)
        for destino in parametros.get("addParents", []):
            if destino != RAIZ and destino not in self.archivos:
                return _no_esta()
            archivo["parents"].append(destino)
        for padre in ",".join(parametros.get("removeParents", [])).split(","):
            if padre in archivo["parents"]:
                archivo["parents"].remove(padre)
        archivo["name"] = cuerpo.get("name", archivo["name"])
        archivo["trashed"] = cuerpo.get("trashed", archivo["trashed"])
        return httpx.Response(200, json={"id": archivo["id"]})

    def _subir(self, pedido: httpx.Request) -> httpx.Response:
        if pedido.method == "POST":
            self._subida = json.loads(pedido.content)
            return httpx.Response(200, headers={"Location": SESION})
        if pedido.method != "PUT" or self._subida is None:
            return _no_esta()
        metadata, self._subida = self._subida, None
        nuevo = _archivo(
            f"subido{len(self.archivos)}",
            metadata["name"],
            tipo=metadata["mimeType"],
            padres=tuple(metadata.get("parents", [RAIZ])),
        )
        self.archivos[nuevo["id"]] = nuevo
        return httpx.Response(200, json=_vista(nuevo))

    def escrituras(self) -> list[httpx.Request]:
        return [p for p in self.pedidos if p.method != "GET"]


def _vista(archivo: dict[str, Any]) -> dict[str, Any]:
    """Sólo los campos que pide el adaptador, como hace Drive con `fields`."""
    campos = ("id", "name", "mimeType", "modifiedTime", "webViewLink", "owners")
    return {clave: archivo[clave] for clave in (*campos, "shared", "permissionIds")}


class Bot:
    """El agente de verdad hasta el HTTP: el modelo, Drive y WhatsApp son falsos.

    `cuenta` es "vigente" o "vencida" (el token se renueva al usarlo). Con
    `adjunto`, la persona lo manda con su primer mensaje.
    """

    def __init__(
        self,
        drive: DriveFalso,
        *guion: AIMessage,
        cuenta: str = "vigente",
        adjunto: ArchivoAdjunto | None = None,
    ) -> None:
        self.drive = drive
        self.autorizador = AutorizadorFalso(usuario_fijo=USUARIO)
        self.tokens = RepositorioOAuthTokenEnMemoria()
        self.recibidos = RecibidosFalsos()
        self._cuenta = cuenta
        self._adjunto = adjunto
        http = httpx.AsyncClient(transport=httpx.MockTransport(drive))
        archivos = DriveGoogle(http, ConectarGoogle(self.tokens, self.autorizador))
        herramientas = construir_herramientas(None, archivos=archivos, recibidos=self.recibidos)
        self.grafo = construir_grafo(ModeloFalso(guion=list(guion)), herramientas, InMemorySaver())
        self.agente = AgenteGemini(self.grafo)

    async def decir(self, texto: str) -> str:
        if not await self.tokens.listar_por_usuario(USUARIO):
            vence = timedelta(hours=-1) if self._cuenta == "vencida" else timedelta(hours=1)
            await self.tokens.guardar(
                OAuthToken(
                    usuario_id=USUARIO,
                    proveedor=ProveedorOAuth.GOOGLE,
                    access_token="access-vigente",
                    refresh_token="refresh",
                    expira_en=datetime.now(UTC) + vence,
                )
            )
        adjunto, self._adjunto = self._adjunto, None
        return await self.agente.responder(
            ConsultaDelUsuario(
                conversacion_id=USUARIO, usuario_id=USUARIO, texto=texto, adjunto=adjunto
            )
        )

    async def respuestas_de_tools(self) -> list[str]:
        estado = await self.grafo.aget_state({"configurable": {"thread_id": str(USUARIO)}})
        return [str(m.content) for m in estado.values["messages"] if isinstance(m, ToolMessage)]


def _llamadas(*pedidos: tuple[str, dict[str, Any]]) -> AIMessage:
    """Un mensaje del modelo con todas las llamadas juntas, como hace Gemini."""
    return AIMessage(
        "",
        tool_calls=[
            {"name": nombre, "args": args, "id": f"c{i}"}
            for i, (nombre, args) in enumerate(pedidos)
        ],
    )


PRESUPUESTO = _archivo("f1", "Presupuesto 2026")
FACTURA = _archivo("f2", "factura.pdf", tipo="application/pdf")
DE_ANA = _archivo("f3", "Notas de Ana", de="Ana Gómez")
FACULTAD = _archivo("c1", "Facultad", tipo=CARPETA)
EQUIPO = _archivo("c2", "Equipo", tipo=CARPETA, compartido=True)


def _drive(*archivos: dict[str, Any]) -> DriveFalso:
    """Un Drive con copias: cada test cambia el suyo."""
    return DriveFalso(*(json.loads(json.dumps(a)) for a in archivos))


# --- Buscar ---------------------------------------------------------------------------------


async def test_buscar_arma_la_consulta_y_cada_linea_trae_el_id_y_el_link() -> None:
    drive = _drive(PRESUPUESTO, FACTURA)
    bot = Bot(drive, _llamadas(("buscar_archivos", {"texto": "presu"})), AIMessage("Está."))

    await bot.decir("¿dónde está el presupuesto?")

    [pedido] = drive.pedidos
    assert _parametros(pedido)["q"] == [
        "trashed = false and (name contains 'presu' or fullText contains 'presu')"
    ]
    [listado] = await bot.respuestas_de_tools()
    assert "«Presupuesto 2026» — planilla" in listado
    assert listado.endswith("id: f1 — link: https://drive.google.com/file/d/f1/view")


# --- Subir ------------------------------------------------------------------------------------


async def test_subir_a_mi_unidad_abre_la_sesion_y_manda_los_bytes() -> None:
    drive = _drive()
    bot = Bot(drive, _llamadas(("subir_archivo", {})), AIMessage("Listo."), adjunto=INFORME)

    respuesta = await bot.decir("subilo a mi Drive")

    assert CONFIRMAR not in respuesta  # crear uno: directo
    abrir, mandar = drive.escrituras()
    assert json.loads(abrir.content) == {"name": "informe.pdf", "mimeType": "application/pdf"}
    assert mandar.method == "PUT"
    assert mandar.content == bot.recibidos.contenido
    [subido] = drive.archivos.values()
    assert subido["parents"] == [RAIZ]


async def test_subir_a_una_carpeta_compartida_espera_el_si() -> None:
    drive = _drive(EQUIPO)
    bot = Bot(
        drive,
        _llamadas(("subir_archivo", {"carpeta": "Equipo"})),
        AIMessage("Listo."),
        adjunto=INFORME,
    )

    pregunta = await bot.decir("subilo a la carpeta Equipo")

    assert CONFIRMAR in pregunta
    assert "esa carpeta está compartida" in pregunta
    assert drive.escrituras() == []
    assert bot.recibidos.descargas == []
    await bot.decir("sí")
    abrir, _ = drive.escrituras()
    assert json.loads(abrir.content)["parents"] == ["c2"]


# --- Compartir --------------------------------------------------------------------------------


async def test_compartir_manda_el_permiso_exacto_recien_con_el_si() -> None:
    drive = _drive(PRESUPUESTO)
    bot = Bot(
        drive,
        _llamadas(("compartir_archivo", {"id_archivo": "f1", "con": "ana@ejemplo.com"})),
        AIMessage("Listo."),
    )

    await bot.decir("compartile el presupuesto a Ana")

    assert drive.escrituras() == []
    await bot.decir("sí")
    [permiso] = drive.escrituras()
    assert permiso.url.path == f"{RUTA}/f1/permissions"
    assert json.loads(permiso.content) == {
        "type": "user",
        "role": "reader",
        "emailAddress": "ana@ejemplo.com",
    }


# --- Renombrar ----------------------------------------------------------------------------------


async def test_buscar_y_renombrar_pide_el_si_y_conserva_la_extension() -> None:
    """El caso de todos los días: el id sale de un listado, y eso activa el modo estricto."""
    drive = _drive(FACTURA)
    bot = Bot(
        drive,
        _llamadas(("buscar_archivos", {"texto": "factura"})),
        _llamadas(("renombrar_archivo", {"id_archivo": "f2", "nombre_nuevo": "Factura de luz"})),
        AIMessage("Listo."),
    )

    pregunta = await bot.decir("a la factura ponele Factura de luz")

    assert "Renombrar «factura.pdf» (PDF) a «Factura de luz.pdf»" in pregunta
    assert drive.escrituras() == []
    await bot.decir("sí")
    [patch] = drive.escrituras()
    assert patch.method == "PATCH"
    assert patch.url.path == f"{RUTA}/f2"
    assert json.loads(patch.content) == {"name": "Factura de luz.pdf"}
    assert drive.archivos["f2"]["name"] == "Factura de luz.pdf"


# --- Mover --------------------------------------------------------------------------------------


async def test_mover_a_una_carpeta_saca_el_archivo_de_mi_unidad() -> None:
    drive = _drive(PRESUPUESTO, FACULTAD)
    bot = Bot(
        drive,
        _llamadas(("mover_archivo", {"id_archivo": "f1", "carpeta": "Facultad"})),
        AIMessage("Listo."),
    )

    await bot.decir("mové el presupuesto a Facultad")

    leer, carpetas, padres, patch = drive.pedidos
    assert (leer.url.path, _parametros(leer)["fields"]) == (f"{RUTA}/f1", [CAMPOS_DE_ARCHIVO])
    assert "'me' in owners" in _parametros(carpetas)["q"][0]  # sólo carpetas propias
    assert _parametros(padres)["fields"] == ["parents"]
    assert patch.method == "PATCH"
    assert _parametros(patch)["addParents"] == ["c1"]
    assert _parametros(patch)["removeParents"] == [RAIZ]
    assert drive.archivos["f1"]["parents"] == ["c1"]


async def test_mover_a_mi_unidad_usa_el_id_real_de_la_raiz() -> None:
    drive = _drive(_archivo("f1", "Presupuesto 2026", padres=("c1",)), FACULTAD)
    bot = Bot(
        drive,
        _llamadas(("mover_archivo", {"id_archivo": "f1", "carpeta": "Mi unidad"})),
        AIMessage("Listo."),
    )

    await bot.decir("sacalo de Facultad")

    # El alias no busca una carpeta que se llame "Mi unidad".
    assert [p.url.path for p in drive.pedidos] == [
        f"{RUTA}/f1",
        f"{RUTA}/root",
        f"{RUTA}/f1",
        f"{RUTA}/f1",
    ]
    patch = drive.pedidos[-1]
    assert _parametros(patch)["addParents"] == [RAIZ]
    assert _parametros(patch)["removeParents"] == ["c1"]
    assert drive.archivos["f1"]["parents"] == [RAIZ]


async def test_mover_adonde_ya_esta_no_modifica_nada() -> None:
    drive = _drive(PRESUPUESTO)
    bot = Bot(
        drive,
        _llamadas(("mover_archivo", {"id_archivo": "f1", "carpeta": "Mi unidad"})),
        AIMessage("Ya estaba."),
    )

    await bot.decir("movelo a mi unidad")

    assert drive.escrituras() == []
    assert (await bot.respuestas_de_tools()) == ["Listo: «Presupuesto 2026» quedó en Mi unidad."]


# --- Mandar a la papelera --------------------------------------------------------------------


@pytest.mark.parametrize(("decision", "en_la_papelera"), [("sí", True), ("no", False)])
async def test_borrar_manda_a_la_papelera_solo_con_el_si(
    decision: str, en_la_papelera: bool
) -> None:
    drive = _drive(PRESUPUESTO)
    bot = Bot(drive, _llamadas(("borrar_archivo", {"id_archivo": "f1"})), AIMessage("Listo."))

    pregunta = await bot.decir("borrá el presupuesto")

    assert "Mandar a la papelera «Presupuesto 2026»" in pregunta
    assert drive.escrituras() == []
    await bot.decir(decision)
    if en_la_papelera:
        [patch] = drive.escrituras()
        assert json.loads(patch.content) == {"trashed": True}  # nunca DELETE
    else:
        assert drive.escrituras() == []
    assert drive.archivos["f1"]["trashed"] is en_la_papelera


# --- Lo que no se toca -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("herramienta", "args"),
    [
        ("renombrar_archivo", {"id_archivo": "f3", "nombre_nuevo": "Mías"}),
        ("mover_archivo", {"id_archivo": "f3", "carpeta": "Facultad"}),
        ("borrar_archivo", {"id_archivo": "f3"}),
        ("mover_archivo", {"id_archivo": "c1", "carpeta": "Mi unidad"}),
        ("borrar_archivo", {"id_archivo": "c1"}),
    ],
    ids=["renombrar_ajeno", "mover_ajeno", "borrar_ajeno", "mover_carpeta", "borrar_carpeta"],
)
async def test_lo_ajeno_y_las_carpetas_no_se_modifican(
    herramienta: str, args: dict[str, str]
) -> None:
    drive = _drive(DE_ANA, FACULTAD)
    bot = Bot(drive, _llamadas((herramienta, args)), AIMessage("No puedo."))

    respuesta = await bot.decir("hacelo")

    assert CONFIRMAR not in respuesta
    assert drive.escrituras() == []


# --- Rechazos y la cuenta ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("motivo", "esperado", "manda_a_conectar"),
    [
        ("insufficientFilePermissions", "Drive no confirmó que se haya hecho", False),
        ("insufficientPermissions", "/conectar", True),  # falta el scope
    ],
)
async def test_un_rechazo_de_drive_trae_su_remedio(
    motivo: str, esperado: str, manda_a_conectar: bool
) -> None:
    drive = _drive(PRESUPUESTO)
    drive.rechazo_al_escribir = httpx.Response(
        403, json={"error": {"code": 403, "errors": [{"reason": motivo}], "message": "x"}}
    )
    bot = Bot(
        drive,
        _llamadas(("renombrar_archivo", {"id_archivo": "f1", "nombre_nuevo": "Otro"})),
        AIMessage("No se pudo."),
    )

    await bot.decir("renombralo")

    [texto] = await bot.respuestas_de_tools()
    assert esperado in texto
    assert ("/conectar" in texto) is manda_a_conectar
    assert drive.archivos["f1"]["name"] == "Presupuesto 2026"


@pytest.mark.parametrize("id_del_modelo", ["../about", "f1/permissions", "f1?alt=media"])
async def test_un_id_del_modelo_no_saca_el_pedido_de_los_archivos(id_del_modelo: str) -> None:
    drive = _drive(PRESUPUESTO)
    bot = Bot(
        drive,
        _llamadas(("renombrar_archivo", {"id_archivo": id_del_modelo, "nombre_nuevo": "x"})),
        AIMessage("No lo encontré."),
    )

    await bot.decir("renombralo")

    assert drive.pedidos == []  # un id que no tiene la forma de Drive ni siquiera sale
    [texto] = await bot.respuestas_de_tools()
    assert "No encontré ese archivo" in texto


async def test_con_el_token_vencido_se_renueva_una_vez_y_escribe_con_el_nuevo() -> None:
    drive = _drive(PRESUPUESTO)
    bot = Bot(
        drive,
        _llamadas(("borrar_archivo", {"id_archivo": "f1"})),
        AIMessage("Listo."),
        cuenta="vencida",
    )

    await bot.decir("borrá el presupuesto")
    await bot.decir("sí")

    assert bot.autorizador.refresh_usados == ["refresh"]
    [patch] = drive.escrituras()
    assert patch.headers["Authorization"] == "Bearer access-de-prueba"  # el renovado
