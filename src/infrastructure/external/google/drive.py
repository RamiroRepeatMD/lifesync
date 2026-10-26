"""Adaptador de Google Drive: buscar y listar (PB-037), subir y compartir (PB-038),
renombrar, mover y mandar a la papelera (PB-039).

Implementa el puerto `Archivos` contra la API REST de Drive v3, con el mismo
trío que Calendar, Tasks y Gmail: el cliente httpx compartido de Google, las
credenciales vigentes de `ConectarGoogle` y la traducción de rechazos de
`transporte`.

Lo que conviene saber antes de tocarlo:

- **La consulta la arma este adaptador; el modelo nunca la escribe.** En Gmail
  la escribe el modelo, pero aquella sintaxis sólo filtra. La de Drive combina
  condiciones con `and` y `or`, y un texto pegado tal cual podría sumar las
  suyas (`x' or trashed = true or name contains '`). El texto de la persona
  entra únicamente como literal, con el escape que pide la documentación de
  Drive: la barra y la comilla simple llevan una barra adelante.
- **El texto se busca en el nombre y en el contenido.** `name contains` busca
  por el comienzo de las palabras del nombre ("presu" encuentra "Presupuesto
  2026"); `fullText contains`, palabras enteras en el nombre, la descripción
  y el contenido ("factura" encuentra el PDF que la menciona adentro).
- **Con texto no se ordena.** Drive devuelve por relevancia, que es lo que
  sirve para "¿dónde está…?", y la documentación no aclara si `orderBy`
  convive con `fullText`. Sin texto —listar— van los últimos modificados.
- **Una sola página, de hasta 10.** La persona pide un archivo o los últimos,
  no un inventario.
- **El permiso es `drive`, el completo** (PB-038): es el único que deja
  compartir cualquier archivo y subir a una carpeta que ya existe. Lo que el
  bot hace con él lo acota este código: buscar, subir, compartir, renombrar,
  mover y mandar a la papelera. Nada lee el contenido ni borra para siempre
  (eso es DELETE, y no se usa).
- **Los rechazos de una escritura no son los de una lectura.** Al escribir,
  sólo la falta de scope manda a /conectar: el resto de los 403 y 400 (sin
  permiso sobre el archivo, una cuenta de Workspace que no comparte afuera,
  Drive lleno) es algo que reconectar no arregla.
- **Mover pide antes el id real de Mi unidad.** `parents` trae ese id, nunca
  el alias `root`: sin él no se sabe si el archivo ya está ahí ni qué padre
  sacar.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import structlog

from src.application.ports.archivos import Archivos
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.archivo import Archivo
from src.domain.exceptions import (
    CuentaNoConectadaError,
    DriveLlenoError,
    DriveRechazoError,
    EntityNotFoundError,
    ServiceUnavailableError,
)
from src.domain.value_objects.permiso_de_archivo import PermisoDeArchivo
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo
from src.infrastructure.external.google.transporte import (
    API_DESHABILITADA,
    LIMITE_DE_USO,
    json_o_vacio,
    motivos_de,
    traducir_rechazo,
)

logger = structlog.get_logger(__name__)

BASE = "https://www.googleapis.com/drive/v3"
SUBIDA = "https://www.googleapis.com/upload/drive/v3/files"
HOST_DE_SUBIDA = "www.googleapis.com"
CANTIDAD_MAXIMA = 10
# Lo justo para mostrar, abrir y decidir qué se puede tocar (PB-039: de quién
# es y si alguien más lo ve): ningún otro campo, y nada del contenido. La
# forma `files(...)` sólo la acepta la búsqueda; un pedido de UN archivo
# (leerlo, subirlo) con eso da 400.
CAMPOS_DE_ARCHIVO = (
    "id,name,mimeType,modifiedTime,webViewLink,owners(displayName,me),shared,permissionIds"
)
CAMPOS = f"files({CAMPOS_DE_ARCHIVO})"
# La forma de un id de Drive. Lo pasa el modelo y termina en una URL: uno con
# otra forma no puede existir, así que ni se pide (la lección de PB-036).
ID_DE_DRIVE = re.compile(r"[A-Za-z0-9_-]{1,128}")
# Una subida grande tarda más que una consulta (el tope es de 20 MB).
TIMEOUT_DE_SUBIDA = httpx.Timeout(60.0, connect=5.0)

# Los 403 de una escritura que SÍ tienen el remedio de siempre: falta el
# scope (→ /conectar), la API deshabilitada o un límite (→ esperar).
_SIN_EL_SCOPE = frozenset({"insufficientPermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT"})
_DE_SIEMPRE = _SIN_EL_SCOPE | API_DESHABILITADA | LIMITE_DE_USO
_DRIVE_LLENO = frozenset({"storageQuotaExceeded"})
_ROL = {PermisoDeArchivo.LECTOR: "reader", PermisoDeArchivo.EDITOR: "writer"}
# De sobra para un nombre o unas palabras: un texto enorme no viaja.
MAX_CARACTERES_DE_TEXTO = 100

# Cada tipo, en el formato de Google y en los de Office: "la planilla" puede
# ser una planilla de Google o un Excel que se subió tal cual.
_FORMATOS: dict[TipoDeArchivo, tuple[str, ...]] = {
    TipoDeArchivo.DOCUMENTO: (
        "application/vnd.google-apps.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ),
    TipoDeArchivo.PLANILLA: (
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ),
    TipoDeArchivo.PRESENTACION: (
        "application/vnd.google-apps.presentation",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    ),
    TipoDeArchivo.PDF: ("application/pdf",),
    TipoDeArchivo.CARPETA: ("application/vnd.google-apps.folder",),
}
_TIPO_POR_FORMATO = {formato: tipo for tipo, formatos in _FORMATOS.items() for formato in formatos}
# Las imágenes son cualquier `image/…`: no tiene sentido enumerar los formatos.
_PREFIJO_DE_IMAGEN = "image/"


class DriveGoogle(Archivos):
    """El puerto `Archivos` hablando con la API real."""

    def __init__(self, cliente: httpx.AsyncClient, conectar: ConectarGoogle) -> None:
        self._cliente = cliente
        self._conectar = conectar

    async def buscar(
        self,
        usuario_id: UUID,
        texto: str,
        tipo: TipoDeArchivo | None,
        desde: datetime | None,
        cantidad: int,
    ) -> tuple[Archivo, ...]:
        token = await self._credencial(usuario_id)
        literal = _literal(texto)
        parametros = {
            "q": _consulta(literal, tipo, desde),
            "pageSize": str(max(1, min(cantidad, CANTIDAD_MAXIMA))),
            "fields": CAMPOS,
        }
        if not literal:
            parametros["orderBy"] = "modifiedTime desc"
        datos = await self._pedir(f"{BASE}/files", token, parametros)
        archivos = tuple(
            archivo
            for item in datos.get("files") or []
            if isinstance(item, dict) and (archivo := _a_archivo(item)) is not None
        )
        # Cantidades, nunca nombres (RF-18).
        logger.info("drive.buscados", usuario_id=str(usuario_id), cantidad=len(archivos))
        return archivos

    async def comprobar_acceso(self, usuario_id: UUID) -> None:
        # Sólo la credencial: a Google se le habla únicamente si hay que renovarla.
        await self._credencial(usuario_id)

    async def obtener(self, usuario_id: UUID, archivo_id: str) -> Archivo:
        url = _url_del_archivo(archivo_id)  # antes que nada: el id lo pasa el modelo
        token = await self._credencial(usuario_id)
        datos = await self._pedir(
            url,
            token,
            {"fields": CAMPOS_DE_ARCHIVO},
            error_404=EntityNotFoundError("Ese archivo no está en tu Drive."),
        )
        archivo = _a_archivo(datos)
        if archivo is None:
            raise EntityNotFoundError("Drive no devolvió el archivo.")
        return archivo

    async def carpetas(self, usuario_id: UUID, nombre: str) -> tuple[Archivo, ...]:
        literal = _literal(nombre)
        if not literal:
            return ()
        token = await self._credencial(usuario_id)
        condiciones = (
            "trashed = false",
            f"mimeType = '{_FORMATOS[TipoDeArchivo.CARPETA][0]}'",
            # Sólo las propias: en una ajena, el archivo heredaría su forma de
            # compartir y le llegaría a un tercero sin el sí de la persona.
            "'me' in owners",
            f"name contains '{literal}'",
        )
        datos = await self._pedir(
            f"{BASE}/files",
            token,
            {
                "q": " and ".join(condiciones),
                "pageSize": str(CANTIDAD_MAXIMA),
                "fields": CAMPOS,
                "orderBy": "modifiedTime desc",
            },
        )
        return tuple(
            archivo
            for item in datos.get("files") or []
            if isinstance(item, dict) and (archivo := _a_archivo(item)) is not None
        )

    async def subir(
        self,
        usuario_id: UUID,
        nombre: str,
        tipo_mime: str,
        contenido: bytes,
        carpeta_id: str | None,
    ) -> Archivo:
        """Subida reanudable: se abre la sesión con la metadata y se mandan los bytes.

        La multipart se recomienda hasta 5 MB; la reanudable vale para cualquier
        tamaño. La URL de la sesión la da Google en `Location` y se valida antes
        de mandarle nada, igual que el link de descarga de Meta.
        """
        token = await self._credencial(usuario_id)
        metadata: dict[str, Any] = {"name": nombre, "mimeType": tipo_mime}
        if carpeta_id is not None:
            metadata["parents"] = [_id_valido(carpeta_id)]
        autorizacion = {"Authorization": f"Bearer {token}"}
        try:
            apertura = await self._cliente.post(
                SUBIDA,
                params={"uploadType": "resumable", "fields": CAMPOS_DE_ARCHIVO},
                json=metadata,
                headers={
                    **autorizacion,
                    "X-Upload-Content-Type": tipo_mime,
                    "X-Upload-Content-Length": str(len(contenido)),
                },
            )
        except httpx.HTTPError as exc:
            logger.error("drive.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Google Drive.") from None
        _revisar_escritura(apertura)

        sesion = _sesion_de(apertura)
        try:
            final = await self._cliente.put(
                sesion,
                content=contenido,
                headers={**autorizacion, "Content-Type": tipo_mime},
                timeout=TIMEOUT_DE_SUBIDA,
            )
        except httpx.HTTPError as exc:
            # Los bytes pudieron haber llegado: quien lo cuenta lo aclara.
            logger.error("drive.subida_interrumpida", tipo=type(exc).__name__)
            raise ServiceUnavailableError("La subida a Drive no terminó.") from None
        _revisar_escritura(final)

        archivo = _a_archivo(json_o_vacio(final))
        if archivo is None:
            logger.error("drive.subida_sin_archivo")
            raise ServiceUnavailableError("Drive no devolvió el archivo subido.")
        # El tamaño y nada más: nunca el nombre (RF-18).
        logger.info("drive.subido", usuario_id=str(usuario_id), bytes=len(contenido))
        return archivo

    async def compartir(
        self,
        usuario_id: UUID,
        archivo_id: str,
        permiso: PermisoDeArchivo,
        direccion: str | None,
    ) -> None:
        url = f"{_url_del_archivo(archivo_id)}/permissions"
        token = await self._credencial(usuario_id)
        # Con una persona, Google le avisa por correo: es su valor por defecto,
        # y sin aviso rechaza a quien no tiene cuenta de Google. Con el link no
        # hay a quién avisarle.
        cuerpo: dict[str, str] = (
            {"type": "user", "role": _ROL[permiso], "emailAddress": direccion}
            if direccion is not None
            else {"type": "anyone", "role": _ROL[permiso]}
        )
        try:
            respuesta = await self._cliente.post(
                url,
                params={"fields": "id"},
                json=cuerpo,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            logger.error("drive.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Google Drive.") from None
        _revisar_escritura(respuesta)
        # A quién no: es una dirección (RF-18). El tipo y el permiso alcanzan.
        logger.info(
            "drive.compartido",
            usuario_id=str(usuario_id),
            con="persona" if direccion is not None else "link",
            permiso=str(permiso),
        )

    async def renombrar(self, usuario_id: UUID, archivo_id: str, nombre: str) -> None:
        url = _url_del_archivo(archivo_id)
        token = await self._credencial(usuario_id)
        await self._modificar(url, token, {"name": nombre})
        # Que se renombró, nunca cómo se llama (RF-18).
        logger.info("drive.renombrado", usuario_id=str(usuario_id))

    async def mover(self, usuario_id: UUID, archivo_id: str, carpeta_id: str | None) -> None:
        url = _url_del_archivo(archivo_id)
        pedido = _id_valido(carpeta_id) if carpeta_id is not None else None
        token = await self._credencial(usuario_id)
        destino = pedido if pedido is not None else await self._raiz(token)
        datos = await self._pedir(
            url,
            token,
            {"fields": "parents"},
            error_404=EntityNotFoundError("Ese archivo no está en tu Drive."),
        )
        padres = [p for p in datos.get("parents") or [] if isinstance(p, str)]
        # Queda sólo en el destino: un archivo viejo puede tener más de un padre.
        quitar = [padre for padre in padres if padre != destino]
        if destino in padres and not quitar:
            return  # ya está ahí
        parametros = {"removeParents": ",".join(quitar)} if quitar else {}
        if destino not in padres:
            parametros["addParents"] = destino
        await self._modificar(url, token, {}, parametros)
        logger.info("drive.movido", usuario_id=str(usuario_id), a_mi_unidad=pedido is None)

    async def a_la_papelera(self, usuario_id: UUID, archivo_id: str) -> None:
        url = _url_del_archivo(archivo_id)
        token = await self._credencial(usuario_id)
        await self._modificar(url, token, {"trashed": True})
        logger.info("drive.a_la_papelera", usuario_id=str(usuario_id))

    # --- Plomería ---------------------------------------------------------

    async def _credencial(self, usuario_id: UUID) -> str:
        token = await self._conectar.credencial_vigente(usuario_id, datetime.now(UTC))
        if token is None:
            raise CuentaNoConectadaError
        return token.access_token

    async def _raiz(self, token: str) -> str:
        """El id real de Mi unidad: `root` es un alias que `parents` nunca trae."""
        raiz = (await self._pedir(f"{BASE}/files/root", token, {"fields": "id"})).get("id")
        if not isinstance(raiz, str) or not ID_DE_DRIVE.fullmatch(raiz):
            logger.error("drive.raiz_inesperada")
            raise ServiceUnavailableError("Drive no dijo cuál es Mi unidad.")
        return raiz

    async def _modificar(
        self,
        url: str,
        token: str,
        cuerpo: dict[str, Any],
        parametros: dict[str, str] | None = None,
    ) -> None:
        """Un `files.update` (PATCH): renombrar, mover o mandar a la papelera."""
        try:
            respuesta = await self._cliente.patch(
                url,
                params={"fields": "id", **(parametros or {})},
                json=cuerpo,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            logger.error("drive.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Google Drive.") from None
        _revisar_escritura(respuesta)

    async def _pedir(
        self,
        url: str,
        token: str,
        parametros: dict[str, str],
        error_404: Exception | None = None,
    ) -> dict[str, Any]:
        try:
            respuesta = await self._cliente.get(
                url, params=parametros, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            logger.error("drive.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Google Drive.") from None

        if respuesta.status_code == httpx.codes.NOT_FOUND and error_404 is not None:
            raise error_404
        traducir_rechazo(respuesta, "drive")
        return json_o_vacio(respuesta)


def _id_valido(archivo_id: str) -> str:
    """El id, sólo si tiene la forma de uno de Drive.

    Raises:
        EntityNotFoundError: Si no la tiene: un archivo así no existe, y armar
            la URL con él sacaría el pedido del recurso de los archivos.
    """
    if not ID_DE_DRIVE.fullmatch(archivo_id):
        # El largo y nada más: el id lo escribió el modelo, quizás inducido.
        logger.warning("drive.id_invalido", largo=len(archivo_id))
        raise EntityNotFoundError("Ese archivo no está en tu Drive.")
    return archivo_id


def _url_del_archivo(archivo_id: str) -> str:
    return f"{BASE}/files/{_id_valido(archivo_id)}"


def _revisar_escritura(respuesta: httpx.Response) -> None:
    """Los rechazos de una escritura, cada uno con su remedio.

    Sólo se loguean los códigos de los motivos, nunca el mensaje de Google: al
    compartir, ese texto puede repetir la dirección invitada (RF-18).
    """
    if respuesta.status_code < httpx.codes.BAD_REQUEST:
        return
    motivos = motivos_de(respuesta)
    if respuesta.status_code == httpx.codes.NOT_FOUND:
        logger.warning("drive.escritura_sin_destino")
        raise EntityNotFoundError("Ese archivo o esa carpeta ya no están en tu Drive.")
    if motivos & _DRIVE_LLENO:
        logger.warning("drive.lleno")
        raise DriveLlenoError("No hay espacio en el Drive.")
    rechazo = respuesta.status_code == httpx.codes.BAD_REQUEST or (
        respuesta.status_code == httpx.codes.FORBIDDEN and not motivos & _DE_SIEMPRE
    )
    if rechazo:
        logger.warning(
            "drive.escritura_rechazada",
            status_code=respuesta.status_code,
            motivos=sorted(motivos),
        )
        raise DriveRechazoError("Drive no dejó hacerlo.")
    traducir_rechazo(respuesta, "drive", registrar_mensaje=False)


def _sesion_de(apertura: httpx.Response) -> httpx.URL:
    """La URL de la sesión de subida, sólo si es de Google: ahí van los bytes."""
    try:
        sesion = httpx.URL(apertura.headers.get("Location", ""))
    except httpx.InvalidURL:
        sesion = None
    valida = (
        sesion is not None
        and sesion.scheme == "https"
        and sesion.host == HOST_DE_SUBIDA
        and not sesion.userinfo
        and sesion.port is None
    )
    if sesion is None or not valida:
        logger.error("drive.sesion_inesperada", host=sesion.host if sesion is not None else "")
        raise ServiceUnavailableError("Drive no abrió la subida.")
    return sesion


# --- La consulta --------------------------------------------------------------------


def _literal(texto: str) -> str:
    """El texto de la persona, listo para ir entre comillas en la consulta.

    En una línea, sin caracteres de control y recortado; después, el escape:
    primero la barra y recién después la comilla. Al revés, la barra que
    protege a la comilla se duplicaría, y la comilla quedaría suelta.
    """
    visible = "".join(c if c.isprintable() else " " for c in texto)
    limpio = " ".join(visible.split())[:MAX_CARACTERES_DE_TEXTO].strip()
    return limpio.replace("\\", "\\\\").replace("'", "\\'")


def _consulta(literal: str, tipo: TipoDeArchivo | None, desde: datetime | None) -> str:
    """La `q` de Drive: las condiciones las pone este código, el texto es sólo un literal."""
    condiciones = ["trashed = false"]
    if literal:
        condiciones.append(f"(name contains '{literal}' or fullText contains '{literal}')")
    if tipo is TipoDeArchivo.IMAGEN:
        condiciones.append(f"mimeType contains '{_PREFIJO_DE_IMAGEN}'")
    elif tipo is not None:
        formatos = " or ".join(f"mimeType = '{formato}'" for formato in _FORMATOS[tipo])
        condiciones.append(f"({formatos})")
    if desde is not None:
        condiciones.append(f"modifiedTime >= '{desde.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ}'")
    return " and ".join(condiciones)


# --- De JSON a entidad -----------------------------------------------------------


def _a_archivo(item: dict[str, Any]) -> Archivo | None:
    """Traduce un archivo de la respuesta; None si le falta el id o la fecha.

    Drive siempre manda los dos: un item sin ellos no es un archivo de verdad,
    y mostrarlo con una fecha inventada engañaría a quien pregunta qué tocó.
    """
    identificador = item.get("id")
    modificado = _fecha(item.get("modifiedTime"))
    if not isinstance(identificador, str) or not identificador or modificado is None:
        return None
    formato = str(item.get("mimeType") or "")
    enlace = item.get("webViewLink")
    return Archivo(
        id=identificador,
        nombre=str(item.get("name") or "(sin nombre)"),
        tipo=(
            TipoDeArchivo.IMAGEN
            if formato.startswith(_PREFIJO_DE_IMAGEN)
            else _TIPO_POR_FORMATO.get(formato)
        ),
        modificado=modificado,
        enlace=enlace if isinstance(enlace, str) and enlace else None,
        propietario=_propietario(item.get("owners")),
        propio=_es_propio(item.get("owners")),
        compartido=_compartido(item),
    )


def _fecha(valor: Any) -> datetime | None:
    """`modifiedTime` viene en RFC 3339 y en UTC (`2026-10-12T21:30:00.000Z`)."""
    if not isinstance(valor, str):
        return None
    try:
        momento = datetime.fromisoformat(valor)
    except ValueError:
        return None
    return momento if momento.tzinfo is not None else None


def _propietario(duenos: Any) -> str | None:
    """De quién es el archivo si NO es de la persona; None si es suyo o no se sabe."""
    lista = [d for d in duenos if isinstance(d, dict)] if isinstance(duenos, list) else []
    if not lista or any(d.get("me") is True for d in lista):
        return None
    return str(lista[0].get("displayName") or "otra persona")


def _es_propio(duenos: Any) -> bool:
    """Sólo con la marca `me` de Drive: un dueño que no se informa no es la persona."""
    return isinstance(duenos, list) and any(
        isinstance(d, dict) and d.get("me") is True for d in duenos
    )


def _compartido(item: dict[str, Any]) -> bool:
    """Si alguien más lo ve: la marca `shared`, o más de un permiso.

    La marca sola no alcanza: una subcarpeta de una carpeta compartida hereda
    el acceso sin tenerla, pero sus permisos sí muestran a esa gente.
    """
    permisos = item.get("permissionIds")
    return item.get("shared") is True or (isinstance(permisos, list) and len(permisos) > 1)


def create_drive_google(cliente: httpx.AsyncClient, conectar: ConectarGoogle) -> DriveGoogle:
    """Factory del adaptador; el composition root la llama."""
    return DriveGoogle(cliente, conectar)
