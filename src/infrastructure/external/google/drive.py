"""Adaptador de Google Drive: buscar y listar archivos (PB-037).

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
- **Permiso mínimo**: `drive.metadata.readonly` alcanza para todo esto, y no
  deja abrir ni tocar ningún archivo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import structlog

from src.application.ports.archivos import Archivos
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.archivo import Archivo
from src.domain.exceptions import CuentaNoConectadaError, ServiceUnavailableError
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo
from src.infrastructure.external.google.transporte import json_o_vacio, traducir_rechazo

logger = structlog.get_logger(__name__)

BASE = "https://www.googleapis.com/drive/v3"
CANTIDAD_MAXIMA = 10
# Lo justo para mostrar y abrir cada archivo: ningún otro campo, y nada del contenido.
CAMPOS = "files(id,name,mimeType,modifiedTime,webViewLink,owners(displayName,me))"
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

    # --- Plomería ---------------------------------------------------------

    async def _credencial(self, usuario_id: UUID) -> str:
        token = await self._conectar.credencial_vigente(usuario_id, datetime.now(UTC))
        if token is None:
            raise CuentaNoConectadaError
        return token.access_token

    async def _pedir(self, url: str, token: str, parametros: dict[str, str]) -> dict[str, Any]:
        try:
            respuesta = await self._cliente.get(
                url, params=parametros, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            logger.error("drive.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Google Drive.") from None

        traducir_rechazo(respuesta, "drive")
        return json_o_vacio(respuesta)


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


def create_drive_google(cliente: httpx.AsyncClient, conectar: ConectarGoogle) -> DriveGoogle:
    """Factory del adaptador; el composition root la llama."""
    return DriveGoogle(cliente, conectar)
