"""Adaptador de lectura de Notion: buscar, leer páginas y consultar bases (PB-014).

Implementa el puerto `Paginas` contra la API pública de Notion (versión
`2026-03-11`, verificada en la referencia el 27/10/2026), con el token de la
persona que guarda `ConectarNotion`.

Lo que conviene saber antes de tocarlo:

- **Sólo ve lo que la persona compartió** al conectar (el selector de páginas de
  Notion). Lo demás contesta 404 `object_not_found`, igual que lo que no existe.
- **La búsqueda es por título**, no por contenido: la API no ofrece otra.
- **Las bases se consultan por "data source"** (desde la versión 2025-09-03): la
  búsqueda devuelve data sources, y el id que ve el modelo es el de ése.
- **El token no informa vencimiento**: ante un 401 se renueva una sola vez
  (`ConectarNotion.renovar`) y se reintenta el pedido.
- **El parser nunca lanza**: un item que no se entiende se descarta. Una
  excepción en una tool tumba el turno entero (`ToolNode` la re-lanza).
- **Los ids los pasa el modelo** y terminan en una URL: se validan con una
  expresión estricta antes de armarla (la lección de PB-036). `uuid.UUID` no
  sirve: acepta llaves, `urn:uuid:`, `0x` y guiones en cualquier lado.
- **Un 403 es nuestra configuración** (a la conexión le falta una capacidad), no
  la cuenta de la persona: reconectar no lo arregla, así que no manda a
  /conectar.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import UUID

import httpx
import structlog

from src.application.ports.paginas import BaseConsultada, FilaDeBase, PaginaLeida, Paginas
from src.application.use_cases.conectar_notion import ConectarNotion
from src.domain.entities.pagina import Pagina
from src.domain.exceptions import (
    AutorizacionFallidaError,
    EntityNotFoundError,
    ServiceUnavailableError,
)
from src.domain.value_objects.tipo_de_pagina import TipoDePagina
from src.infrastructure.external.notion.transporte import ENCABEZADOS, codigo_de, json_o_vacio

logger = structlog.get_logger(__name__)

BASE = "https://api.notion.com/v1"
CANTIDAD_MAXIMA = 10
FILAS_MAXIMAS = 20
PROPIEDADES_POR_FILA = 6
# De sobra para unas palabras de un título: un texto enorme no viaja.
MAX_CARACTERES_DE_TEXTO = 100
_RECIENTES = {"timestamp": "last_edited_time", "direction": "descending"}
# Las dos formas de un id de Notion: 32 hex, o 8-4-4-4-12 con guiones.
_ID_DE_NOTION = re.compile(
    r"[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


class NotionApi(Paginas):
    """El puerto `Paginas` hablando con la API real de Notion."""

    def __init__(self, cliente: httpx.AsyncClient, conectar: ConectarNotion) -> None:
        self._cliente = cliente
        self._conectar = conectar

    async def buscar(self, usuario_id: UUID, texto: str, cantidad: int) -> tuple[Pagina, ...]:
        cuerpo: dict[str, Any] = {"page_size": max(1, min(cantidad, CANTIDAD_MAXIMA))}
        consulta = " ".join(texto.split())[:MAX_CARACTERES_DE_TEXTO]
        if consulta:
            cuerpo["query"] = consulta
        else:
            cuerpo["sort"] = _RECIENTES  # sin texto, "listar": las últimas editadas
        datos = await self._pedir(usuario_id, "POST", f"{BASE}/search", cuerpo)
        paginas = tuple(
            pagina for item in _lista(datos.get("results")) if (pagina := _a_pagina(item))
        )
        # Cantidades, nunca títulos (RF-18).
        logger.info("notion.buscadas", usuario_id=str(usuario_id), cantidad=len(paginas))
        return paginas

    async def leer(self, usuario_id: UUID, pagina_id: str) -> PaginaLeida:
        url = f"{BASE}/pages/{_id_valido(pagina_id)}"  # antes que nada: lo pasa el modelo
        pagina = _a_pagina(await self._pedir(usuario_id, "GET", url))
        if pagina is None or pagina.tipo is not TipoDePagina.PAGINA:
            raise EntityNotFoundError("Notion no devolvió una página.")
        # El Markdown no trae el título: sale del pedido de arriba.
        contenido = await self._pedir(usuario_id, "GET", f"{url}/markdown")
        markdown = contenido.get("markdown")
        texto = markdown if isinstance(markdown, str) else ""
        recortada = contenido.get("truncated") is True
        logger.info(
            "notion.leida", usuario_id=str(usuario_id), caracteres=len(texto), recortada=recortada
        )
        return PaginaLeida(pagina=pagina, markdown=texto, recortada=recortada)

    async def consultar(self, usuario_id: UUID, base_id: str, cantidad: int) -> BaseConsultada:
        url = f"{BASE}/data_sources/{_id_valido(base_id)}"
        base = _a_pagina(await self._pedir(usuario_id, "GET", url))
        if base is None or base.tipo is not TipoDePagina.BASE:
            raise EntityNotFoundError("Notion no devolvió una base.")
        cuerpo = {"page_size": max(1, min(cantidad, FILAS_MAXIMAS)), "sorts": [_RECIENTES]}
        datos = await self._pedir(usuario_id, "POST", f"{url}/query", cuerpo)
        filas = tuple(fila for item in _lista(datos.get("results")) if (fila := _a_fila(item)))
        hay_mas = datos.get("has_more") is True
        logger.info(
            "notion.consultada", usuario_id=str(usuario_id), filas=len(filas), hay_mas=hay_mas
        )
        return BaseConsultada(base=base, filas=filas, hay_mas=hay_mas)

    # --- Plomería ---------------------------------------------------------

    async def _pedir(
        self, usuario_id: UUID, metodo: str, url: str, cuerpo: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Un pedido con el token de la persona; ante un 401, renueva y reintenta una vez."""
        token = await self._conectar.credencial(usuario_id)
        respuesta = await self._enviar(metodo, url, token.access_token, cuerpo)
        if respuesta.status_code == httpx.codes.UNAUTHORIZED:
            logger.info("notion.token_rechazado", usuario_id=str(usuario_id))
            token = await self._conectar.renovar(usuario_id, token.access_token)
            respuesta = await self._enviar(metodo, url, token.access_token, cuerpo)
            if respuesta.status_code == httpx.codes.UNAUTHORIZED:
                logger.warning("notion.token_rechazado_tras_renovar", usuario_id=str(usuario_id))
                raise AutorizacionFallidaError("Notion rechazó también la credencial renovada.")
        _revisar(respuesta)
        return json_o_vacio(respuesta)

    async def _enviar(
        self, metodo: str, url: str, token: str, cuerpo: dict[str, Any] | None
    ) -> httpx.Response:
        try:
            return await self._cliente.request(
                metodo,
                url,
                json=cuerpo,
                headers={"Authorization": f"Bearer {token}", **ENCABEZADOS},
            )
        except httpx.HTTPError as exc:
            logger.error("notion.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Notion.") from None


def _revisar(respuesta: httpx.Response) -> None:
    """Cada rechazo con su remedio. Se loguea el código, nunca el mensaje de Notion."""
    if respuesta.status_code < httpx.codes.BAD_REQUEST:
        return
    codigo = codigo_de(respuesta)
    if respuesta.status_code == httpx.codes.NOT_FOUND:
        logger.info("notion.no_encontrado", codigo=codigo)
        raise EntityNotFoundError("No está en el Notion de la persona, o no está compartido.")
    if respuesta.status_code == httpx.codes.TOO_MANY_REQUESTS:
        logger.warning("notion.limite_de_uso", reintentar_en=respuesta.headers.get("Retry-After"))
        raise ServiceUnavailableError("Notion limitó los pedidos por un rato.")
    if respuesta.status_code == httpx.codes.FORBIDDEN:
        logger.error("notion.sin_capacidad", codigo=codigo)
        raise ServiceUnavailableError("A la conexión de Notion le falta una capacidad.")
    logger.error("notion.rechazado", status_code=respuesta.status_code, codigo=codigo)
    raise ServiceUnavailableError("Notion no pudo responder.")


def _id_valido(identificador: str) -> str:
    """El id en su forma canónica (minúsculas, con guiones), sólo si es de Notion.

    Raises:
        EntityNotFoundError: Si no tiene la forma de un id de Notion: algo así no
            existe, y armar la URL con eso sacaría el pedido del recurso.
    """
    if not _ID_DE_NOTION.fullmatch(identificador):
        # El largo y nada más: el id lo escribió el modelo, quizás inducido.
        logger.warning("notion.id_invalido", largo=len(identificador))
        raise EntityNotFoundError("Eso no es un id de Notion.")
    hexa = identificador.replace("-", "").lower()
    return f"{hexa[:8]}-{hexa[8:12]}-{hexa[12:16]}-{hexa[16:20]}-{hexa[20:]}"


# --- De JSON a entidad -----------------------------------------------------------


def _lista(valor: Any) -> list[Any]:
    return valor if isinstance(valor, list) else []


def _a_pagina(item: Any) -> Pagina | None:
    """Una página o un data source de la respuesta; None si no se entiende.

    El título de una página es la propiedad de tipo `title` (se busca por tipo:
    el nombre lo elige quien la creó); el de un data source, su `title`.
    """
    if not isinstance(item, dict):
        return None
    identificador = item.get("id")
    editada = _fecha(item.get("last_edited_time"))
    if not isinstance(identificador, str) or not identificador or editada is None:
        return None
    objeto = item.get("object")
    if objeto == "page":
        tipo, titulo = TipoDePagina.PAGINA, _titulo_de(item.get("properties"))
    elif objeto in ("data_source", "database"):
        tipo, titulo = TipoDePagina.BASE, _texto(item.get("title"))
    else:
        return None
    return Pagina(
        id=identificador,
        titulo=titulo or "(sin título)",
        tipo=tipo,
        editada=editada,
        enlace=_enlace(item.get("url")),
    )


def _a_fila(item: Any) -> FilaDeBase | None:
    """Una fila de la consulta (una página de la base), con sus propiedades en texto."""
    if not isinstance(item, dict) or item.get("object") != "page":
        return None
    propiedades = item.get("properties")
    propiedades = propiedades if isinstance(propiedades, dict) else {}
    valores: list[tuple[str, str]] = []
    for nombre, propiedad in propiedades.items():
        if len(valores) >= PROPIEDADES_POR_FILA:
            break
        if not isinstance(propiedad, dict) or propiedad.get("type") == "title":
            continue
        valor = _valor_de(propiedad)
        if valor:
            valores.append((str(nombre), valor))
    return FilaDeBase(
        titulo=_titulo_de(propiedades) or "(sin título)",
        propiedades=tuple(valores),
        enlace=_enlace(item.get("url")),
    )


def _titulo_de(propiedades: Any) -> str:
    if not isinstance(propiedades, dict):
        return ""
    for propiedad in propiedades.values():
        if isinstance(propiedad, dict) and propiedad.get("type") == "title":
            return _texto(propiedad.get("title"))
    return ""


def _texto(partes: Any) -> str:
    """Un texto enriquecido de Notion (una lista de partes) a texto plano."""
    return "".join(
        parte["plain_text"]
        for parte in _lista(partes)
        if isinstance(parte, dict) and isinstance(parte.get("plain_text"), str)
    ).strip()


def _valor_de(propiedad: dict[str, Any]) -> str:
    """El valor de una propiedad en texto; vacío si es de un tipo que no se muestra.

    Las personas pueden venir sin nombre: con "No user information", Notion no
    lo manda. Entonces no se muestran.
    """
    tipo = propiedad.get("type")
    dato = propiedad.get(tipo) if isinstance(tipo, str) else None
    if tipo == "rich_text":
        return _texto(dato)
    if tipo == "number":
        return _numero(dato)
    if tipo in ("select", "status"):
        return _nombre(dato)
    if tipo == "multi_select":
        return ", ".join(nombre for opcion in _lista(dato) if (nombre := _nombre(opcion)))
    if tipo == "people":
        return ", ".join(nombre for persona in _lista(dato) if (nombre := _nombre(persona)))
    if tipo == "date":
        return _rango(dato)
    if tipo == "checkbox":
        return {True: "sí", False: "no"}.get(dato, "") if isinstance(dato, bool) else ""
    if tipo in ("url", "email", "phone_number", "created_time", "last_edited_time"):
        return dato if isinstance(dato, str) else ""
    if tipo == "formula" and isinstance(dato, dict) and dato.get("type") != "formula":
        return _valor_de(dato)  # trae su propio `type`: string, number, boolean o date
    if tipo in ("string", "boolean"):  # los resultados de una fórmula
        return dato if isinstance(dato, str) else _valor_de({"type": "checkbox", "checkbox": dato})
    return ""


def _numero(dato: Any) -> str:
    if isinstance(dato, bool) or not isinstance(dato, int | float):
        return ""
    return str(int(dato)) if isinstance(dato, float) and dato.is_integer() else str(dato)


def _nombre(dato: Any) -> str:
    nombre = dato.get("name") if isinstance(dato, dict) else None
    return nombre if isinstance(nombre, str) else ""


def _rango(dato: Any) -> str:
    if not isinstance(dato, dict) or not isinstance(dato.get("start"), str):
        return ""
    fin = dato.get("end")
    return f"{dato['start']} → {fin}" if isinstance(fin, str) else dato["start"]


def _fecha(valor: Any) -> datetime | None:
    """Las fechas de Notion vienen en ISO 8601 y en UTC (`2026-10-12T21:30:00.000Z`)."""
    if not isinstance(valor, str):
        return None
    try:
        momento = datetime.fromisoformat(valor)
    except ValueError:
        return None
    return momento if momento.tzinfo is not None else None


def _enlace(valor: Any) -> str | None:
    """El link que da Notion, sólo si es https. Un data source puede no traerlo."""
    return valor if isinstance(valor, str) and valor.startswith("https://") else None


def create_notion_api(cliente: httpx.AsyncClient, conectar: ConectarNotion) -> NotionApi:
    """Factory del adaptador; el composition root la llama."""
    return NotionApi(cliente, conectar)
