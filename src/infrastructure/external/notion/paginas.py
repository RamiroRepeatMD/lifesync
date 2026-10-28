"""Adaptador de Notion: leer (PB-014) y escribir (RF-07).

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
- **Al leer, un 403 es nuestra configuración** (a la conexión le falta una
  capacidad), no la cuenta de la persona: no manda a /conectar.
- **Al escribir, cada rechazo tiene su remedio** (RF-07): un 403 puede ser un
  token de antes de que la conexión pudiera escribir (→ /conectar notion), un
  400 es un dato que Notion no acepta, y un 5xx o una respuesta perdida dejan
  la escritura **en duda**: Notion avisa que un cambio puede guardarse aunque
  la respuesta sea un error.
- **Se escribe sólo con bloques propios**, nunca con el Markdown de Notion, que
  admite menciones, embeds y bases.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any
from uuid import UUID

import httpx
import structlog

from src.application.ports.paginas import (
    BaseConsultada,
    Bloque,
    FilaDeBase,
    PaginaLeida,
    Paginas,
    Propiedad,
    TipoDeBloque,
    TipoDePropiedad,
    ValorDePropiedad,
)
from src.application.use_cases.conectar_notion import ConectarNotion
from src.domain.entities.pagina import Pagina
from src.domain.exceptions import (
    AutorizacionFallidaError,
    EntityNotFoundError,
    EscrituraEnDudaError,
    NotionRechazoError,
    NotionSinPermisoError,
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
# Notion corta cada texto en 2000 caracteres contados en UTF-16: con 1900 sobra.
LARGO_DEL_TROZO = 1900
# Una subida grande tarda más que una consulta (el tope es de 20 MB).
TIMEOUT_DE_SUBIDA = httpx.Timeout(60.0, connect=5.0)
# Lo que garantiza que el pedido no llegó: después de esto, nada quedó en duda.
_SIN_CONEXION = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_JSON_DE_BLOQUE = {
    TipoDeBloque.PARRAFO: "paragraph",
    TipoDeBloque.TITULO_1: "heading_1",
    TipoDeBloque.TITULO_2: "heading_2",
    TipoDeBloque.TITULO_3: "heading_3",
    TipoDeBloque.VINETA: "bulleted_list_item",
    TipoDeBloque.NUMERADO: "numbered_list_item",
    TipoDeBloque.TAREA: "to_do",
}
_TIPO_DE_PROPIEDAD = {
    "title": TipoDePropiedad.TITULO,
    "rich_text": TipoDePropiedad.TEXTO,
    "number": TipoDePropiedad.NUMERO,
    "checkbox": TipoDePropiedad.CASILLA,
    "date": TipoDePropiedad.FECHA,
    "select": TipoDePropiedad.SELECCION,
    "status": TipoDePropiedad.ESTADO,
}
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

    # --- Escribir (RF-07) --------------------------------------------------

    async def obtener(self, usuario_id: UUID, pagina_id: str) -> Pagina:
        identificador = _id_valido(pagina_id)
        respuesta = await self._respuesta(usuario_id, "GET", f"{BASE}/pages/{identificador}")
        if respuesta.status_code in (httpx.codes.NOT_FOUND, httpx.codes.BAD_REQUEST):
            # Puede ser una base: el id de un data source no es el de una página,
            # y la doc no dice si Notion contesta 404 o 400 al pedirlo así.
            url = f"{BASE}/data_sources/{identificador}"
            respuesta = await self._respuesta(usuario_id, "GET", url)
        _revisar(respuesta)
        pagina = _a_pagina(json_o_vacio(respuesta))
        if pagina is None:
            raise EntityNotFoundError("Notion no devolvió una página ni una base.")
        return pagina

    async def propiedades(
        self, usuario_id: UUID, pagina_id: str
    ) -> tuple[Pagina, tuple[Propiedad, ...]]:
        datos = await self._pedir(usuario_id, "GET", f"{BASE}/pages/{_id_valido(pagina_id)}")
        pagina = _a_pagina(datos)
        if pagina is None or pagina.tipo is not TipoDePagina.PAGINA:
            raise EntityNotFoundError("Notion no devolvió una página.")
        opciones: dict[str, tuple[str, ...]] = {}
        base_id = _base_de(datos)
        if base_id is not None:
            # Las opciones de selección y estado viven en el esquema de la base.
            esquema = await self._pedir(usuario_id, "GET", f"{BASE}/data_sources/{base_id}")
            opciones = _opciones_del_esquema(esquema)
        return pagina, _propiedades_editables(datos.get("properties"), opciones)

    async def crear(
        self, usuario_id: UUID, padre: Pagina, titulo: str, bloques: tuple[Bloque, ...]
    ) -> Pagina:
        clave = "data_source_id" if padre.tipo is TipoDePagina.BASE else "page_id"
        cuerpo: dict[str, Any] = {
            "parent": {"type": clave, clave: _id_valido(padre.id)},
            # Por el id `title`: vale para una página suelta y para la columna de
            # título de una base, se llame como se llame.
            "properties": {"title": {"title": _texto_enriquecido(titulo)}},
        }
        if bloques:
            cuerpo["children"] = [_bloque_json(bloque) for bloque in bloques]
        datos = await self._pedir(usuario_id, "POST", f"{BASE}/pages", cuerpo, escritura=True)
        creada = _a_pagina(datos)
        if creada is None:
            logger.error("notion.creada_sin_respuesta")
            raise EscrituraEnDudaError("Notion no devolvió la página creada.")
        logger.info(
            "notion.creada",
            usuario_id=str(usuario_id),
            fila=padre.tipo is TipoDePagina.BASE,
            bloques=len(bloques),
        )
        return creada

    async def agregar(self, usuario_id: UUID, pagina_id: str, bloques: tuple[Bloque, ...]) -> None:
        url = f"{BASE}/blocks/{_id_valido(pagina_id)}/children"
        cuerpo = {"children": [_bloque_json(bloque) for bloque in bloques]}
        await self._pedir(usuario_id, "PATCH", url, cuerpo, escritura=True)
        logger.info("notion.agregado", usuario_id=str(usuario_id), bloques=len(bloques))

    async def cambiar(
        self, usuario_id: UUID, pagina_id: str, propiedad: Propiedad, valor: ValorDePropiedad
    ) -> None:
        url = f"{BASE}/pages/{_id_valido(pagina_id)}"
        # Por el id de la propiedad: no cambia si alguien la renombra.
        cuerpo = {"properties": {propiedad.id: _valor_json(propiedad.tipo, valor)}}
        await self._pedir(usuario_id, "PATCH", url, cuerpo, escritura=True)
        logger.info("notion.cambiado", usuario_id=str(usuario_id), tipo=str(propiedad.tipo))

    async def mover(self, usuario_id: UUID, pagina_id: str, destino_id: str) -> None:
        url = f"{BASE}/pages/{_id_valido(pagina_id)}/move"
        cuerpo = {"parent": {"type": "page_id", "page_id": _id_valido(destino_id)}}
        await self._pedir(usuario_id, "POST", url, cuerpo, escritura=True)
        logger.info("notion.movida", usuario_id=str(usuario_id))

    async def a_la_papelera(self, usuario_id: UUID, pagina_id: str) -> None:
        url = f"{BASE}/pages/{_id_valido(pagina_id)}"
        await self._pedir(usuario_id, "PATCH", url, {"in_trash": True}, escritura=True)
        logger.info("notion.a_la_papelera", usuario_id=str(usuario_id))

    async def limite_de_subida(self, usuario_id: UUID) -> int | None:
        # Del bot no se loguea nada: su `owner` puede traer el correo de la persona.
        datos = await self._pedir(usuario_id, "GET", f"{BASE}/users/me")
        bot = datos.get("bot")
        limites = bot.get("workspace_limits") if isinstance(bot, dict) else None
        valor = limites.get("max_file_upload_size_in_bytes") if isinstance(limites, dict) else None
        valido = isinstance(valor, int) and not isinstance(valor, bool) and valor > 0
        return valor if valido else None

    async def subir(
        self, usuario_id: UUID, pagina_id: str, nombre: str, tipo_mime: str, contenido: bytes
    ) -> None:
        """Crear la subida → mandar los bytes → agregarla al final de la página.

        Sólo el último paso cambia la página: una subida que no se adjunta vence
        sola en una hora. Por eso un corte antes de adjuntar no deja nada en duda.
        """
        url_de_la_pagina = f"{BASE}/blocks/{_id_valido(pagina_id)}/children"
        tipo = tipo_mime.split(";")[0].strip().lower() or "application/octet-stream"
        try:
            creada = await self._pedir(
                usuario_id,
                "POST",
                f"{BASE}/file_uploads",
                {"mode": "single_part", "filename": nombre, "content_type": tipo},
                escritura=True,
            )
            subida = creada.get("id")
            if not isinstance(subida, str) or not _ID_DE_NOTION.fullmatch(subida):
                logger.error("notion.subida_sin_id")
                raise ServiceUnavailableError("Notion no abrió la subida.")
            # La URL se arma con el id validado, nunca con algo de la respuesta.
            subida = _id_valido(subida)
            await self._pedir(
                usuario_id,
                "POST",
                f"{BASE}/file_uploads/{subida}/send",
                archivos={"file": (nombre, contenido, tipo)},
                escritura=True,
                tiempo=TIMEOUT_DE_SUBIDA,
            )
        except EscrituraEnDudaError:
            raise ServiceUnavailableError("La subida a Notion no terminó.") from None

        bloque = _bloque_de_archivo(tipo)
        adjunto = {"type": bloque, bloque: {"type": "file_upload", "file_upload": {"id": subida}}}
        await self._pedir(
            usuario_id, "PATCH", url_de_la_pagina, {"children": [adjunto]}, escritura=True
        )
        # El tamaño y el tipo de bloque, nunca el nombre (RF-18).
        logger.info(
            "notion.subido", usuario_id=str(usuario_id), bytes=len(contenido), bloque=bloque
        )

    # --- Plomería ---------------------------------------------------------

    async def _pedir(
        self,
        usuario_id: UUID,
        metodo: str,
        url: str,
        cuerpo: dict[str, Any] | None = None,
        *,
        escritura: bool = False,
        archivos: dict[str, tuple[str, bytes, str]] | None = None,
        tiempo: httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        """Un pedido ya revisado: el cuerpo de la respuesta, o la excepción que toca."""
        respuesta = await self._respuesta(
            usuario_id, metodo, url, cuerpo, escritura=escritura, archivos=archivos, tiempo=tiempo
        )
        _revisar(respuesta, escritura=escritura)
        return json_o_vacio(respuesta)

    async def _respuesta(
        self,
        usuario_id: UUID,
        metodo: str,
        url: str,
        cuerpo: dict[str, Any] | None = None,
        *,
        escritura: bool = False,
        archivos: dict[str, tuple[str, bytes, str]] | None = None,
        tiempo: httpx.Timeout | None = None,
    ) -> httpx.Response:
        """Un pedido con el token de la persona; ante un 401, renueva y reintenta una vez."""
        token = await self._conectar.credencial(usuario_id)
        respuesta = await self._enviar(
            metodo, url, token.access_token, cuerpo, escritura, archivos, tiempo
        )
        if respuesta.status_code == httpx.codes.UNAUTHORIZED:
            logger.info("notion.token_rechazado", usuario_id=str(usuario_id))
            token = await self._conectar.renovar(usuario_id, token.access_token)
            respuesta = await self._enviar(
                metodo, url, token.access_token, cuerpo, escritura, archivos, tiempo
            )
            if respuesta.status_code == httpx.codes.UNAUTHORIZED:
                logger.warning("notion.token_rechazado_tras_renovar", usuario_id=str(usuario_id))
                raise AutorizacionFallidaError("Notion rechazó también la credencial renovada.")
        return respuesta

    async def _enviar(
        self,
        metodo: str,
        url: str,
        token: str,
        cuerpo: dict[str, Any] | None,
        escritura: bool,
        archivos: dict[str, tuple[str, bytes, str]] | None,
        tiempo: httpx.Timeout | None,
    ) -> httpx.Response:
        try:
            return await self._cliente.request(
                metodo,
                url,
                json=cuerpo,
                files=archivos,
                headers={"Authorization": f"Bearer {token}", **ENCABEZADOS},
                timeout=tiempo if tiempo is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except _SIN_CONEXION as exc:
            # El pedido no llegó: seguro que no se hizo nada.
            logger.error("notion.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Notion.") from None
        except httpx.HTTPError as exc:
            logger.error("notion.error_transporte", tipo=type(exc).__name__, escritura=escritura)
            if escritura:
                # El pedido salió y la respuesta no volvió: pudo haberse hecho.
                raise EscrituraEnDudaError("Notion no confirmó la escritura.") from None
            raise ServiceUnavailableError("No se pudo contactar a Notion.") from None


def _revisar(respuesta: httpx.Response, *, escritura: bool = False) -> None:
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
    if escritura:
        _revisar_escritura(respuesta, codigo)
    if respuesta.status_code == httpx.codes.FORBIDDEN:
        logger.error("notion.sin_capacidad", codigo=codigo)
        raise ServiceUnavailableError("A la conexión de Notion le falta una capacidad.")
    logger.error("notion.rechazado", status_code=respuesta.status_code, codigo=codigo)
    raise ServiceUnavailableError("Notion no pudo responder.")


def _revisar_escritura(respuesta: httpx.Response, codigo: str | None) -> None:
    """Los rechazos de una escritura (RF-07): cada uno dice si se hizo algo."""
    if respuesta.status_code == httpx.codes.FORBIDDEN:
        logger.warning("notion.escritura_sin_permiso", codigo=codigo)
        raise NotionSinPermisoError("Notion no dejó escribir.")
    if respuesta.status_code == httpx.codes.BAD_REQUEST:
        logger.warning("notion.escritura_rechazada", codigo=codigo)
        raise NotionRechazoError("Notion no aceptó la escritura.")
    if respuesta.status_code == httpx.codes.CONFLICT:
        logger.warning("notion.escritura_en_conflicto", codigo=codigo)
        raise ServiceUnavailableError("Notion pidió repetir la escritura.")
    if respuesta.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
        logger.error("notion.escritura_en_duda", status_code=respuesta.status_code, codigo=codigo)
        raise EscrituraEnDudaError("Notion no confirmó la escritura.")


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
    if any(item.get(clave) is True for clave in ("in_trash", "archived", "is_archived")):
        return None  # lo que está en la papelera, para el bot no está
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
    padre = item.get("parent")
    en_base = isinstance(padre, dict) and padre.get("type") in ("data_source_id", "database_id")
    return Pagina(
        id=identificador,
        titulo=titulo or "(sin título)",
        tipo=tipo,
        editada=editada,
        enlace=_enlace(item.get("url")),
        en_base=en_base and tipo is TipoDePagina.PAGINA,
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
    identificador = item.get("id")
    return FilaDeBase(
        titulo=_titulo_de(propiedades) or "(sin título)",
        propiedades=tuple(valores),
        enlace=_enlace(item.get("url")),
        id=identificador if isinstance(identificador, str) else "",
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


# --- De entidad a JSON (RF-07) -------------------------------------------------------


def _texto_enriquecido(texto: str) -> list[dict[str, Any]]:
    """Un texto en partes de `LARGO_DEL_TROZO`: Notion corta cada parte en 2000."""
    trozos = [texto[i : i + LARGO_DEL_TROZO] for i in range(0, len(texto), LARGO_DEL_TROZO)]
    return [{"type": "text", "text": {"content": trozo}} for trozo in trozos or [""]]


def _bloque_json(bloque: Bloque) -> dict[str, Any]:
    tipo = _JSON_DE_BLOQUE[bloque.tipo]
    contenido: dict[str, Any] = {"rich_text": _texto_enriquecido(bloque.texto)}
    if bloque.tipo is TipoDeBloque.TAREA:
        contenido["checked"] = bloque.hecho
    return {"object": "block", "type": tipo, tipo: contenido}


def _bloque_de_archivo(tipo_mime: str) -> str:
    """Cómo se muestra en la página: imagen, PDF o archivo a secas."""
    if tipo_mime.startswith("image/"):
        return "image"
    return "pdf" if tipo_mime == "application/pdf" else "file"


def _valor_json(tipo: TipoDePropiedad, valor: ValorDePropiedad) -> dict[str, Any]:
    """El valor ya validado, en la forma de Notion para su tipo.

    Raises:
        NotionRechazoError: Si el valor no es del tipo que corresponde (no debería
            pasar: la tool lo valida antes), o es un número que no se puede mandar.
    """
    if tipo in (TipoDePropiedad.TITULO, TipoDePropiedad.TEXTO) and isinstance(valor, str):
        clave = "title" if tipo is TipoDePropiedad.TITULO else "rich_text"
        return {clave: _texto_enriquecido(valor)}
    if tipo is TipoDePropiedad.NUMERO and isinstance(valor, int | float):
        if isinstance(valor, bool) or not math.isfinite(valor):
            raise NotionRechazoError("Ese número no se puede guardar.")
        return {"number": int(valor) if float(valor).is_integer() else valor}
    if tipo is TipoDePropiedad.CASILLA and isinstance(valor, bool):
        return {"checkbox": valor}
    if tipo is TipoDePropiedad.FECHA and isinstance(valor, date):
        return {"date": {"start": valor.isoformat()}}
    if tipo in (TipoDePropiedad.SELECCION, TipoDePropiedad.ESTADO) and isinstance(valor, str):
        clave = "select" if tipo is TipoDePropiedad.SELECCION else "status"
        return {clave: {"name": valor}}
    raise NotionRechazoError("El valor no corresponde a ese dato.")


def _base_de(pagina: dict[str, Any]) -> str | None:
    """El data source de una fila, para leer las opciones de su esquema; None si no es fila."""
    padre = pagina.get("parent")
    if not isinstance(padre, dict) or padre.get("type") != "data_source_id":
        return None
    identificador = padre.get("data_source_id")
    if not isinstance(identificador, str) or not _ID_DE_NOTION.fullmatch(identificador):
        return None
    return _id_valido(identificador)


def _opciones_del_esquema(esquema: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Las opciones de cada propiedad de selección o estado, por el id de la propiedad."""
    propiedades = esquema.get("properties")
    opciones: dict[str, tuple[str, ...]] = {}
    for propiedad in (propiedades if isinstance(propiedades, dict) else {}).values():
        if not isinstance(propiedad, dict) or not isinstance(propiedad.get("id"), str):
            continue
        tipo = propiedad.get("type")
        datos = propiedad.get(tipo) if tipo in ("select", "status") else None
        lista = datos.get("options") if isinstance(datos, dict) else None
        opciones[propiedad["id"]] = tuple(
            nombre for opcion in _lista(lista) if (nombre := _nombre(opcion))
        )
    return opciones


def _propiedades_editables(
    propiedades: Any, opciones: dict[str, tuple[str, ...]]
) -> tuple[Propiedad, ...]:
    """Las propiedades de una página que se pueden cambiar, con su valor actual."""
    editables: list[Propiedad] = []
    for nombre, propiedad in (propiedades if isinstance(propiedades, dict) else {}).items():
        if not isinstance(propiedad, dict) or not isinstance(propiedad.get("id"), str):
            continue
        tipo = _TIPO_DE_PROPIEDAD.get(str(propiedad.get("type")))
        if tipo is None:
            continue
        actual = (
            _texto(propiedad.get("title"))
            if tipo is TipoDePropiedad.TITULO
            else _valor_de(propiedad)
        )
        editables.append(
            Propiedad(
                id=propiedad["id"],
                nombre=str(nombre),
                tipo=tipo,
                valor=actual,
                opciones=opciones.get(propiedad["id"], ()),
            )
        )
    return tuple(editables)


def create_notion_api(cliente: httpx.AsyncClient, conectar: ConectarNotion) -> NotionApi:
    """Factory del adaptador; el composition root la llama."""
    return NotionApi(cliente, conectar)
