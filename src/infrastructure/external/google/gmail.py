"""Adaptador de Gmail: leer, buscar (PB-033), enviar (PB-032), responder y reenviar (PB-034).

Implementa el puerto `Correos` contra la API REST de Gmail, con el mismo trío
que Calendar y Tasks: el cliente httpx compartido de Google, las credenciales
vigentes de `ConectarGoogle` y la traducción de rechazos de `transporte`.

Lo que conviene saber antes de tocarlo:

- **Buscar son 1 + N pedidos.** `messages.list` devuelve sólo ids; remitente,
  asunto y fecha salen de un `messages.get` por correo, que se piden en
  paralelo (como los calendarios en PB-015).
- **El cuerpo es un árbol MIME.** Se prefiere `text/plain`; si sólo hay HTML
  se le extrae el texto con la stdlib. Los adjuntos se nombran, no se bajan.
- **El cuerpo se recorta** (`MAX_CARACTERES_DE_CUERPO`): cada turno reenvía el
  historial al modelo, y el contenido de un correo lo escribió un tercero:
  cuanto menos texto ajeno en el contexto, menos costo y menos superficie para
  una inyección.
- **Un envío que falla tiene que decir si salió** (PB-035). `_publicar` separa
  lo que seguro no salió (no se pudo conectar, Gmail lo rechazó o lo frenó por
  un límite) de lo que quedó en duda (el pedido llegó y la respuesta no, o
  Gmail falló a mitad de camino): sólo eso es `EnvioInciertoError`.
"""

from __future__ import annotations

import asyncio
import base64
import re
from datetime import UTC, datetime
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import getaddresses, parseaddr
from html import unescape
from html.parser import HTMLParser
from typing import Any
from uuid import UUID

import httpx
import structlog

from src.application.ports.correos import Correos
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.correo import Correo, CorreoSaliente
from src.domain.exceptions import (
    CuentaNoConectadaError,
    EntityNotFoundError,
    EnvioInciertoError,
    EnvioRechazadoError,
    ServiceUnavailableError,
)
from src.infrastructure.external.google.transporte import (
    json_o_vacio,
    motivo_de,
    traducir_rechazo,
)

logger = structlog.get_logger(__name__)

BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
CONSULTA_POR_DEFECTO = "in:inbox"
CANTIDAD_MAXIMA = 10
MAX_CARACTERES_DE_CUERPO = 2000
ENCABEZADOS_DE_LISTADO = ("From", "Subject")
# Las fallas en las que el pedido nunca llegó a Gmail: lo que se quiso mandar
# seguro no salió. Cualquier otra falla del POST deja el envío en duda.
_SIN_CONEXION = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
# La forma de un id de Gmail. El de `leer` lo pasa el modelo: armado tal cual
# en la URL, "../profile" pedía `/users/me/profile` (bug de PB-036). Un id sin
# esta forma no puede existir, así que ni se pide.
_ID_DE_GMAIL = re.compile(r"[A-Za-z0-9_-]{1,128}")

# Tupla y no lista: las tuplas son covariantes, y httpx tipa los valores
# más amplio que `str`.
Parametros = tuple[tuple[str, str], ...]


class GmailGoogle(Correos):
    """El puerto `Correos` hablando con la API real."""

    def __init__(self, cliente: httpx.AsyncClient, conectar: ConectarGoogle) -> None:
        self._cliente = cliente
        self._conectar = conectar

    async def buscar(self, usuario_id: UUID, consulta: str, cantidad: int) -> tuple[Correo, ...]:
        token = await self._credencial(usuario_id)
        cantidad = max(1, min(cantidad, CANTIDAD_MAXIMA))
        lista = await self._pedir(
            f"{BASE}/messages",
            token,
            (("q", consulta.strip() or CONSULTA_POR_DEFECTO), ("maxResults", str(cantidad))),
        )
        ids = [
            m["id"]
            for m in lista.get("messages") or []
            if isinstance(m, dict)
            and isinstance(m.get("id"), str)
            and _ID_DE_GMAIL.fullmatch(m["id"])
        ]
        metadatos: Parametros = (
            ("format", "metadata"),
            *(("metadataHeaders", encabezado) for encabezado in ENCABEZADOS_DE_LISTADO),
        )
        mensajes = await asyncio.gather(
            *(self._pedir(_url_del_mensaje(i), token, metadatos) for i in ids)
        )
        correos = tuple(_a_correo(mensaje) for mensaje in mensajes)
        # Cantidades, nunca remitentes ni asuntos (RF-18).
        logger.info("gmail.buscados", usuario_id=str(usuario_id), cantidad=len(correos))
        return correos

    async def leer(self, usuario_id: UUID, correo_id: str, *, completo: bool = False) -> Correo:
        url = _url_del_mensaje(correo_id)  # antes que nada: lo pasa el modelo
        token = await self._credencial(usuario_id)
        mensaje = await self._pedir(
            url,
            token,
            (("format", "full"),),
            error_404=EntityNotFoundError("Ese correo no existe o ya no está."),
        )
        correo = _a_correo(mensaje, con_cuerpo=True, completo=completo)
        logger.info("gmail.leido", usuario_id=str(usuario_id), adjuntos=len(correo.adjuntos))
        return correo

    async def enviar(self, usuario_id: UUID, saliente: CorreoSaliente) -> str:
        token = await self._credencial(usuario_id)
        mensaje = EmailMessage()
        # Sin `From`: Gmail pone la cuenta autenticada, que es la única válida.
        mensaje["To"] = ", ".join(saliente.destinatarios)
        mensaje["Subject"] = saliente.asunto
        if saliente.en_respuesta_a:
            # Lo que Gmail (y cualquier cliente) usa para enganchar la
            # respuesta al hilo: RFC 2822, más el threadId en el cuerpo.
            mensaje["In-Reply-To"] = saliente.en_respuesta_a
            cadena = f"{saliente.referencias} {saliente.en_respuesta_a}".strip()
            mensaje["References"] = cadena
        mensaje.set_content(saliente.cuerpo)
        crudo = base64.urlsafe_b64encode(mensaje.as_bytes()).decode("ascii")

        cuerpo: dict[str, Any] = {"raw": crudo}
        if saliente.hilo_id:
            cuerpo["threadId"] = saliente.hilo_id
        datos = await self._publicar(f"{BASE}/messages/send", token, cuerpo)
        # La cantidad de destinatarios, nunca las direcciones (RF-18).
        logger.info(
            "gmail.enviado",
            usuario_id=str(usuario_id),
            destinatarios=len(saliente.destinatarios),
        )
        return str(datos.get("id") or "")

    async def comprobar_acceso(self, usuario_id: UUID) -> None:
        # Sólo la credencial: a Google se le habla únicamente si hay que renovarla.
        await self._credencial(usuario_id)

    # --- Plomería ---------------------------------------------------------

    async def _credencial(self, usuario_id: UUID) -> str:
        token = await self._conectar.credencial_vigente(usuario_id, datetime.now(UTC))
        if token is None:
            raise CuentaNoConectadaError
        return token.access_token

    async def _pedir(
        self,
        url: str,
        token: str,
        parametros: Parametros,
        error_404: Exception | None = None,
    ) -> dict[str, Any]:
        try:
            respuesta = await self._cliente.get(
                url, params=parametros, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            logger.error("gmail.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Gmail.") from None

        if respuesta.status_code == httpx.codes.NOT_FOUND and error_404 is not None:
            raise error_404
        traducir_rechazo(respuesta, "gmail")
        return json_o_vacio(respuesta)

    async def _publicar(self, url: str, token: str, cuerpo: dict[str, Any]) -> dict[str, Any]:
        """El POST del envío, que dice si el correo salió (PB-035).

        Sólo `EnvioInciertoError` deja la duda; con cualquier otra excepción el
        correo seguro no salió. La lectura (`_pedir`) no distingue: leer de
        nuevo no hace daño, mandar de nuevo puede duplicar un correo.
        """
        try:
            respuesta = await self._cliente.post(
                url, json=cuerpo, headers={"Authorization": f"Bearer {token}"}
            )
        except _SIN_CONEXION as exc:
            logger.error("gmail.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Gmail.") from None
        except httpx.HTTPError as exc:
            # El pedido salió y la respuesta no volvió: Gmail pudo haberlo mandado.
            logger.error("gmail.envio_incierto", tipo=type(exc).__name__)
            raise EnvioInciertoError("Gmail no confirmó el envío.") from None

        if respuesta.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            logger.error(
                "gmail.envio_incierto",
                status_code=respuesta.status_code,
                motivo=motivo_de(respuesta),
            )
            raise EnvioInciertoError("Gmail falló mientras procesaba el envío.")
        if respuesta.status_code == httpx.codes.BAD_REQUEST:
            # El motivo es el texto de Google ("Invalid To header"), sin direcciones.
            logger.warning("gmail.envio_rechazado", motivo=motivo_de(respuesta))
            raise EnvioRechazadoError("Gmail rechazó el correo.")
        # El resto de los 4xx (401, los tres 403, el 429 del límite de envíos):
        # Gmail frenó el pedido antes de mandar nada.
        traducir_rechazo(respuesta, "gmail")
        return json_o_vacio(respuesta)


def _url_del_mensaje(correo_id: str) -> str:
    """La URL de un mensaje, sólo si el id tiene la forma de Gmail.

    Raises:
        EntityNotFoundError: Si no la tiene: un correo así no existe, y armar
            la URL con él sacaría el pedido del recurso de los mensajes.
    """
    if not _ID_DE_GMAIL.fullmatch(correo_id):
        # El largo y nada más: el id lo escribió el modelo, quizás inducido.
        logger.warning("gmail.id_invalido", largo=len(correo_id))
        raise EntityNotFoundError("Ese correo no existe o ya no está.")
    return f"{BASE}/messages/{correo_id}"


# --- De JSON a entidad ---------------------------------------------------------


def _a_correo(mensaje: dict[str, Any], con_cuerpo: bool = False, completo: bool = False) -> Correo:
    payload = mensaje.get("payload") if isinstance(mensaje.get("payload"), dict) else {}
    encabezados = _encabezados(payload or {})
    fragmento = unescape(str(mensaje.get("snippet") or ""))

    cuerpo: str | None = None
    adjuntos: tuple[str, ...] = ()
    if con_cuerpo:
        textos, htmls, nombres = _recorrer(payload or {})
        crudo = "\n".join(textos) or "\n".join(_html_a_texto(h) for h in htmls) or fragmento
        cuerpo = _entero(crudo) if completo else _limpiar(crudo)
        adjuntos = tuple(nombres)

    return Correo(
        id=str(mensaje.get("id") or ""),
        remitente=_remitente(encabezados.get("from", "")),
        # Vacío si no tiene: "(sin asunto)" lo pone quien lo muestra. En la
        # entidad viajaba al responder: "Re: (sin asunto)" (bug de PB-036).
        asunto=_decodificar_encabezado(encabezados.get("subject", "")),
        fecha=_fecha(mensaje.get("internalDate")),
        no_leido="UNREAD" in (mensaje.get("labelIds") or []),
        fragmento=fragmento,
        cuerpo=cuerpo,
        adjuntos=adjuntos,
        hilo_id=str(mensaje["threadId"]) if mensaje.get("threadId") else None,
        mensaje_id=_una_linea(encabezados.get("message-id", "")) or None,
        referencias=_una_linea(encabezados.get("references", "")),
        responder_a=_direcciones_de(encabezados.get("reply-to", "")),
    )


def _una_linea(valor: str) -> str:
    """Un encabezado en una sola línea: los ajenos pueden venir plegados."""
    return " ".join(valor.split())


def _direcciones_de(valor: str) -> tuple[str, ...]:
    """Las direcciones de un encabezado como `Reply-To`, sin los nombres."""
    pares = getaddresses([_decodificar_encabezado(valor)]) if valor else []
    return tuple(direccion for _, direccion in pares if "@" in direccion)


def _encabezados(parte: dict[str, Any]) -> dict[str, str]:
    """Los encabezados de una parte, con el nombre en minúsculas."""
    encabezados: dict[str, str] = {}
    for encabezado in parte.get("headers") or []:
        if isinstance(encabezado, dict) and isinstance(encabezado.get("name"), str):
            encabezados[encabezado["name"].lower()] = str(encabezado.get("value") or "")
    return encabezados


def _decodificar_encabezado(valor: str) -> str:
    """Decodifica palabras RFC 2047 (`=?UTF-8?B?...?=`) si las hubiera."""
    try:
        return str(make_header(decode_header(valor))).strip()
    except (ValueError, LookupError):
        return valor.strip()


def _remitente(valor: str) -> str:
    """ "Nombre <direccion>", o sólo la dirección si no trae nombre."""
    nombre, direccion = parseaddr(_decodificar_encabezado(valor))
    if nombre and direccion:
        return f"{nombre} <{direccion}>"
    return direccion or nombre or "(remitente desconocido)"


def _fecha(interna: Any) -> datetime:
    """`internalDate` es epoch en milisegundos, UTC: más fiable que el header Date."""
    try:
        return datetime.fromtimestamp(int(interna) / 1000, tz=UTC)
    except (TypeError, ValueError, OverflowError):
        return datetime.now(UTC)


def _recorrer(raiz: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Recorre el árbol MIME y separa texto plano, HTML y nombres de adjuntos."""
    textos: list[str] = []
    htmls: list[str] = []
    adjuntos: list[str] = []

    def visitar(parte: dict[str, Any]) -> None:
        tipo = str(parte.get("mimeType") or "").lower()
        nombre = parte.get("filename")
        if isinstance(nombre, str) and nombre:
            adjuntos.append(nombre)
            return
        if tipo.startswith("multipart/"):
            for hija in parte.get("parts") or []:
                if isinstance(hija, dict):
                    visitar(hija)
            return
        cuerpo = parte.get("body") if isinstance(parte.get("body"), dict) else {}
        datos = (cuerpo or {}).get("data")
        if not isinstance(datos, str):
            return
        texto = _decodificar_datos(datos, _charset(parte))
        if tipo == "text/plain":
            textos.append(texto)
        elif tipo == "text/html":
            htmls.append(texto)

    visitar(raiz)
    return textos, htmls, adjuntos


def _charset(parte: dict[str, Any]) -> str:
    tipo = _encabezados(parte).get("content-type", "")
    coincidencia = re.search(r'charset="?([\w.:-]+)"?', tipo, re.IGNORECASE)
    return coincidencia.group(1) if coincidencia else "utf-8"


def _decodificar_datos(datos: str, charset: str) -> str:
    """Base64url (Gmail a veces lo manda sin relleno) y el charset de la parte."""
    crudo = base64.urlsafe_b64decode(datos + "=" * (-len(datos) % 4))
    try:
        return crudo.decode(charset, errors="replace")
    except LookupError:  # charset que Python no conoce
        return crudo.decode("utf-8", errors="replace")


class _ExtractorDeTexto(HTMLParser):
    """Texto de un HTML, sin scripts ni estilos y con saltos en los bloques."""

    _BLOQUES = frozenset({"br", "p", "div", "li", "tr", "h1", "h2", "h3", "table"})
    _IGNORADOS = frozenset({"script", "style", "head"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.partes: list[str] = []
        self._ignorando = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._IGNORADOS:
            self._ignorando += 1
        elif tag in self._BLOQUES:
            self.partes.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORADOS and self._ignorando:
            self._ignorando -= 1
        elif tag in self._BLOQUES:
            self.partes.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignorando:
            self.partes.append(data)


def _html_a_texto(html: str) -> str:
    extractor = _ExtractorDeTexto()
    extractor.feed(html)
    extractor.close()
    return "".join(extractor.partes)


def _entero(texto: str) -> str:
    """El texto para reenviar: sin recorte y con sus citas, sólo normalizado."""
    return re.sub(r"\n{3,}", "\n\n", texto.replace("\r\n", "\n")).strip()


def _limpiar(texto: str) -> str:
    """Saca lo citado de hilos viejos, compacta espacios y recorta."""
    lineas = [
        linea.rstrip()
        for linea in texto.replace("\r\n", "\n").split("\n")
        if not linea.lstrip().startswith(">")
    ]
    limpio = re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()
    if len(limpio) > MAX_CARACTERES_DE_CUERPO:
        limpio = limpio[:MAX_CARACTERES_DE_CUERPO].rstrip() + "…\n[El correo sigue: se recortó.]"
    return limpio


def create_gmail_google(cliente: httpx.AsyncClient, conectar: ConectarGoogle) -> GmailGoogle:
    """Factory del adaptador; el composition root la llama."""
    return GmailGoogle(cliente, conectar)
