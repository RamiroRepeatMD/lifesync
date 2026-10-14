"""Cliente de la WhatsApp Cloud API: enviar (PB-004) y bajar archivos recibidos (PB-038).

El cliente HTTP se abre una vez en el `lifespan` y se comparte durante todo el
proceso: abrir uno por mensaje agregaría un handshake TLS a cada respuesta y
pondría en riesgo el RNF de contestar en ≤ 3 segundos.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from src.application.ports.archivos_recibidos import ArchivoDescargado, ArchivosRecibidos
from src.application.ports.whatsapp import MensajeroWhatsApp
from src.domain.exceptions import (
    ArchivoDemasiadoGrandeError,
    ArchivoNoDisponibleError,
    MensajeNoEnviadoError,
    MensajeRechazadoError,
    ServiceUnavailableError,
)
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp
from src.infrastructure.config.settings import Settings
from src.infrastructure.external.whatsapp.parser import ID_DE_MEDIA

logger = structlog.get_logger(__name__)

# Versión fijada a propósito: una URL sin versión resuelve a la más vieja
# disponible y cambia de comportamiento sin aviso.
VERSION_GRAPH = "v23.0"
BASE_GRAPH = "https://graph.facebook.com"

TIMEOUT_SEGUNDOS = 10.0
TIMEOUT_CONEXION_SEGUNDOS = 5.0

# Tope de envíos concurrentes. Un lote de webhooks puede disparar muchos a la
# vez y Meta responde 130429 si se pasa el throughput.
ENVIOS_CONCURRENTES = 8

# Códigos de Meta que no tiene sentido reintentar: el problema no se arregla solo.
# 190 = token vencido o inválido; reintentar con el mismo token no cambia nada.
CODIGOS_PERMANENTES = frozenset({131026, 131047, 131051, 100, 131030, 190})

# --- El 9 de los móviles argentinos -----------------------------------------
#
# El circuito argentino es asimétrico y hay que corregirlo justo acá, en el
# borde de salida. Verificado empíricamente contra la Graph API con un número
# real:
#
#   to: "+5491160007044"  -> 131030 "Recipient phone number not in allowed list"
#   to: "5491160007044"   -> 131030
#   to: "+541160007044"   -> entregado, y la respuesta trae wa_id "5491160007044"
#   to: "541160007044"    -> entregado, mismo wa_id
#
# Es decir: Meta ENTREGA los webhooks con el 9, pero al ENVIAR exige el número
# SIN el 9 y le re-agrega el 9 por su cuenta al resolver el destinatario. El
# "+" es indistinto para ese matcheo.
#
# Por eso la corrección vive en el adaptador y no en `NumeroWhatsApp`: la
# identidad del usuario —la que se guarda en Supabase y con la que se lo
# busca— sigue siendo la que Meta entregó, con el 9. Lo único que cambia es
# el formato de cable del campo `to`.
# Archivos recibidos (PB-038). Meta sirve la media desde su CDN; el cliente
# manda nuestro token en TODOS los pedidos (header por defecto), así que un
# link que no sea de ahí se lo entregaría a un tercero: se valida antes.
DOMINIO_DE_MEDIA = "fbsbx.com"
TIMEOUT_DE_DESCARGA_SEGUNDOS = 60.0

PREFIJO_AR_MOVIL = "+549"
PREFIJO_AR = "+54"


def destino_para_meta(numero: NumeroWhatsApp) -> str:
    """Convierte el número canónico al formato que Meta acepta en `to`.

    Para los móviles argentinos saca el `9` (ver el comentario de
    PREFIJO_AR_MOVIL). Para el resto devuelve el número tal cual.

    No modifica la identidad del usuario: `numero.valor` sigue siendo lo que
    se persiste y con lo que se lo busca.
    """
    if numero.valor.startswith(PREFIJO_AR_MOVIL):
        return PREFIJO_AR + numero.valor[len(PREFIJO_AR_MOVIL) :]
    return numero.valor


class ClienteWhatsApp(MensajeroWhatsApp, ArchivosRecibidos):
    """`MensajeroWhatsApp` y `ArchivosRecibidos` contra la Graph API de Meta."""

    def __init__(self, cliente: httpx.AsyncClient, settings: Settings) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._cliente = cliente
        self._settings = settings
        self._semaforo = asyncio.Semaphore(ENVIOS_CONCURRENTES)

    async def enviar_texto(self, destino: NumeroWhatsApp, texto: str) -> None:
        """Envía un mensaje de texto al usuario."""
        cuerpo: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            # Siempre con el "+": si se omite, Meta antepone el código de país
            # de nuestro número de negocio y el mensaje se entrega a otro.
            "to": destino_para_meta(destino),
            "type": "text",
            "text": {"preview_url": False, "body": texto},
        }

        async with self._semaforo:
            respuesta = await self._postear(cuerpo)

        self._revisar_respuesta(respuesta, destino)

    async def _postear(self, cuerpo: dict[str, Any]) -> httpx.Response:
        """Hace el POST a Graph, traduciendo los fallos de red."""
        try:
            return await self._cliente.post(self._url_de_envio(), json=cuerpo)
        except httpx.HTTPError as exc:
            logger.error("whatsapp.envio.error_transporte", tipo=type(exc).__name__)
            raise MensajeNoEnviadoError("No se pudo contactar a WhatsApp.") from None

    async def descargar(self, referencia: str, maximo_bytes: int) -> ArchivoDescargado:
        """Baja un archivo recibido: primero sus datos en Graph, después los bytes.

        El link que devuelve Graph vale 5 minutos y está firmado: se pide al
        toque y nunca se loguea. Todo dentro de un tiempo máximo: el timeout de
        httpx es por lectura, no por descarga completa.
        """
        if not ID_DE_MEDIA.fullmatch(referencia):
            # Termina en la URL de Graph, que va con nuestro token.
            raise ArchivoNoDisponibleError("La referencia no es un id de media.")
        try:
            async with asyncio.timeout(TIMEOUT_DE_DESCARGA_SEGUNDOS):
                datos = await self._datos_de_media(referencia)
                tamaño = _tamaño_de(datos)
                if tamaño is not None and tamaño > maximo_bytes:
                    logger.info("whatsapp.media.demasiado_grande", bytes=tamaño)
                    raise ArchivoDemasiadoGrandeError("El archivo supera el tope.")
                contenido = await self._bajar(_link_de_media(datos), maximo_bytes)
        except TimeoutError:
            logger.error("whatsapp.media.timeout")
            raise ServiceUnavailableError(
                "WhatsApp tardó demasiado en entregar el archivo."
            ) from None

        tipo = datos.get("mime_type")
        logger.info("whatsapp.media.descargada", bytes=len(contenido))
        return ArchivoDescargado(
            contenido=contenido,
            tipo_mime=tipo if isinstance(tipo, str) and tipo else "application/octet-stream",
        )

    async def _datos_de_media(self, referencia: str) -> dict[str, Any]:
        """`GET /{id}` en Graph, sólo para media de nuestro número."""
        try:
            respuesta = await self._cliente.get(
                f"{BASE_GRAPH}/{VERSION_GRAPH}/{referencia}",
                params={"phone_number_id": self._settings.whatsapp_phone_number_id or ""},
            )
        except httpx.HTTPError as exc:
            logger.error("whatsapp.media.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a WhatsApp.") from None

        if respuesta.status_code in (httpx.codes.BAD_REQUEST, httpx.codes.NOT_FOUND):
            # Un id que ya venció (los de lo recibido duran días) o que no
            # existe: Graph contesta 400 (código 100) o 404.
            error = _error_de(respuesta)
            logger.warning(
                "whatsapp.media.no_disponible",
                status_code=respuesta.status_code,
                codigo_meta=error.get("code"),
                subcodigo=error.get("error_subcode"),
            )
            raise ArchivoNoDisponibleError("WhatsApp ya no tiene ese archivo.")
        if respuesta.status_code >= httpx.codes.BAD_REQUEST:
            error = _error_de(respuesta)
            logger.error(
                "whatsapp.media.rechazada",
                status_code=respuesta.status_code,
                codigo_meta=error.get("code"),
            )
            raise ServiceUnavailableError("WhatsApp no entregó el archivo.")
        try:
            datos = respuesta.json()
        except ValueError:
            datos = None
        return datos if isinstance(datos, dict) else {}

    async def _bajar(self, link: httpx.URL, maximo_bytes: int) -> bytes:
        """Los bytes del archivo, cortando apenas superan el tope."""
        partes: list[bytes] = []
        total = 0
        try:
            async with self._cliente.stream("GET", link, follow_redirects=False) as respuesta:
                if respuesta.status_code == httpx.codes.NOT_FOUND:
                    logger.warning("whatsapp.media.link_vencido")
                    raise ArchivoNoDisponibleError("El link del archivo venció.")
                if respuesta.status_code != httpx.codes.OK:
                    # Incluye las redirecciones: no se sigue ninguna.
                    logger.error(
                        "whatsapp.media.descarga_fallida", status_code=respuesta.status_code
                    )
                    raise ServiceUnavailableError("WhatsApp no entregó el archivo.")
                async for parte in respuesta.aiter_bytes():
                    total += len(parte)
                    if total > maximo_bytes:
                        logger.info("whatsapp.media.demasiado_grande", bytes=total)
                        raise ArchivoDemasiadoGrandeError("El archivo supera el tope.")
                    partes.append(parte)
        except httpx.HTTPError as exc:
            logger.error("whatsapp.media.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo bajar el archivo de WhatsApp.") from None
        return b"".join(partes)

    def _url_de_envio(self) -> str:
        """Arma la URL desde la configuración, nunca desde el payload entrante.

        Tomar el `phone_number_id` de un webhook para armar esta URL sería
        dejar que un tercero elija el destino de una petición que lleva nuestro
        token adjunto.
        """
        return f"{BASE_GRAPH}/{VERSION_GRAPH}/{self._settings.whatsapp_phone_number_id}/messages"

    def _revisar_respuesta(self, respuesta: httpx.Response, destino: NumeroWhatsApp) -> None:
        """Interpreta la respuesta de Meta y loguea el resultado."""
        if respuesta.status_code >= httpx.codes.BAD_REQUEST:
            error = _error_de(respuesta)
            codigo = error.get("code")
            logger.error(
                "whatsapp.envio.rechazado",
                status_code=respuesta.status_code,
                codigo_meta=codigo,
                subcodigo=error.get("error_subcode"),
                # Meta explica el rechazo en texto; sin esto hay que salir a
                # reproducir la llamada con curl para saber qué pasó. No es
                # dato personal: son diagnósticos de la API.
                mensaje_meta=error.get("message"),
                detalle_meta=_detalle_de(error),
                permanente=codigo in CODIGOS_PERMANENTES,
            )
            # El despachador de recordatorios (PB-030) necesita distinguirlos:
            # un rechazo permanente no se reintenta, un corte sí.
            if codigo in CODIGOS_PERMANENTES:
                raise MensajeRechazadoError(f"WhatsApp rechazó el envío (código {codigo}).")
            raise MensajeNoEnviadoError(f"WhatsApp rechazó el envío (código {codigo}).")

        self._avisar_si_meta_normalizo(respuesta, destino)
        logger.info("whatsapp.mensaje.enviado", status_code=respuesta.status_code)

    @staticmethod
    def _avisar_si_meta_normalizo(respuesta: httpx.Response, destino: NumeroWhatsApp) -> None:
        """Detecta si Meta reescribió el número que le mandamos.

        Es el sensor del problema del 9 argentino: si Meta devuelve un `wa_id`
        distinto del que enviamos, queremos enterarnos. Se loguean los largos y
        nunca los números, que son dato personal (RF-18).
        """
        try:
            contactos = respuesta.json().get("contacts")
            if not isinstance(contactos, list) or not contactos:
                return
            devuelto = contactos[0].get("wa_id")
        except (ValueError, AttributeError, TypeError, IndexError):
            return

        if not isinstance(devuelto, str):
            return

        enviado = destino.valor.removeprefix("+")
        if devuelto != enviado:
            logger.warning(
                "whatsapp.numero_normalizado_por_meta",
                largo_enviado=len(enviado),
                largo_devuelto=len(devuelto),
            )


def _tamaño_de(datos: dict[str, Any]) -> int | None:
    """`file_size` de Graph, que la referencia documenta como texto."""
    try:
        return int(datos["file_size"])
    except (KeyError, TypeError, ValueError):
        return None


def _link_de_media(datos: dict[str, Any]) -> httpx.URL:
    """El link de descarga, sólo si es de Meta: https, su dominio, sin trampas.

    Se valida con `httpx.URL` y no por substring: `fbsbx.com.evil.example` o
    `usuario@…` pasarían un `in`. Al host ajeno no se le pide nada.
    """
    crudo = datos.get("url")
    try:
        link = httpx.URL(crudo) if isinstance(crudo, str) else None
    except httpx.InvalidURL:
        link = None
    host = link.host if link is not None else ""
    de_meta = host == DOMINIO_DE_MEDIA or host.endswith(f".{DOMINIO_DE_MEDIA}")
    if link is None or link.scheme != "https" or not de_meta or link.userinfo or link.port:
        logger.error("whatsapp.media.link_inesperado", host=host)
        raise ServiceUnavailableError("WhatsApp devolvió un link de descarga inesperado.")
    return link


def _error_de(respuesta: httpx.Response) -> dict[str, Any]:
    """Extrae el objeto `error` del cuerpo de una respuesta fallida.

    Devuelve un diccionario vacío si el cuerpo no es JSON o no tiene la forma
    esperada, para que el llamador no tenga que defenderse.
    """
    try:
        error = respuesta.json().get("error")
    except ValueError:
        return {}
    return error if isinstance(error, dict) else {}


def _detalle_de(error: dict[str, Any]) -> str | None:
    """Saca el texto explicativo de `error.error_data.details`.

    Es donde Meta escribe la causa concreta: por ejemplo, que el destinatario
    no está en la lista de autorizados.
    """
    datos = error.get("error_data")
    if not isinstance(datos, dict):
        return None
    detalle = datos.get("details")
    return detalle if isinstance(detalle, str) else None


def create_whatsapp_client(settings: Settings) -> httpx.AsyncClient:
    """Abre el cliente HTTP hacia Graph, con timeouts y auth ya configurados.

    Raises:
        ServiceUnavailableError: Si faltan credenciales.
    """
    token = settings.whatsapp_token
    if token is None or settings.whatsapp_phone_number_id is None:
        raise ServiceUnavailableError(
            "Faltan WHATSAPP_TOKEN y/o WHATSAPP_PHONE_NUMBER_ID para enviar mensajes."
        )

    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token.get_secret_value()}"},
        timeout=httpx.Timeout(TIMEOUT_SEGUNDOS, connect=TIMEOUT_CONEXION_SEGUNDOS),
    )


async def close_whatsapp_client(cliente: httpx.AsyncClient | None) -> None:
    """Cierra el cliente HTTP. Tolera None y errores de cierre."""
    if cliente is None:
        return
    try:
        await cliente.aclose()
    except Exception as exc:  # fallar al apagar no rompe el shutdown
        logger.warning("whatsapp.cierre_con_error", tipo=type(exc).__name__)
