"""Cliente de envío contra la WhatsApp Cloud API (PB-004).

El cliente HTTP se abre una vez en el `lifespan` y se comparte durante todo el
proceso: abrir uno por mensaje agregaría un handshake TLS a cada respuesta y
pondría en riesgo el RNF de contestar en ≤ 3 segundos.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from src.application.ports.whatsapp import MensajeroWhatsApp
from src.domain.exceptions import RepositoryError, ServiceUnavailableError
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp
from src.infrastructure.config.settings import Settings

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
CODIGOS_PERMANENTES = frozenset({131026, 131047, 131051, 100, 131030})


class ClienteWhatsApp(MensajeroWhatsApp):
    """Implementación de `MensajeroWhatsApp` contra la Graph API de Meta."""

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
            # Con el "+": si se omite, Meta antepone el código de país de
            # nuestro número y el mensaje se entrega a otra persona.
            "to": destino.valor,
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
            raise ServiceUnavailableError("No se pudo contactar a WhatsApp.") from None

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
            codigo = _codigo_de_error(respuesta)
            logger.error(
                "whatsapp.envio.rechazado",
                status_code=respuesta.status_code,
                codigo_meta=codigo,
                permanente=codigo in CODIGOS_PERMANENTES,
            )
            raise RepositoryError(f"WhatsApp rechazó el envío (código {codigo}).")

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


def _codigo_de_error(respuesta: httpx.Response) -> int | None:
    """Extrae `error.code` del cuerpo de una respuesta fallida."""
    try:
        error = respuesta.json().get("error")
    except ValueError:
        return None
    if not isinstance(error, dict):
        return None
    codigo = error.get("code")
    return codigo if isinstance(codigo, int) else None


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
