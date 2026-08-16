"""Webhook de WhatsApp Cloud API (PB-004).

Dos endpoints:

- `GET`  — handshake de verificación con Meta. Devuelve el `hub.challenge` en
  **texto plano**; si se devolviera un `str` desde FastAPI, saldría entre
  comillas como JSON y Meta rechazaría la verificación en silencio.
- `POST` — recepción de eventos. Valida la firma sobre los bytes crudos,
  responde 200 de inmediato y procesa en segundo plano.

**El POST sólo devuelve 200 o 403.** Cualquier otro código haría que Meta
reintente durante horas: un payload con forma rara es un problema nuestro para
loguear, no algo que se arregle reintentando.

Regla del trabajo en segundo plano: recibe **valores planos**, nunca el
`Request` ni dependencias con `yield`. Después de enviada la respuesta el
`Request` ya no tiene cuerpo, y una excepción que escape se convierte en
`RuntimeError("response already started")`, perdiendo el log y cortando la
conexión.
"""

from __future__ import annotations

import hmac
import json
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Query, Response, status
from fastapi.responses import PlainTextResponse
from starlette.requests import Request

from src.application.dto.mensaje_entrante import MensajeEntrante
from src.application.use_cases.procesar_mensaje_entrante import ProcesarMensajeEntrante
from src.domain.exceptions import LifeSyncError
from src.infrastructure.external.whatsapp.deduplicador import DeduplicadorDeMensajes
from src.infrastructure.external.whatsapp.firma import HEADER_FIRMA, firma_valida
from src.infrastructure.external.whatsapp.parser import parsear
from src.interfaces.api.dependencies import (
    DeduplicadorWhatsAppDep,
    ProcesadorDeMensajesDep,
    SettingsDep,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhooks/whatsapp", tags=["whatsapp"])

# Meta manda payloads de hasta 3 MB, pero un webhook legítimo de mensajes de
# texto pesa unos pocos KB. Cortar antes evita gastar CPU en el HMAC de algo
# que no vamos a procesar.
LIMITE_CUERPO_BYTES = 1024 * 1024

# El challenge es un entero según la documentación. Se valida antes de
# reflejarlo para no convertir el endpoint en un reflector de contenido ajeno.
LARGO_MAXIMO_CHALLENGE = 64

MENSAJE_DE_ERROR = "Tuve un problema para procesar tu mensaje. Probá de nuevo en un minuto."


@router.get(
    "",
    response_class=PlainTextResponse,
    summary="Verificación del webhook (handshake de Meta)",
    include_in_schema=False,
)
async def verificar(
    settings: SettingsDep,
    hub_mode: Annotated[str | None, Query(alias="hub.mode")] = None,
    hub_challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
    hub_verify_token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
) -> Response:
    """Responde el handshake de verificación de Meta.

    Los parámetros son opcionales para que un GET incompleto dé 403 y no un 422
    con el detalle de qué campos faltan.
    """
    esperado = settings.whatsapp_verify_token
    if esperado is None:
        logger.error("whatsapp.webhook.sin_verify_token")
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    coincide = hub_verify_token is not None and hmac.compare_digest(
        hub_verify_token, esperado.get_secret_value()
    )
    challenge_valido = (
        hub_challenge is not None
        and hub_challenge.isdigit()
        and len(hub_challenge) <= LARGO_MAXIMO_CHALLENGE
    )

    if hub_mode != "subscribe" or not coincide or not challenge_valido:
        # Sin detalle del motivo: quien pregunta es Meta o un atacante.
        logger.warning("whatsapp.webhook.verificacion_rechazada", modo=hub_mode)
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    logger.info("whatsapp.webhook.verificado")
    return PlainTextResponse(hub_challenge)


@router.post(
    "",
    summary="Recepción de eventos de WhatsApp",
    include_in_schema=False,
)
async def recibir(
    request: Request,
    tareas: BackgroundTasks,
    settings: SettingsDep,
    procesador: ProcesadorDeMensajesDep,
    deduplicador: DeduplicadorWhatsAppDep,
) -> Response:
    """Recibe un evento, valida la firma y agenda el procesamiento."""
    cuerpo = await request.body()

    if len(cuerpo) > LIMITE_CUERPO_BYTES:
        logger.warning("whatsapp.webhook.cuerpo_demasiado_grande", bytes=len(cuerpo))
        return _recibido()

    if not _firma_aceptable(cuerpo, request, settings.firma_exigida, settings):
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    try:
        payload = json.loads(cuerpo)
    except ValueError:
        logger.warning("whatsapp.webhook.json_invalido", bytes=len(cuerpo))
        return _recibido()

    contenido = parsear(
        payload,
        phone_number_id_propio=settings.whatsapp_phone_number_id,
    )

    logger.info(
        "whatsapp.webhook.recibido",
        mensajes=len(contenido.mensajes),
        acuses=contenido.acuses,
        no_soportados=contenido.no_soportados,
        vencidos=contenido.vencidos,
        ignorados=contenido.ignorados,
        truncados=contenido.truncados,
        sin_remitente=contenido.sin_remitente,
        ajenos=contenido.ajenos,
        errores_de_meta=contenido.errores_de_meta,
        claves_desconocidas=list(contenido.claves_desconocidas),
    )

    if not contenido.hay_trabajo:
        return _recibido()

    if procesador is None:
        # Modo degradado: falta Supabase o faltan credenciales de WhatsApp.
        # Se responde 200 igual, para que Meta no reintente contra un entorno
        # que sabemos que no puede atenderlo.
        logger.warning("whatsapp.sin_persistencia", mensajes=len(contenido.mensajes))
        return _recibido()

    tareas.add_task(
        _procesar_en_segundo_plano,
        procesador=procesador,
        deduplicador=deduplicador,
        mensajes=contenido.mensajes,
    )
    return _recibido()


def _recibido() -> Response:
    """Respuesta estándar de aceptación para Meta."""
    return PlainTextResponse("EVENT_RECEIVED")


def _firma_aceptable(
    cuerpo: bytes,
    request: Request,
    exigida: bool,
    settings: SettingsDep,
) -> bool:
    """Valida `X-Hub-Signature-256`, o la omite si el entorno lo permite."""
    secreto = settings.whatsapp_app_secret

    if secreto is None:
        if exigida:
            # Defensa en profundidad: el validador de Settings ya impide
            # arrancar así en producción, pero no se confía sólo en eso.
            logger.error("whatsapp.webhook.sin_app_secret")
            return False
        logger.warning(
            "whatsapp.webhook.firma_omitida", motivo="WHATSAPP_APP_SECRET sin configurar"
        )
        return True

    if firma_valida(cuerpo, request.headers.get(HEADER_FIRMA), secreto.get_secret_value()):
        return True

    # Sin el cuerpo ni la cabecera recibida: es entrada no confiable.
    logger.warning("whatsapp.webhook.firma_invalida", bytes=len(cuerpo))
    return False


async def _procesar_en_segundo_plano(
    *,
    procesador: ProcesarMensajeEntrante,
    deduplicador: DeduplicadorDeMensajes,
    mensajes: tuple[MensajeEntrante, ...],
) -> None:
    """Procesa los mensajes después de haber respondido 200.

    Esta función **no puede dejar escapar ninguna excepción**: corre con la
    respuesta ya enviada, así que cualquier error se convertiría en
    `RuntimeError("response already started")` y se perdería el diagnóstico.
    """
    for mensaje in mensajes:
        if not deduplicador.marcar_si_es_nuevo(mensaje.wamid):
            logger.info("whatsapp.mensaje_duplicado", wamid=mensaje.wamid)
            continue

        try:
            await procesador.ejecutar(mensaje)
        except Exception as exc:  # el borde del background nunca propaga
            # Se olvida para que el reintento de Meta tenga otra oportunidad:
            # un fallo transitorio no debe dejar el mensaje sin respuesta.
            deduplicador.olvidar(mensaje.wamid)
            logger.exception(
                "whatsapp.procesamiento_fallido",
                wamid=mensaje.wamid,
                tipo=type(exc).__name__,
            )
            await _avisar_del_error(procesador, mensaje, exc)


async def _avisar_del_error(
    procesador: ProcesarMensajeEntrante,
    mensaje: MensajeEntrante,
    exc: Exception,
) -> None:
    """Intenta avisarle al usuario que algo falló (RF-19). Best effort."""
    texto = exc.mensaje_usuario if isinstance(exc, LifeSyncError) else MENSAJE_DE_ERROR
    try:
        await procesador.avisar(mensaje.remitente, texto)
    except Exception:  # si tampoco se puede avisar, sólo queda el log
        logger.warning("whatsapp.aviso_de_error_fallido", wamid=mensaje.wamid)
