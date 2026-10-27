"""Endpoints del flujo OAuth2 con Google (PB-009).

Dos endpoints, los dos pensados para que los abra una persona en el navegador:

- `GET /oauth/google/iniciar`  — redirige a la pantalla de consentimiento.
- `GET /oauth/google/callback` — la vuelta de Google con el código.

**Devuelven HTML, no JSON.** Del otro lado hay alguien mirando la pantalla del
teléfono, no un cliente de API: un `{"error": {...}}` no le dice nada.

Sobre el `code` que llega en la query string: es una credencial de un solo uso.
Nuestro middleware loguea sólo `request.url.path`, sin query, y `uvicorn.access`
está silenciado desde PB-004, así que no queda escrito. Lo que **no** podemos
evitar es que quede en el historial del navegador y en los logs de borde de la
plataforma: es inherente al flujo de authorization code, y por eso los códigos
duran minutos y se queman al usarse.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Query, status
from fastapi.responses import RedirectResponse
from starlette.responses import Response

from src.domain.exceptions import AutorizacionFallidaError, InvalidValueError
from src.interfaces.api.dependencies import ConectarGoogleDep
from src.interfaces.api.routers.html import pagina

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/oauth/google", tags=["oauth"])

_NO_DISPONIBLE = (
    "Conexión no disponible",
    "El asistente no tiene configurada la conexión con Google.",
)
_ENLACE_INVALIDO = (
    "Enlace inválido o vencido",
    "Volvé a WhatsApp y escribí <b>/conectar</b> para pedir uno nuevo.",
)


@router.get("/iniciar", include_in_schema=False)
async def iniciar(
    conectar: ConectarGoogleDep,
    state: Annotated[str | None, Query()] = None,
) -> Response:
    """Redirige a la pantalla de consentimiento de Google.

    Existe para que el enlace que se manda por WhatsApp sea corto y de nuestro
    dominio, en vez de los ~400 caracteres de la URL de Google. No agrega
    superficie: el `state` que recibe es el mismo firmado, y se vuelve a
    verificar antes de redirigir.
    """
    if conectar is None:
        return pagina(*_NO_DISPONIBLE, icono="🔌", codigo=status.HTTP_503_SERVICE_UNAVAILABLE)
    if state is None:
        return pagina(*_ENLACE_INVALIDO, icono="⏳", codigo=status.HTTP_400_BAD_REQUEST)

    ahora = datetime.now(UTC)
    try:
        usuario_id = conectar.usuario_del_estado(state, ahora)
    except InvalidValueError:
        logger.warning("google.iniciar.state_invalido")
        return pagina(*_ENLACE_INVALIDO, icono="⏳", codigo=status.HTTP_400_BAD_REQUEST)

    logger.info("google.iniciar.redirigiendo", usuario_id=str(usuario_id))
    return RedirectResponse(
        conectar.link_de_autorizacion(usuario_id, ahora),
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )


@router.get("/callback", include_in_schema=False)
async def callback(
    conectar: ConectarGoogleDep,
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
) -> Response:
    """Recibe la vuelta de Google, canjea el código y guarda las credenciales.

    Los parámetros son opcionales para que una vuelta incompleta muestre una
    página entendible en vez de un 422 con el detalle de qué campos faltan.
    """
    if conectar is None:
        return pagina(*_NO_DISPONIBLE, icono="🔌", codigo=status.HTTP_503_SERVICE_UNAVAILABLE)

    if error is not None:
        # El caso más común es access_denied: la persona apretó "Cancelar".
        # No es un fallo del sistema, así que se responde 200.
        logger.info("google.callback.rechazado_por_el_usuario", motivo=error)
        return pagina(
            "No se conectó tu cuenta",
            "Cancelaste la autorización. Si fue sin querer, escribí "
            "<b>/conectar</b> en WhatsApp para volver a intentarlo.",
            icono="🚫",
        )

    if code is None or state is None:
        logger.warning("google.callback.incompleto", trajo_code=code is not None)
        return pagina(*_ENLACE_INVALIDO, icono="⏳", codigo=status.HTTP_400_BAD_REQUEST)

    try:
        usuario_id = await conectar.completar(code, state, datetime.now(UTC))
    except InvalidValueError:
        # State deformado, mal firmado o vencido. Sin detalle: quien lo manda
        # es un atacante o alguien con un enlace viejo.
        logger.warning("google.callback.state_invalido")
        return pagina(*_ENLACE_INVALIDO, icono="⏳", codigo=status.HTTP_400_BAD_REQUEST)
    except AutorizacionFallidaError:
        return pagina(
            "No pude completar la conexión",
            "Google rechazó la autorización. Probá de nuevo en un minuto con <b>/conectar</b>.",
            icono="⚠️",
            codigo=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    logger.info("google.callback.completado", usuario_id=str(usuario_id))
    return pagina(
        "¡Listo!",
        "Tu cuenta de Google quedó conectada. Volvé a WhatsApp y preguntame por tu agenda.",
        icono="✅",
    )
