"""El callback del flujo OAuth2 con Notion (PB-014).

`GET /oauth/notion/callback`: la vuelta de Notion con el código. Uno solo, y no
dos como en Google: `/conectar notion` manda la URL de Notion directo, así que
un `/iniciar` no lo usaría nadie. (El de Google, además, vuelve a firmar el
`state` y le estira la vida: es deuda anotada.)

Devuelve HTML, como el de Google: del otro lado hay una persona mirando el
teléfono. El `code` de la query string no queda escrito en nuestros logs por
lo mismo que allá: el middleware loguea sólo el path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Query, status
from starlette.responses import Response

from src.domain.exceptions import AutorizacionFallidaError, InvalidValueError
from src.interfaces.api.dependencies import ConectarNotionDep
from src.interfaces.api.routers.html import pagina

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/oauth/notion", tags=["oauth"])

_NO_DISPONIBLE = (
    "Conexión no disponible",
    "El asistente no tiene configurada la conexión con Notion.",
)
_ENLACE_INVALIDO = (
    "Enlace inválido o vencido",
    "Volvé a WhatsApp y escribí <b>/conectar notion</b> para pedir uno nuevo.",
)


@router.get("/callback", include_in_schema=False)
async def callback(
    conectar: ConectarNotionDep,
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
) -> Response:
    """Recibe la vuelta de Notion, canjea el código y guarda las credenciales.

    Los parámetros son opcionales para que una vuelta incompleta muestre una
    página entendible en vez de un 422 con el detalle de qué campos faltan.
    """
    if conectar is None:
        return pagina(*_NO_DISPONIBLE, icono="🔌", codigo=status.HTTP_503_SERVICE_UNAVAILABLE)

    if error is not None:
        # Lo más común es access_denied: la persona apretó "Cancelar". No es un
        # fallo del sistema, así que se responde 200.
        logger.info("notion.callback.rechazado_por_el_usuario", motivo=error)
        return pagina(
            "No se conectó tu Notion",
            "Cancelaste la autorización. Si fue sin querer, escribí "
            "<b>/conectar notion</b> en WhatsApp para volver a intentarlo.",
            icono="🚫",
        )

    if code is None or state is None:
        logger.warning("notion.callback.incompleto", trajo_code=code is not None)
        return pagina(*_ENLACE_INVALIDO, icono="⏳", codigo=status.HTTP_400_BAD_REQUEST)

    try:
        usuario_id = await conectar.completar(code, state, datetime.now(UTC))
    except InvalidValueError:
        # State deformado, mal firmado, vencido o firmado para Google. Sin
        # detalle: quien lo manda es un atacante o alguien con un enlace viejo.
        logger.warning("notion.callback.state_invalido")
        return pagina(*_ENLACE_INVALIDO, icono="⏳", codigo=status.HTTP_400_BAD_REQUEST)
    except AutorizacionFallidaError:
        return pagina(
            "No pude completar la conexión",
            "Notion rechazó la autorización. Probá de nuevo en un minuto con "
            "<b>/conectar notion</b>.",
            icono="⚠️",
            codigo=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    logger.info("notion.callback.completado", usuario_id=str(usuario_id))
    return pagina(
        "¡Listo!",
        "Tu Notion quedó conectado. Volvé a WhatsApp y preguntame por tus páginas.",
        icono="✅",
    )
