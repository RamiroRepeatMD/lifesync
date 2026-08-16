"""Ciclo de vida del cliente asíncrono de Supabase (PB-003).

El cliente se abre una vez en el `lifespan` de FastAPI y se comparte durante
todo el proceso: crear uno por request agregaría un handshake TLS a cada
mensaje y pondría en riesgo el RNF de responder en ≤ 3 segundos.

Se usa el cliente **asíncrono** (`acreate_client`) porque los handlers de
FastAPI son `async`: el cliente sincrónico bloquearía el event loop.
"""

from __future__ import annotations

import asyncio

import structlog
from supabase import AsyncClient, AsyncClientOptions, acreate_client

from src.domain.exceptions import ServiceUnavailableError
from src.infrastructure.config.settings import Settings

logger = structlog.get_logger(__name__)

# Tabla liviana contra la que se hace el ping de readiness.
TABLA_PING = "usuarios"

TIMEOUT_CONSULTA_SEGUNDOS = 10.0
# El readiness debe responder rápido: una probe que cuelga 10 s hace que la
# plataforma dé por muerta la instancia.
TIMEOUT_PING_SEGUNDOS = 2.0


async def create_supabase_client(settings: Settings) -> AsyncClient:
    """Abre el cliente de Supabase con las credenciales de configuración.

    Raises:
        ServiceUnavailableError: Si faltan credenciales o el cliente no se
            puede construir.
    """
    if settings.supabase_url is None or settings.supabase_key is None:
        raise ServiceUnavailableError(
            "Faltan SUPABASE_URL y/o SUPABASE_KEY para conectarse a Supabase."
        )

    opciones = AsyncClientOptions(
        schema="public",
        # El backend usa la service_role key: no hay sesión de usuario que
        # refrescar ni persistir, y evitamos una tarea de fondo inútil.
        auto_refresh_token=False,
        persist_session=False,
        postgrest_client_timeout=TIMEOUT_CONSULTA_SEGUNDOS,
    )

    try:
        cliente = await acreate_client(
            settings.supabase_url,
            settings.supabase_key.get_secret_value(),
            opciones,
        )
    except Exception as exc:
        # No se incluye la excepción original en el mensaje: puede traer la key.
        logger.error("supabase.conexion_fallida", tipo=type(exc).__name__)
        raise ServiceUnavailableError("No se pudo conectar con Supabase.") from exc

    # Ojo: `acreate_client` no hace I/O. Una URL o una key equivocadas no
    # fallan acá, fallan en la primera consulta. Por eso existe el readiness.
    return cliente


async def close_supabase_client(cliente: AsyncClient | None) -> None:
    """Cierra las conexiones HTTP del cliente. Tolera None y errores de cierre.

    `AsyncClient` no expone un cierre único, así que se cierran sus dos
    subclientes con estado propio.
    """
    if cliente is None:
        return

    try:
        await cliente.postgrest.aclose()
        await cliente.auth.close()
    except Exception as exc:
        # Fallar al apagar no debe romper el shutdown del proceso.
        logger.warning("supabase.cierre_con_error", tipo=type(exc).__name__)


async def ping(cliente: AsyncClient) -> bool:
    """Comprueba que la base responde, para el readiness probe.

    Hace la consulta más barata posible: una fila de `usuarios`. Devuelve False
    en vez de propagar, porque el health check reporta estado, no falla.
    """
    try:
        async with asyncio.timeout(TIMEOUT_PING_SEGUNDOS):
            await cliente.table(TABLA_PING).select("id").limit(1).execute()
    except Exception as exc:
        logger.warning("supabase.ping_fallido", tipo=type(exc).__name__)
        return False
    return True
