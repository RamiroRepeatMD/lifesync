"""Factory de la aplicación FastAPI.

`create_app()` es el punto donde se ensambla el sistema: se lee la
configuración, se configura el logging, se registran middlewares, manejadores
de error y routers. Es una factory (y no un `app` global) para que los tests
puedan levantar instancias aisladas con configuración propia.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.infrastructure.config.logging import configure_logging
from src.infrastructure.config.settings import Settings, get_settings
from src.infrastructure.external.whatsapp.cliente import (
    close_whatsapp_client,
    create_whatsapp_client,
)
from src.infrastructure.external.whatsapp.deduplicador import DeduplicadorDeMensajes
from src.infrastructure.persistence.encryption import TokenCipher
from src.infrastructure.persistence.supabase_client import (
    close_supabase_client,
    create_supabase_client,
)
from src.interfaces.api.errors import register_exception_handlers
from src.interfaces.api.middleware.request_context import RequestContextMiddleware
from src.interfaces.api.routers import health
from src.interfaces.webhooks import whatsapp as webhook_whatsapp

logger = structlog.get_logger(__name__)

DESCRIPCION = (
    "Asistente personal digital conversacional accesible por WhatsApp. "
    "Proyecto P18 – Seminario de Integración Profesional, USAL 2026."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Arranque y apagado ordenado del proceso.

    Acá se abren y cierran los recursos de larga vida: el cliente de Supabase
    (PB-003) y, más adelante, el cliente HTTP de WhatsApp (PB-004) y el grafo
    de LangGraph (PB-005). Crear el cliente por request agregaría un handshake
    TLS a cada mensaje.

    Si faltan credenciales, la app arranca igual en **modo degradado**: sin
    persistencia, pero con el liveness probe respondiendo. Es lo que permite
    desarrollar y testear sin un proyecto de Supabase.
    """
    settings: Settings = app.state.settings
    logger.info(
        "app.startup",
        entorno=settings.environment.value,
        version=settings.app_version,
    )

    await _iniciar_persistencia(app, settings)
    _iniciar_whatsapp(app, settings)
    try:
        yield
    finally:
        # Cada recurso en su propio try: que uno falle al cerrar no debe
        # impedir cerrar el otro.
        try:
            await close_supabase_client(app.state.supabase)
        except Exception as exc:  # el shutdown no puede romperse
            logger.warning("app.shutdown.error", recurso="supabase", tipo=type(exc).__name__)
        try:
            await close_whatsapp_client(app.state.whatsapp)
        except Exception as exc:  # el shutdown no puede romperse
            logger.warning("app.shutdown.error", recurso="whatsapp", tipo=type(exc).__name__)
        logger.info("app.shutdown")


async def _iniciar_persistencia(app: FastAPI, settings: Settings) -> None:
    """Abre el cliente de Supabase y el cifrador, si hay credenciales."""
    clave_de_cifrado = settings.token_encryption_key
    if not settings.supabase_configurado or clave_de_cifrado is None:
        logger.warning(
            "supabase.no_configurado",
            motivo="faltan SUPABASE_URL, SUPABASE_KEY o TOKEN_ENCRYPTION_KEY",
            consecuencia="la app arranca sin persistencia y /health/ready da 503",
        )
        return

    # El cifrador primero: si la clave es inválida conviene fallar acá y no
    # después de haber abierto la conexión.
    app.state.token_cipher = TokenCipher(clave_de_cifrado.get_secret_value())
    app.state.supabase = await create_supabase_client(settings)
    logger.info("supabase.conectado", url=settings.supabase_url)


def _iniciar_whatsapp(app: FastAPI, settings: Settings) -> None:
    """Abre el cliente HTTP hacia Graph, si hay credenciales."""
    if not settings.whatsapp_configurado:
        logger.warning(
            "whatsapp.no_configurado",
            motivo="faltan WHATSAPP_TOKEN o WHATSAPP_PHONE_NUMBER_ID",
            consecuencia="el webhook responde 200 pero no contesta mensajes",
        )
        return

    app.state.whatsapp = create_whatsapp_client(settings)
    logger.info("whatsapp.conectado", firma_exigida=settings.firma_exigida)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construye y configura la aplicación.

    Args:
        settings: Configuración a usar. Si es None, se lee del entorno.
    """
    settings = settings or get_settings()
    configure_logging(log_level=settings.log_level, json_logs=settings.use_json_logs)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPCION,
        lifespan=lifespan,
        # La documentación interactiva no se publica en producción.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings
    # Se declaran acá para que existan siempre, incluso en modo degradado o si
    # el lifespan todavía no corrió. Los completa `_iniciar_persistencia`.
    app.state.supabase = None
    app.state.token_cipher = None
    app.state.whatsapp = None
    # Vive todo el proceso: es lo que evita responder dos veces cuando Meta
    # reintrega el mismo mensaje.
    app.state.deduplicador_whatsapp = DeduplicadorDeMensajes()

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    # Fuera del prefijo /api/v1: lo consumen las probes de la plataforma.
    app.include_router(health.router)
    # Fuera del prefijo /api/v1: la URL la configura Meta y conviene que sea
    # estable e independiente del versionado de nuestra API.
    app.include_router(webhook_whatsapp.router)

    return app
