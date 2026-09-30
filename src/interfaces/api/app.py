"""Factory de la aplicación FastAPI.

`create_app()` es el punto donde se ensambla el sistema: se lee la
configuración, se configura el logging, se registran middlewares, manejadores
de error y routers. Es una factory (y no un `app` global) para que los tests
puedan levantar instancias aisladas con configuración propia.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI

from src.application.ports.calendario import Calendario
from src.application.ports.correos import Correos
from src.application.ports.tareas import Tareas
from src.application.use_cases.conectar_google import ConectarGoogle
from src.infrastructure.config.logging import configure_logging
from src.infrastructure.config.settings import Environment, Settings, get_settings
from src.infrastructure.external.google.oauth import (
    close_google_oauth_client,
    create_google_oauth_client,
)
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
from src.infrastructure.persistence.supabase_oauth_token_repository import (
    SupabaseOAuthTokenRepository,
)
from src.interfaces.api.errors import register_exception_handlers
from src.interfaces.api.middleware.request_context import RequestContextMiddleware
from src.interfaces.api.routers import health, oauth_google
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
    # El orden importa: el agente recibe el calendario y el checkpointer, así
    # que sus proveedores arrancan antes.
    _iniciar_google_oauth(app, settings)
    await _iniciar_memoria_del_agente(app, settings)
    _iniciar_agente(app, settings)
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
        try:
            await close_google_oauth_client(app.state.google_oauth)
        except Exception as exc:  # el shutdown no puede romperse
            logger.warning("app.shutdown.error", recurso="google_oauth", tipo=type(exc).__name__)
        try:
            from src.infrastructure.llm.checkpointer import cerrar_pool

            await cerrar_pool(app.state.checkpointer_pool)
        except Exception as exc:  # el shutdown no puede romperse
            logger.warning("app.shutdown.error", recurso="checkpointer", tipo=type(exc).__name__)
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


async def _iniciar_memoria_del_agente(app: FastAPI, settings: Settings) -> None:
    """Abre la memoria persistida del agente, si está configurada (PB-013).

    Sin `SUPABASE_DB_URL` el agente funciona igual con memoria RAM, pero cada
    redeploy borra las conversaciones y las confirmaciones pendientes: se avisa
    fuerte porque es un modo degradado, no una preferencia.
    """
    if not settings.checkpointer_configurado:
        logger.warning(
            "checkpointer.no_configurado",
            motivo="falta SUPABASE_DB_URL (o TOKEN_ENCRYPTION_KEY)",
            consecuencia="la memoria del agente vive en RAM y muere en cada redeploy",
        )
        return

    try:
        # El import diferido va ADENTRO del try, y no es prolijidad: quedó
        # afuera en el deploy del 01/09 y un ImportError (faltaba libpq en la
        # imagen) mató el arranque completo en vez de degradar a RAM. La
        # memoria es opcional; su import también tiene que serlo.
        from src.infrastructure.llm.checkpointer import crear_checkpointer_postgres

        pool, saver = await crear_checkpointer_postgres(settings)
    except Exception as exc:
        # Una base inalcanzable —o una dependencia rota— no puede impedir que
        # el bot conteste: se degrada a RAM y queda el error para diagnosticar.
        logger.error("checkpointer.fallo_al_iniciar", tipo=type(exc).__name__)
        return

    app.state.checkpointer_pool = pool
    app.state.checkpointer = saver


def _iniciar_agente(app: FastAPI, settings: Settings) -> None:
    """Compila el grafo del agente, si hay API key (PB-005).

    Se hace una sola vez: el grafo lleva adentro la memoria de todas las
    conversaciones (RF-09), así que rearmarlo por mensaje sería empezar cada
    charla de cero.
    """
    if not settings.agente_configurado:
        logger.warning(
            "agente.no_configurado",
            motivo="falta GOOGLE_API_KEY",
            consecuencia="los comandos siguen andando; el lenguaje natural no",
        )
        return

    # Import diferido: sin key no se paga el costo de importar todo el stack de
    # LangChain, que es casi un segundo de arranque.
    from src.infrastructure.llm.agente_gemini import crear_agente_gemini

    app.state.agente = crear_agente_gemini(
        settings,
        _calendario_de(app),
        _tareas_de(app),
        _correos_de(app),
        checkpointer=app.state.checkpointer,
    )


def _conectar_google_de(
    app: FastAPI, capacidad: str
) -> tuple[httpx.AsyncClient, ConectarGoogle] | None:
    """Las piezas que necesita cualquier integración con Google, o None.

    Es `None` cuando falta cualquiera —Supabase, el cifrador o el cliente de
    Google—, y entonces el agente simplemente no ofrece esa capacidad. Es la
    degradación de siempre: sin la pieza, la capacidad no existe, pero el resto
    del sistema sigue en pie. Cada integración construye su `ConectarGoogle`:
    son objetos baratos sin estado propio, y así ninguna depende de otra.
    """
    supabase = app.state.supabase
    cipher = app.state.token_cipher
    http = app.state.google_oauth
    if supabase is None or cipher is None or http is None:
        logger.info(f"{capacidad}.no_disponible", motivo="falta Supabase o la conexión con Google")
        return None

    from src.infrastructure.external.google.oauth import create_autorizador_google

    settings: Settings = app.state.settings
    conectar = ConectarGoogle(
        SupabaseOAuthTokenRepository(supabase, cipher),
        create_autorizador_google(http, settings),
    )
    return http, conectar


def _calendario_de(app: FastAPI) -> Calendario | None:
    """El adaptador de Google Calendar (PB-015), si están sus piezas."""
    piezas = _conectar_google_de(app, "calendar")
    if piezas is None:
        return None
    from src.infrastructure.external.google.calendario import create_calendario_google

    return create_calendario_google(*piezas)


def _tareas_de(app: FastAPI) -> Tareas | None:
    """El adaptador de Google Tasks (PB-028), si están sus piezas."""
    piezas = _conectar_google_de(app, "tasks")
    if piezas is None:
        return None
    from src.infrastructure.external.google.tareas import create_tareas_google

    return create_tareas_google(*piezas)


def _correos_de(app: FastAPI) -> Correos | None:
    """El adaptador de Gmail (PB-033), si están sus piezas."""
    piezas = _conectar_google_de(app, "gmail")
    if piezas is None:
        return None
    from src.infrastructure.external.google.gmail import create_gmail_google

    return create_gmail_google(*piezas)


def _iniciar_google_oauth(app: FastAPI, settings: Settings) -> None:
    """Abre el cliente HTTP hacia Google, si hay credenciales de OAuth (PB-009).

    A diferencia de las otras nueve variables, éstas NO son obligatorias en
    producción: sin ellas todo lo demás funciona y /conectar avisa. Es una
    integración, no el circuito central.
    """
    if not settings.google_oauth_configurado:
        logger.warning(
            "google_oauth.no_configurado",
            motivo="faltan GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET o GOOGLE_REDIRECT_URI",
            consecuencia="no se pueden conectar cuentas de Google; el resto anda igual",
        )
        return

    app.state.google_oauth = create_google_oauth_client()
    logger.info("google_oauth.listo")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construye y configura la aplicación.

    Args:
        settings: Configuración a usar. Si es None, se lee del entorno.
    """
    settings = settings or get_settings()
    configure_logging(
        log_level=settings.log_level,
        json_logs=settings.use_json_logs,
        # En testing se desactiva el caché de loggers para que `capture_logs`
        # pueda interceptarlos. Ver el docstring de `configure_logging`.
        cache_loggers=settings.environment is not Environment.TESTING,
    )

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
    app.state.agente = None
    app.state.google_oauth = None
    app.state.checkpointer = None
    app.state.checkpointer_pool = None
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
    # Fuera del prefijo /api/v1: la URL queda registrada en Google Cloud
    # Console y moverla obliga a reconfigurarla allá (PB-009).
    app.include_router(oauth_google.router)

    return app
