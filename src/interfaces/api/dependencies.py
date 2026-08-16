"""Dependencias compartidas de FastAPI.

Acá se resuelve la inyección de dependencias de la capa de interfaces: es el
lugar donde los casos de uso se arman con sus adaptadores concretos, para que
los routers no conozcan la infraestructura.

Nótese que los alias exportados anotan **las interfaces del dominio**
(`OAuthTokenRepository`, `UsuarioRepository`), nunca las clases de Supabase:
un router pide "un repositorio de tokens" y no se entera de quién lo implementa.
"""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import Depends
from starlette.requests import Request
from supabase import AsyncClient

from src.application.ports.whatsapp import MensajeroWhatsApp
from src.application.use_cases.procesar_mensaje_entrante import ProcesarMensajeEntrante
from src.domain.exceptions import ServiceUnavailableError
from src.domain.repositories.oauth_token_repository import OAuthTokenRepository
from src.domain.repositories.usuario_repository import UsuarioRepository
from src.infrastructure.config.settings import Settings
from src.infrastructure.external.whatsapp.cliente import ClienteWhatsApp
from src.infrastructure.external.whatsapp.deduplicador import DeduplicadorDeMensajes
from src.infrastructure.persistence.encryption import TokenCipher
from src.infrastructure.persistence.supabase_oauth_token_repository import (
    SupabaseOAuthTokenRepository,
)
from src.infrastructure.persistence.supabase_usuario_repository import SupabaseUsuarioRepository

MENSAJE_SIN_PERSISTENCIA = (
    "La persistencia no está configurada: faltan SUPABASE_URL, SUPABASE_KEY o TOKEN_ENCRYPTION_KEY."
)


def get_app_settings(request: Request) -> Settings:
    """Configuración asociada a esta aplicación.

    Se lee de `app.state` (no del caché global) para que los tests puedan
    construir la app con settings propias sin sobreescribir dependencias.
    """
    settings: Settings = request.app.state.settings
    return settings


def get_supabase_client(request: Request) -> AsyncClient:
    """Cliente de Supabase abierto en el lifespan.

    Raises:
        ServiceUnavailableError: Si la app arrancó en modo degradado. Se
            traduce a 503, no a 500: no es un bug, es una falta de config.
    """
    cliente: AsyncClient | None = request.app.state.supabase
    if cliente is None:
        raise ServiceUnavailableError(MENSAJE_SIN_PERSISTENCIA)
    return cliente


def get_token_cipher(request: Request) -> TokenCipher:
    """Cifrador de credenciales construido en el lifespan."""
    cipher: TokenCipher | None = request.app.state.token_cipher
    if cipher is None:
        raise ServiceUnavailableError(MENSAJE_SIN_PERSISTENCIA)
    return cipher


def get_oauth_token_repository(
    cliente: Annotated[AsyncClient, Depends(get_supabase_client)],
    cipher: Annotated[TokenCipher, Depends(get_token_cipher)],
) -> OAuthTokenRepository:
    """Repositorio de tokens OAuth2, con cifrado en reposo."""
    return SupabaseOAuthTokenRepository(cliente, cipher)


def get_usuario_repository(
    cliente: Annotated[AsyncClient, Depends(get_supabase_client)],
) -> UsuarioRepository:
    """Repositorio de usuarios."""
    return SupabaseUsuarioRepository(cliente)


# --- WhatsApp (PB-004) ------------------------------------------------------
#
# Estos providers son TOLERANTES: devuelven None en vez de lanzar cuando falta
# configuración. Los de arriba son estrictos porque un endpoint de negocio sin
# base de datos debe dar 503; el webhook no puede hacer eso, porque un 503 le
# dice a Meta que reintente durante horas contra un entorno que sabemos que no
# va a poder atenderlo.


def get_mensajero_whatsapp(request: Request) -> MensajeroWhatsApp | None:
    """Adaptador de envío por WhatsApp, o None si no está configurado."""
    cliente: httpx.AsyncClient | None = request.app.state.whatsapp
    if cliente is None:
        return None
    settings: Settings = request.app.state.settings
    return ClienteWhatsApp(cliente, settings)


def get_procesador_de_mensajes(request: Request) -> ProcesarMensajeEntrante | None:
    """Caso de uso ya armado, o None si falta la base o el mensajero.

    Se construye acá y no dentro del `BackgroundTask` porque ahí el `Request`
    ya no está disponible.
    """
    supabase: AsyncClient | None = request.app.state.supabase
    mensajero = get_mensajero_whatsapp(request)
    if supabase is None or mensajero is None:
        return None
    return ProcesarMensajeEntrante(SupabaseUsuarioRepository(supabase), mensajero)


def get_deduplicador_whatsapp(request: Request) -> DeduplicadorDeMensajes:
    """Caché de deduplicación, con vida de proceso."""
    deduplicador: DeduplicadorDeMensajes = request.app.state.deduplicador_whatsapp
    return deduplicador


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SupabaseDep = Annotated[AsyncClient, Depends(get_supabase_client)]
OAuthTokenRepositoryDep = Annotated[OAuthTokenRepository, Depends(get_oauth_token_repository)]
UsuarioRepositoryDep = Annotated[UsuarioRepository, Depends(get_usuario_repository)]
MensajeroWhatsAppDep = Annotated[MensajeroWhatsApp | None, Depends(get_mensajero_whatsapp)]
ProcesadorDeMensajesDep = Annotated[
    ProcesarMensajeEntrante | None, Depends(get_procesador_de_mensajes)
]
DeduplicadorWhatsAppDep = Annotated[DeduplicadorDeMensajes, Depends(get_deduplicador_whatsapp)]
