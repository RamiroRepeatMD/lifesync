"""Configuración de la aplicación, cargada desde variables de entorno.

La configuración es un detalle de infraestructura: el dominio y los casos de
uso nunca leen variables de entorno, reciben lo que necesitan por constructor.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Environment(StrEnum):
    """Entornos de ejecución soportados."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Configuración tipada e inmutable de LifeSync.

    Los nombres de los campos se mapean automáticamente a variables de entorno
    en mayúsculas: `environment` -> `ENVIRONMENT`, `log_level` -> `LOG_LEVEL`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolera las variables de PBs futuros que ya están en .env
        frozen=True,
    )

    # --- Identidad de la app ---
    app_name: str = "LifeSync"
    app_version: str = "0.1.0"
    environment: Environment = Environment.DEVELOPMENT

    # --- Logging ---
    log_level: LogLevel = "INFO"
    log_json: bool | None = None  # None = deducir del entorno

    # --- Servidor HTTP ---
    host: str = "0.0.0.0"  # noqa: S104 - necesario para escuchar dentro del contenedor
    port: int = 8000
    reload: bool = False

    # --- API ---
    api_prefix: str = "/api/v1"

    # --- Supabase y cifrado de credenciales (PB-003) ---
    # Se usan SecretStr para que un `repr` o un log de la config no filtre
    # nada: pydantic los muestra como "**********".
    supabase_url: str | None = None
    supabase_key: SecretStr | None = None
    supabase_jwt_secret: SecretStr | None = None  # se usa recién en PB-009
    token_encryption_key: SecretStr | None = None

    @property
    def is_production(self) -> bool:
        """True si corremos en producción (endurece defaults, oculta /docs)."""
        return self.environment is Environment.PRODUCTION

    @property
    def supabase_configurado(self) -> bool:
        """True si están las tres credenciales necesarias para persistir.

        La clave de cifrado cuenta: guardar tokens sin cifrarlos violaría
        RF-18, así que sin ella la persistencia queda deshabilitada.
        """
        return all(
            (
                self.supabase_url,
                self.supabase_key,
                self.token_encryption_key,
            )
        )

    @model_validator(mode="after")
    def _exigir_credenciales_en_produccion(self) -> Settings:
        """Impide desplegar a producción sin persistencia cifrada (RF-18).

        En desarrollo y testing la app arranca igual, en modo degradado: es lo
        que permite trabajar y correr los tests sin un proyecto de Supabase.
        """
        if self.is_production and not self.supabase_configurado:
            raise ValueError(
                "En producción son obligatorias SUPABASE_URL, SUPABASE_KEY y TOKEN_ENCRYPTION_KEY."
            )
        return self

    @property
    def use_json_logs(self) -> bool:
        """JSON fuera de desarrollo, salvo que se fuerce con `LOG_JSON`."""
        if self.log_json is not None:
            return self.log_json
        return self.environment is not Environment.DEVELOPMENT


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración del proceso (cacheada: se lee una sola vez)."""
    return Settings()
