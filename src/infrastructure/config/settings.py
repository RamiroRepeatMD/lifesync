"""Configuración de la aplicación, cargada desde variables de entorno.

La configuración es un detalle de infraestructura: el dominio y los casos de
uso nunca leen variables de entorno, reciben lo que necesitan por constructor.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
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

    # --- WhatsApp Cloud API (PB-004) ---
    whatsapp_token: SecretStr | None = None  # System User token para llamar a Graph
    whatsapp_phone_number_id: str | None = None  # nuestro número; arma la URL de envío
    whatsapp_verify_token: SecretStr | None = None  # sólo para el handshake GET
    whatsapp_app_secret: SecretStr | None = None  # firma HMAC de los POST; NO es el verify token

    @field_validator(
        "supabase_url",
        "supabase_key",
        "supabase_jwt_secret",
        "token_encryption_key",
        "whatsapp_token",
        "whatsapp_phone_number_id",
        "whatsapp_verify_token",
        "whatsapp_app_secret",
        mode="before",
    )
    @classmethod
    def _vacio_es_ausente(cls, valor: object) -> object:
        """Trata una variable vacía como no configurada.

        En el panel de una plataforma es trivial dejar `WHATSAPP_APP_SECRET=`
        sin valor. Sin esto, el campo quedaría en `SecretStr('')`, que **no es
        None**, así que las comprobaciones de tipo `is not None` lo darían por
        configurado: la app arrancaría en producción y el webhook validaría las
        firmas HMAC contra un secreto vacío, que cualquiera puede reproducir
        (RF-18).

        Normalizar acá arregla a todos los consumidores de una vez, en vez de
        pedirle a cada uno que se acuerde de comprobar el caso.
        """
        if isinstance(valor, str) and not valor.strip():
            return None
        return valor

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

    @property
    def whatsapp_configurado(self) -> bool:
        """True si se puede ENVIAR por WhatsApp (token + número propio)."""
        return all((self.whatsapp_token, self.whatsapp_phone_number_id))

    @property
    def firma_exigida(self) -> bool:
        """True si hay que validar `X-Hub-Signature-256` en los POST entrantes.

        Se separa a propósito de `whatsapp_configurado`: colapsarlas dejaría a
        un desarrollador sin app secret sin poder enviar, o publicaría un
        webhook que acepta POSTs de cualquiera.

        En producción es siempre True (el validador de abajo garantiza que
        entonces exista el secreto). En desarrollo se exige sólo si el secreto
        está configurado, para poder probar con curl sin firmar.
        """
        return self.is_production or self.whatsapp_app_secret is not None

    @model_validator(mode="after")
    def _exigir_credenciales_en_produccion(self) -> Settings:
        """Impide desplegar a producción sin persistencia cifrada ni webhook firmado.

        En desarrollo y testing la app arranca igual, en modo degradado: es lo
        que permite trabajar y correr los tests sin proyecto de Supabase ni
        credenciales de Meta.

        El orden importa: Supabase se valida primero para no cambiar el mensaje
        que ya esperan los tests de PB-003.
        """
        if not self.is_production:
            return self

        if not self.supabase_configurado:
            raise ValueError(
                "En producción son obligatorias SUPABASE_URL, SUPABASE_KEY y TOKEN_ENCRYPTION_KEY."
            )
        if not self.whatsapp_configurado:
            raise ValueError(
                "En producción son obligatorias WHATSAPP_TOKEN y WHATSAPP_PHONE_NUMBER_ID."
            )
        if self.whatsapp_verify_token is None:
            raise ValueError("En producción es obligatoria WHATSAPP_VERIFY_TOKEN.")
        if self.whatsapp_app_secret is None:
            raise ValueError(
                "En producción es obligatoria WHATSAPP_APP_SECRET: sin ella el webhook "
                "aceptaría POSTs de cualquiera (RF-18)."
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
