"""Tests de la configuración por entorno."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.infrastructure.config.settings import Environment, Settings

# Producción exige persistencia cifrada (RF-18), así que los tests que
# construyen Settings en ese entorno tienen que pasar las tres credenciales.
CREDENCIALES: dict[str, object] = {
    "supabase_url": "https://proyecto.supabase.co",
    "supabase_key": "service-role-de-prueba",
    "token_encryption_key": "clave-fernet-de-prueba",
}


def _settings(**overrides: object) -> Settings:
    """Settings aisladas del `.env` local, para que el test sea determinístico."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _settings_produccion(**overrides: object) -> Settings:
    """Settings de producción, con las credenciales que ese entorno exige."""
    return _settings(environment=Environment.PRODUCTION, **CREDENCIALES, **overrides)


# --- Entorno y logging ------------------------------------------------------


def test_defaults_apuntan_a_desarrollo() -> None:
    settings = _settings()

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.is_production is False


def test_desarrollo_usa_logs_de_consola() -> None:
    assert _settings(environment=Environment.DEVELOPMENT).use_json_logs is False


@pytest.mark.parametrize("entorno", [Environment.TESTING, Environment.STAGING])
def test_fuera_de_desarrollo_usa_logs_json(entorno: Environment) -> None:
    assert _settings(environment=entorno).use_json_logs is True


def test_produccion_usa_logs_json() -> None:
    assert _settings_produccion().use_json_logs is True


def test_log_json_explicito_gana_sobre_el_entorno() -> None:
    assert _settings_produccion(log_json=False).use_json_logs is False


def test_produccion_se_detecta_correctamente() -> None:
    assert _settings_produccion().is_production is True


# --- Supabase y cifrado (PB-003) --------------------------------------------


def test_sin_credenciales_la_persistencia_queda_deshabilitada() -> None:
    """Modo degradado: la app arranca igual, sin base."""
    assert _settings().supabase_configurado is False


def test_con_las_tres_credenciales_la_persistencia_queda_habilitada() -> None:
    assert _settings(**CREDENCIALES).supabase_configurado is True


@pytest.mark.parametrize(
    "faltante",
    ["supabase_url", "supabase_key", "token_encryption_key"],
)
def test_falta_cualquier_credencial_y_la_persistencia_queda_deshabilitada(
    faltante: str,
) -> None:
    """Sin clave de cifrado tampoco se persiste: guardar en claro violaría RF-18."""
    incompletas = {k: v for k, v in CREDENCIALES.items() if k != faltante}

    assert _settings(**incompletas).supabase_configurado is False


def test_produccion_sin_credenciales_no_arranca() -> None:
    """Nunca se despliega sin persistencia cifrada (RF-18)."""
    with pytest.raises(ValidationError, match="TOKEN_ENCRYPTION_KEY"):
        _settings(environment=Environment.PRODUCTION)


def test_las_claves_no_aparecen_en_el_repr_de_settings() -> None:
    """Si la config termina en un traceback o en un log, no filtra secretos."""
    settings = _settings(**CREDENCIALES)

    representacion = repr(settings)
    assert "service-role-de-prueba" not in representacion
    assert "clave-fernet-de-prueba" not in representacion
    assert "**********" in representacion


def test_la_clave_sigue_siendo_recuperable_para_usarla() -> None:
    settings = _settings(**CREDENCIALES)

    assert settings.supabase_key is not None
    assert settings.supabase_key.get_secret_value() == "service-role-de-prueba"
