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

# PB-004 sumó WhatsApp a lo que producción exige.
CREDENCIALES_WHATSAPP: dict[str, object] = {
    "whatsapp_token": "token-de-graph-de-prueba",
    "whatsapp_phone_number_id": "106540352242922",
    "whatsapp_verify_token": "verify-de-prueba",
    "whatsapp_app_secret": "app-secret-de-prueba",
}


def _settings(**overrides: object) -> Settings:
    """Settings aisladas del `.env` local, para que el test sea determinístico."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _settings_produccion(**overrides: object) -> Settings:
    """Settings de producción, con las credenciales que ese entorno exige."""
    return _settings(
        environment=Environment.PRODUCTION,
        **CREDENCIALES,
        **CREDENCIALES_WHATSAPP,
        **overrides,
    )


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


# --- WhatsApp (PB-004) ------------------------------------------------------


def test_sin_credenciales_whatsapp_queda_deshabilitado() -> None:
    assert _settings().whatsapp_configurado is False


def test_con_token_y_numero_whatsapp_queda_habilitado() -> None:
    assert _settings(**CREDENCIALES_WHATSAPP).whatsapp_configurado is True


@pytest.mark.parametrize("faltante", ["whatsapp_token", "whatsapp_phone_number_id"])
def test_falta_una_credencial_y_whatsapp_queda_deshabilitado(faltante: str) -> None:
    incompletas = {k: v for k, v in CREDENCIALES_WHATSAPP.items() if k != faltante}

    assert _settings(**incompletas).whatsapp_configurado is False


def test_la_firma_no_se_exige_en_desarrollo_sin_app_secret() -> None:
    """Permite probar el webhook con curl sin tener que firmar a mano."""
    assert _settings(whatsapp_token="x", whatsapp_phone_number_id="1").firma_exigida is False


def test_la_firma_se_exige_apenas_hay_app_secret() -> None:
    assert _settings(whatsapp_app_secret="secreto").firma_exigida is True


def test_la_firma_siempre_se_exige_en_produccion() -> None:
    assert _settings_produccion().firma_exigida is True


def test_poder_enviar_es_independiente_de_exigir_firma() -> None:
    """Colapsarlas dejaría a un dev sin app secret sin poder enviar mensajes."""
    settings = _settings(whatsapp_token="x", whatsapp_phone_number_id="1")

    assert settings.whatsapp_configurado is True
    assert settings.firma_exigida is False


@pytest.mark.parametrize(
    ("faltante", "esperado"),
    [
        ("whatsapp_token", "WHATSAPP_TOKEN"),
        ("whatsapp_phone_number_id", "WHATSAPP_PHONE_NUMBER_ID"),
        ("whatsapp_verify_token", "WHATSAPP_VERIFY_TOKEN"),
        ("whatsapp_app_secret", "WHATSAPP_APP_SECRET"),
    ],
)
def test_produccion_sin_una_credencial_de_whatsapp_no_arranca(faltante: str, esperado: str) -> None:
    """Nunca se despliega un webhook público sin validación de firma (RF-18)."""
    incompletas = {k: v for k, v in CREDENCIALES_WHATSAPP.items() if k != faltante}

    with pytest.raises(ValidationError, match=esperado):
        _settings(environment=Environment.PRODUCTION, **CREDENCIALES, **incompletas)


def test_supabase_se_valida_antes_que_whatsapp() -> None:
    """El orden importa: mantiene el mensaje que ya esperaban los tests de PB-003."""
    with pytest.raises(ValidationError, match="TOKEN_ENCRYPTION_KEY"):
        _settings(environment=Environment.PRODUCTION, **CREDENCIALES_WHATSAPP)


def test_las_credenciales_de_whatsapp_no_aparecen_en_el_repr() -> None:
    representacion = repr(_settings(**CREDENCIALES_WHATSAPP))

    assert "token-de-graph-de-prueba" not in representacion
    assert "app-secret-de-prueba" not in representacion
    assert "verify-de-prueba" not in representacion
