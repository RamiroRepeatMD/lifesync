"""Tests de la configuración de logging estructurado.

El grupo importante es el de RF-18: httpx registra en INFO la URL completa de
cada request, y los filtros de PostgREST viajan en el query string. Sin
silenciarlo, cada búsqueda de usuario por teléfono dejaría el número de
WhatsApp en texto plano en los logs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import structlog

from src.infrastructure.config.logging import (
    _LOGGERS_SILENCIADOS,
    NIVEL_LOGGERS_SILENCIADOS,
    configure_logging,
)

TELEFONO = "+5491122334455"
URL_CON_TELEFONO = f"https://x.supabase.co/rest/v1/usuarios?telefono_whatsapp=eq.{TELEFONO}"

# Lo que el stack de Gemini escribiría en DEBUG: el prompt entero, con la
# conversación adentro (PB-005).
CONTENIDO_PRIVADO = "mi diagnostico medico es"
PAYLOAD_CON_LA_CHARLA = f'{{"contents": [{{"parts": [{{"text": "{CONTENIDO_PRIVADO}"}}]}}]}}'


@pytest.fixture(autouse=True)
def _restaurar_logging() -> Iterator[None]:
    """Devuelve el logging global a su estado previo al terminar cada test."""
    root = logging.getLogger()
    handlers_previos = root.handlers[:]
    nivel_previo = root.level
    niveles_previos = {nombre: logging.getLogger(nombre).level for nombre in _LOGGERS_SILENCIADOS}

    yield

    root.handlers[:] = handlers_previos
    root.setLevel(nivel_previo)
    for nombre, nivel in niveles_previos.items():
        logging.getLogger(nombre).setLevel(nivel)


# --- RF-18: no filtrar datos personales por los logs de librerías -----------


@pytest.mark.parametrize("nombre", _LOGGERS_SILENCIADOS)
def test_los_loggers_ruidosos_quedan_en_warning(nombre: str) -> None:
    configure_logging(log_level="INFO", json_logs=False)

    assert logging.getLogger(nombre).level == NIVEL_LOGGERS_SILENCIADOS


@pytest.mark.parametrize("nombre", _LOGGERS_SILENCIADOS)
def test_ni_siquiera_en_debug_se_habilitan(nombre: str) -> None:
    """Depurar no debe ser una puerta trasera para volcar datos personales."""
    configure_logging(log_level="DEBUG", json_logs=False)

    assert logging.getLogger(nombre).isEnabledFor(logging.INFO) is False


# Nota: estos tests usan `capsys` y no `caplog`. `configure_logging` limpia los
# handlers de la raíz —se lleva puesto el de caplog— y escribe por su propio
# StreamHandler a stdout. Con caplog el texto capturado queda vacío y las
# aserciones negativas pasarían sin probar nada.


def test_httpx_no_emite_la_url_de_la_consulta(capsys: pytest.CaptureFixture[str]) -> None:
    """El caso concreto que motiva el silenciado (PB-004)."""
    configure_logging(log_level="DEBUG", json_logs=False)

    logging.getLogger("httpx").info("HTTP Request: GET %s", URL_CON_TELEFONO)

    salida = capsys.readouterr().out
    assert TELEFONO not in salida
    assert salida == ""


def test_httpx_sigue_reportando_sus_errores(capsys: pytest.CaptureFixture[str]) -> None:
    """Silenciar el ruido no debe ocultar fallas reales de red."""
    configure_logging(log_level="INFO", json_logs=False)

    logging.getLogger("httpx").warning("connection failed")

    assert "connection failed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "nombre",
    ["google_genai", "langchain_google_genai", "langchain_core", "langgraph"],
)
def test_el_stack_del_agente_no_emite_la_conversacion(
    nombre: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """El vector de PB-005, y el más grave de los tres.

    Estas librerías vuelcan en DEBUG el payload entero que le mandan a Gemini:
    el system prompt, el historial y lo que acaba de escribir la persona. No es
    un identificador, es la charla completa.
    """
    configure_logging(log_level="DEBUG", json_logs=False)

    logging.getLogger(nombre).debug("request payload: %s", PAYLOAD_CON_LA_CHARLA)

    salida = capsys.readouterr().out
    assert CONTENIDO_PRIVADO not in salida
    assert salida == ""


def test_el_stack_del_agente_sigue_reportando_sus_errores(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Si Gemini rechaza la petición, eso tiene que verse."""
    configure_logging(log_level="INFO", json_logs=False)

    logging.getLogger("google_genai").warning("429 RESOURCE_EXHAUSTED")

    assert "429 RESOURCE_EXHAUSTED" in capsys.readouterr().out


# --- El resto de la configuración sigue en pie ------------------------------


def test_nuestros_loggers_no_quedan_silenciados(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(log_level="INFO", json_logs=False)

    logging.getLogger("src.interfaces.api.app").info("app.startup")

    assert "app.startup" in capsys.readouterr().out


def test_el_nivel_raiz_respeta_la_configuracion() -> None:
    configure_logging(log_level="WARNING", json_logs=False)

    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_es_idempotente() -> None:
    """Se llama una vez por `create_app`, y los tests crean varias apps."""
    configure_logging(log_level="INFO", json_logs=False)
    configure_logging(log_level="INFO", json_logs=False)

    assert len(logging.getLogger().handlers) == 1


# --- La trampa de capture_logs (PB-009) -------------------------------------
#
# Tercera trampa de la familia, y del mismo tipo que la de caplog: el test pasa
# por el motivo equivocado. Apareció al escribir la primera aserción POSITIVA
# sobre un log; las negativas la venían tapando desde PB-004.
#
# Mecanismo, verificado quitando el arreglo y viendo fallar los tests: con
# `cache_logger_on_first_use=True`, el proxy de structlog **reemplaza su propio
# `bind`** por el logger ya armado en el primer uso. Un `capture_logs`
# posterior cambia los procesadores globales, pero ese logger ya no los
# consulta: `eventos` queda vacío y `assert secreto not in eventos` no prueba
# nada.
#
# Por eso `create_app` pasa `cache_loggers=False` en ENVIRONMENT=testing.


def test_capture_logs_captura_de_verdad_sin_cache() -> None:
    """Si esto falla, TODAS las aserciones negativas de RF-18 son papel pintado."""
    configure_logging(log_level="INFO", json_logs=False, cache_loggers=False)
    logger = structlog.get_logger("sonda.sin_cache")
    logger.info("evento.calienta_el_cache")

    with structlog.testing.capture_logs() as eventos:
        logger.info("evento.esperado")

    assert [e["event"] for e in eventos] == ["evento.esperado"]


def test_los_tests_corren_sin_cache_de_loggers() -> None:
    """El arreglo tiene que estar puesto donde se arma la app, no sólo acá."""
    from src.infrastructure.config.settings import Environment, Settings
    from src.interfaces.api.app import create_app

    create_app(Settings(_env_file=None, environment=Environment.TESTING, log_level="WARNING"))

    assert structlog.get_config()["cache_logger_on_first_use"] is False
