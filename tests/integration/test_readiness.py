"""Tests de los health checks con y sin persistencia configurada."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from tests.dobles import FakeSupabaseClient

TABLA_PING = "usuarios"


# --- Modo degradado: sin credenciales de Supabase ---------------------------


def test_liveness_sigue_ok_sin_supabase(client: TestClient) -> None:
    """Lo entregado en PB-002 no se rompe: /health no depende de la base."""
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"


def _dependencia(respuesta: object, nombre: str) -> dict[str, object]:
    """Busca una dependencia por nombre y no por posición.

    Indexar por posición haría que agregar una dependencia nueva —WhatsApp en
    PB-004, el agente en PB-005— rompiera tests que no tienen nada que ver.
    """
    cuerpo = respuesta.json()  # type: ignore[attr-defined]
    for dependencia in cuerpo["dependencies"]:
        if dependencia["name"] == nombre:
            return dict(dependencia)
    raise AssertionError(f"No se reportó la dependencia '{nombre}'.")


def test_readiness_da_503_sin_supabase(client: TestClient) -> None:
    respuesta = client.get("/health/ready")

    assert respuesta.status_code == 503
    assert respuesta.json()["status"] == "not_ready"

    supabase = _dependencia(respuesta, "supabase")
    assert supabase["ready"] is False
    assert "No configurado" in str(supabase["detail"])


def test_readiness_reporta_whatsapp_sin_configurar(client: TestClient) -> None:
    whatsapp = _dependencia(client.get("/health/ready"), "whatsapp")

    assert whatsapp["ready"] is False
    assert "No configurado" in str(whatsapp["detail"])


# --- Con persistencia simulada ----------------------------------------------


def test_readiness_reporta_supabase_listo_cuando_responde(
    client_con_supabase: TestClient, supabase_falso: FakeSupabaseClient
) -> None:
    supabase_falso.respuestas[TABLA_PING] = []

    respuesta = client_con_supabase.get("/health/ready")

    assert _dependencia(respuesta, "supabase") == {
        "name": "supabase",
        "ready": True,
        "detail": None,
    }


def test_readiness_da_200_con_todo_configurado(
    client_con_whatsapp_y_supabase: TestClient, supabase_falso: FakeSupabaseClient
) -> None:
    supabase_falso.respuestas[TABLA_PING] = []

    respuesta = client_con_whatsapp_y_supabase.get("/health/ready")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ready"


def test_readiness_da_503_cuando_supabase_no_responde(
    client_con_supabase: TestClient, supabase_falso: FakeSupabaseClient
) -> None:
    supabase_falso.errores[TABLA_PING] = httpx.ConnectError("sin conexión")

    respuesta = client_con_supabase.get("/health/ready")

    assert respuesta.status_code == 503
    assert respuesta.json()["status"] == "not_ready"
    assert _dependencia(respuesta, "supabase")["ready"] is False


def test_el_readiness_consulta_la_tabla_usuarios(
    client_con_supabase: TestClient, supabase_falso: FakeSupabaseClient
) -> None:
    supabase_falso.respuestas[TABLA_PING] = []

    client_con_supabase.get("/health/ready")

    llamada = supabase_falso.ultima_llamada()
    assert llamada.tabla == TABLA_PING
    assert llamada.operacion == "select"


def test_el_readiness_no_expone_credenciales(
    client_con_supabase: TestClient, supabase_falso: FakeSupabaseClient
) -> None:
    """La respuesta es pública para las probes: no debe filtrar configuración."""
    supabase_falso.respuestas[TABLA_PING] = []

    cuerpo = client_con_supabase.get("/health/ready").text

    for prohibido in ("supabase_key", "SUPABASE_KEY", "token_encryption", "eyJ"):
        assert prohibido not in cuerpo
