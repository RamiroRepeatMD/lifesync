"""Tests del health check y del contexto de request."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.interfaces.api.middleware.request_context import REQUEST_ID_HEADER


def test_health_responde_ok(client: TestClient) -> None:
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["service"] == "LifeSync"
    assert cuerpo["environment"] == "testing"
    assert cuerpo["timestamp"]


def test_health_genera_request_id(client: TestClient) -> None:
    respuesta = client.get("/health")

    assert respuesta.headers[REQUEST_ID_HEADER]


def test_health_respeta_el_request_id_entrante(client: TestClient) -> None:
    respuesta = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc-123"})

    assert respuesta.headers[REQUEST_ID_HEADER] == "trace-abc-123"


def test_ruta_inexistente_devuelve_404(client: TestClient) -> None:
    assert client.get("/no-existe").status_code == 404
