"""Health checks (RNF – Operabilidad; base de PB-006).

Quedan fuera del prefijo de la API porque los consumen las probes de la
plataforma de hosting (Railway/Render), no clientes de negocio.

Hay dos, y la distinción importa para el despliegue:

- `GET /health` — **liveness**: ¿el proceso está vivo? No toca dependencias.
  Si esto falla, la plataforma debe reiniciar el contenedor.
- `GET /health/ready` — **readiness**: ¿puede atender tráfico de verdad?
  Verifica Supabase. Si falla, la plataforma debe sacarlo del balanceador
  pero no reiniciarlo: la base puede volver sola.

Con PB-004 se suma WhatsApp a las dependencias verificadas por readiness.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field
from starlette.requests import Request

from src.infrastructure.persistence.supabase_client import ping
from src.interfaces.api.dependencies import SettingsDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: Literal["ok"] = "ok"
    service: str = Field(description="Nombre del servicio")
    version: str = Field(description="Versión desplegada")
    environment: str = Field(description="Entorno de ejecución")
    timestamp: datetime = Field(description="Momento de la respuesta (UTC)")


class DependencyStatus(BaseModel):
    """Estado de una dependencia externa."""

    name: str = Field(description="Nombre de la dependencia")
    ready: bool = Field(description="Si responde correctamente")
    detail: str | None = Field(default=None, description="Motivo cuando no está lista")


class ReadinessResponse(BaseModel):
    """Resultado del readiness probe."""

    status: Literal["ready", "not_ready"]
    dependencies: list[DependencyStatus]
    timestamp: datetime = Field(description="Momento de la respuesta (UTC)")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Devuelve 200 si el servicio está en ejecución. No consulta dependencias.",
)
async def health(settings: SettingsDep) -> HealthResponse:
    """Reporta que el proceso está vivo y con qué configuración corre."""
    return HealthResponse(
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value,
        timestamp=datetime.now(UTC),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Devuelve 200 si las dependencias externas responden, 503 si no.",
    responses={503: {"model": ReadinessResponse, "description": "Alguna dependencia no responde"}},
)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """Verifica que las dependencias externas estén operativas.

    Devuelve 503 en vez de lanzar una excepción: un health check informa
    estado, no es un error de la aplicación.
    """
    dependencias = [await _estado_de_supabase(request)]
    listo = all(dependencia.ready for dependencia in dependencias)

    if not listo:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if listo else "not_ready",
        dependencies=dependencias,
        timestamp=datetime.now(UTC),
    )


async def _estado_de_supabase(request: Request) -> DependencyStatus:
    """Comprueba la conexión con Supabase sin propagar errores."""
    cliente = request.app.state.supabase
    if cliente is None:
        return DependencyStatus(
            name="supabase",
            ready=False,
            detail="No configurado: faltan SUPABASE_URL, SUPABASE_KEY o TOKEN_ENCRYPTION_KEY.",
        )

    if not await ping(cliente):
        return DependencyStatus(
            name="supabase",
            ready=False,
            detail="No responde a la consulta de prueba.",
        )

    return DependencyStatus(name="supabase", ready=True)
