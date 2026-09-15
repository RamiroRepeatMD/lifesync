"""Adaptador de Google Tasks (PB-028).

Implementa el puerto `Tareas` contra la API REST de Tasks, con el mismo trío
que el calendario: el cliente httpx compartido de Google y las credenciales
vigentes que administra `ConectarGoogle` (refresco perezoso incluido).

Decisiones que conviene conocer:

- **Sólo la lista `@default`.** Todo el mundo la tiene y es donde caen las
  tareas creadas sin más datos. Soportar varias listas es post-MVP.
- **`due` es sólo una fecha.** La API acepta un RFC3339 completo pero
  descarta la hora (documentado por Google): acá se manda la medianoche UTC
  del día y al leer se recorta a `date`. Prometer horas sería mentir.
- **`showCompleted=false` al listar**: pendientes significa pendientes. La
  API además esconde las completadas viejas detrás de `showHidden`, así que
  el filtro explícito evita depender de ese matiz.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID

import httpx
import structlog

from src.application.ports.tareas import Tareas
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.tarea import Tarea
from src.domain.exceptions import (
    CuentaNoConectadaError,
    EntityNotFoundError,
    ServiceUnavailableError,
)
from src.infrastructure.external.google.transporte import json_o_vacio, traducir_rechazo

logger = structlog.get_logger(__name__)

BASE = "https://tasks.googleapis.com/tasks/v1"
LISTA = "@default"
# La API pagina de a 20 por defecto; con esto una lista personal entra en un
# solo viaje. Paginar de verdad queda para cuando alguien tenga más.
MAXIMO_DE_TAREAS = 100


class TareasGoogle(Tareas):
    """El puerto `Tareas` hablando con la API real."""

    def __init__(self, cliente: httpx.AsyncClient, conectar: ConectarGoogle) -> None:
        self._cliente = cliente
        self._conectar = conectar

    async def pendientes(self, usuario_id: UUID) -> tuple[Tarea, ...]:
        token = await self._credencial(usuario_id)
        datos = await self._pedir(
            "GET",
            f"{BASE}/lists/{LISTA}/tasks",
            token,
            params={"showCompleted": "false", "maxResults": str(MAXIMO_DE_TAREAS)},
        )
        tareas = [_a_tarea(item) for item in datos.get("items", []) if isinstance(item, dict)]
        # La API ordena por posición manual; para conversar sirve más el
        # vencimiento: primero lo que vence antes, al final lo sin fecha.
        tareas.sort(key=lambda t: (t.vencimiento is None, t.vencimiento or date.max))
        logger.info("tasks.listadas", usuario_id=str(usuario_id), cantidad=len(tareas))
        return tuple(tareas)

    async def crear(self, usuario_id: UUID, tarea: Tarea) -> Tarea:
        token = await self._credencial(usuario_id)
        cuerpo: dict[str, Any] = {"title": tarea.titulo}
        if tarea.notas:
            cuerpo["notes"] = tarea.notas
        if tarea.vencimiento is not None:
            # Medianoche UTC del día: la única parte que la API conserva.
            cuerpo["due"] = f"{tarea.vencimiento.isoformat()}T00:00:00.000Z"

        datos = await self._pedir("POST", f"{BASE}/lists/{LISTA}/tasks", token, json=cuerpo)
        creada = _a_tarea(datos)
        logger.info("tasks.creada", usuario_id=str(usuario_id))
        return creada if creada.id is not None else tarea

    async def completar(self, usuario_id: UUID, tarea_id: str) -> None:
        token = await self._credencial(usuario_id)
        await self._pedir(
            "PATCH",
            f"{BASE}/lists/{LISTA}/tasks/{httpx.URL(tarea_id)}",
            token,
            json={"status": "completed"},
            # Completar algo que ya no existe no tiene estado final
            # equivalente: es un error con nombre, como en modificar eventos.
            error_404=EntityNotFoundError("Esa tarea ya no existe en la lista."),
        )
        logger.info("tasks.completada", usuario_id=str(usuario_id))

    # --- Plomería ---------------------------------------------------------

    async def _credencial(self, usuario_id: UUID) -> str:
        token = await self._conectar.credencial_vigente(usuario_id, datetime.now(UTC))
        if token is None:
            raise CuentaNoConectadaError
        return token.access_token

    async def _pedir(
        self,
        metodo: str,
        url: str,
        token: str,
        params: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        error_404: Exception | None = None,
    ) -> dict[str, Any]:
        try:
            respuesta = await self._cliente.request(
                metodo,
                url,
                params=params,
                json=json,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            logger.error("tasks.error_transporte", tipo=type(exc).__name__)
            raise ServiceUnavailableError("No se pudo contactar a Google Tasks.") from None

        if respuesta.status_code == httpx.codes.NOT_FOUND and error_404 is not None:
            raise error_404
        traducir_rechazo(respuesta, "tasks")
        return json_o_vacio(respuesta)


def _a_tarea(item: dict[str, Any]) -> Tarea:
    """Traduce el JSON de la API a la entidad, tolerando campos ausentes."""
    due = item.get("due")
    vencimiento: date | None = None
    if isinstance(due, str) and len(due) >= 10:
        try:
            vencimiento = date.fromisoformat(due[:10])
        except ValueError:
            vencimiento = None
    return Tarea(
        titulo=str(item.get("title") or "(sin título)"),
        vencimiento=vencimiento,
        notas=str(item["notes"]) if item.get("notes") else None,
        completada=item.get("status") == "completed",
        id=str(item["id"]) if item.get("id") else None,
    )


def create_tareas_google(cliente: httpx.AsyncClient, conectar: ConectarGoogle) -> TareasGoogle:
    """Factory del adaptador; el composition root la llama."""
    return TareasGoogle(cliente, conectar)
