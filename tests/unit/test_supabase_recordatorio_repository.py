"""Tests del repositorio Supabase de recordatorios (PB-030).

Las aserciones van sobre lo que realmente sale hacia la base: que el texto
viaje cifrado y que el cambio de estado sea un compare-and-set de verdad.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from cryptography.fernet import Fernet
from supabase import AsyncClient

from src.domain.entities.recordatorio import EstadoDeRecordatorio, Recordatorio
from src.infrastructure.persistence.encryption import TokenCipher
from src.infrastructure.persistence.supabase_recordatorio_repository import (
    SupabaseRecordatorioRepository,
)
from tests.dobles import FakeSupabaseClient

CIFRADOR = TokenCipher(Fernet.generate_key().decode())
USUARIO = uuid4()
MOMENTO = datetime(2026, 10, 1, 0, 55, tzinfo=UTC)
MIGRACION = Path(__file__).parents[2] / "db" / "migrations" / "004_recordatorios.sql"


def _fila(texto: str = "sacar la pizza", estado: str = "pendiente") -> dict[str, Any]:
    """Lo que devolvería PostgREST: el texto, cifrado."""
    return {
        "id": str(uuid4()),
        "usuario_id": str(USUARIO),
        "texto_cifrado": CIFRADOR.cifrar(texto),
        "momento": MOMENTO.isoformat(),
        "estado": estado,
        "enviado_en": None,
    }


def _repo(cliente: FakeSupabaseClient) -> SupabaseRecordatorioRepository:
    return SupabaseRecordatorioRepository(cast("AsyncClient", cliente), CIFRADOR)


async def test_crear_manda_el_texto_cifrado_y_nunca_en_claro() -> None:
    cliente = FakeSupabaseClient(respuestas={"recordatorios": [_fila()]})

    creado = await _repo(cliente).crear(
        Recordatorio(usuario_id=USUARIO, texto="sacar la pizza", momento=MOMENTO)
    )

    llamada = cliente.ultima_llamada()
    assert llamada.operacion == "insert"
    assert llamada.payload["texto_cifrado"].startswith("gA")
    assert "pizza" not in str(llamada.payload)
    assert CIFRADOR.descifrar(llamada.payload["texto_cifrado"]) == "sacar la pizza"
    assert llamada.payload["momento"] == MOMENTO.isoformat()
    assert llamada.payload["estado"] == "pendiente"
    assert creado.texto == "sacar la pizza"  # vuelve descifrado
    assert creado.id is not None


async def test_vencidos_pide_los_pendientes_hasta_ahora() -> None:
    cliente = FakeSupabaseClient(respuestas={"recordatorios": [_fila()]})
    hasta = datetime(2026, 10, 1, 1, 0, tzinfo=UTC)

    vencidos = await _repo(cliente).vencidos(hasta, limite=20)

    llamada = cliente.ultima_llamada()
    assert llamada.operacion == "select"
    assert llamada.filtros == {"estado": "pendiente", "momento<=": hasta.isoformat()}
    assert [r.texto for r in vencidos] == ["sacar la pizza"]
    assert vencidos[0].momento == MOMENTO


async def test_pendientes_de_filtra_por_persona_y_estado() -> None:
    cliente = FakeSupabaseClient(respuestas={"recordatorios": [_fila()]})

    await _repo(cliente).pendientes_de(USUARIO)

    assert cliente.ultima_llamada().filtros == {"usuario_id": str(USUARIO), "estado": "pendiente"}


async def test_cambiar_estado_es_un_compare_and_set() -> None:
    cliente = FakeSupabaseClient(respuestas={"recordatorios": [_fila(estado="enviando")]})
    rid = uuid4()

    gano = await _repo(cliente).cambiar_estado(
        rid, EstadoDeRecordatorio.PENDIENTE, EstadoDeRecordatorio.ENVIANDO
    )

    llamada = cliente.ultima_llamada()
    assert llamada.operacion == "update"
    assert llamada.payload == {"estado": "enviando"}
    # La condición viaja en el mismo UPDATE: es lo que lo vuelve atómico.
    assert llamada.filtros == {"id": str(rid), "estado": "pendiente"}
    assert gano is True


async def test_si_no_cambio_ninguna_fila_perdio_el_reclamo() -> None:
    cliente = FakeSupabaseClient(respuestas={"recordatorios": []})

    gano = await _repo(cliente).cambiar_estado(
        uuid4(), EstadoDeRecordatorio.PENDIENTE, EstadoDeRecordatorio.ENVIANDO
    )

    assert gano is False


async def test_al_marcarlo_enviado_queda_asentado_cuando() -> None:
    cliente = FakeSupabaseClient(respuestas={"recordatorios": [_fila(estado="enviado")]})

    await _repo(cliente).cambiar_estado(
        uuid4(), EstadoDeRecordatorio.ENVIANDO, EstadoDeRecordatorio.ENVIADO
    )

    payload = cliente.ultima_llamada().payload
    assert payload["estado"] == "enviado"
    assert datetime.fromisoformat(payload["enviado_en"]).tzinfo is not None


def test_la_migracion_cierra_la_tabla_y_exige_cifrado() -> None:
    sql = MIGRACION.read_text(encoding="utf-8")

    assert "alter table public.recordatorios enable row level security;" in sql
    assert "revoke all on public.recordatorios from anon, authenticated;" in sql
    assert "check (texto_cifrado ~ '^gA" in sql


def test_los_estados_de_la_migracion_son_los_del_dominio() -> None:
    sql = MIGRACION.read_text(encoding="utf-8")

    for estado in EstadoDeRecordatorio:
        assert f"'{estado.value}'" in sql
