"""Tests del despachador de recordatorios (PB-030): qué sale, a quién, y qué pasa si falla.

El reloj es un parámetro (`ejecutar(ahora)`), así que acá se usa una fecha
fija: no pasa por ninguna validación de "hoy" (regla 5 de docs/CLAUDE.md §9).
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from src.application.use_cases.enviar_recordatorios_vencidos import EnviarRecordatoriosVencidos
from src.domain.entities.recordatorio import EstadoDeRecordatorio, Recordatorio
from src.domain.entities.usuario import Usuario
from src.domain.exceptions import (
    InvalidValueError,
    MensajeNoEnviadoError,
    MensajeRechazadoError,
    RepositoryError,
)
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp
from src.infrastructure.config.zona import ZONA_HORARIA
from src.interfaces.jobs.recordatorios import despachar_recordatorios
from tests.dobles import MensajeroFalso, RecordatoriosEnMemoria, RepositorioUsuarioEnMemoria

TELEFONO = "+5491123456789"
# 22:00 UTC = 19:00 en Argentina.
AHORA = datetime(2026, 9, 30, 22, 0, tzinfo=UTC)


class Escenario:
    """Un usuario real, sus recordatorios y el caso de uso armado con dobles."""

    def __init__(self, mensajero: MensajeroFalso | None = None) -> None:
        self.usuarios = RepositorioUsuarioEnMemoria()
        self.recordatorios = RecordatoriosEnMemoria()
        self.mensajero = mensajero or MensajeroFalso()
        self.caso = EnviarRecordatoriosVencidos(
            self.recordatorios, self.usuarios, self.mensajero, zona=ZONA_HORARIA
        )

    async def con_usuario(self) -> Escenario:
        usuario = await self.usuarios.crear(Usuario(telefono_whatsapp=TELEFONO))
        assert usuario.id is not None
        self.usuario_id = usuario.id
        return self

    async def programar(self, texto: str, momento: datetime) -> Recordatorio:
        return await self.recordatorios.crear(
            Recordatorio(usuario_id=self.usuario_id, texto=texto, momento=momento)
        )


class MensajeroQueFallaLaPrimeraVez(MensajeroFalso):
    """Un corte de red en el primer envío; los siguientes andan."""

    def __init__(self) -> None:
        super().__init__()
        self.intentos = 0

    async def enviar_texto(self, destino: NumeroWhatsApp, texto: str) -> None:
        self.intentos += 1
        if self.intentos == 1:
            raise MensajeNoEnviadoError("No se pudo contactar a WhatsApp.")
        await super().enviar_texto(destino, texto)


# --- Lo que sale -------------------------------------------------------------


async def test_manda_solo_los_vencidos_al_numero_de_su_dueno() -> None:
    e = await Escenario().con_usuario()
    vencido = await e.programar("sacar la pizza", AHORA - timedelta(seconds=10))
    futuro = await e.programar("llamar al banco", AHORA + timedelta(minutes=5))

    enviados = await e.caso.ejecutar(AHORA)

    assert enviados == 1
    [(destino, texto)] = e.mensajero.enviados
    assert destino.valor == TELEFONO
    assert texto == "⏰ Recordatorio: sacar la pizza"
    assert e.recordatorios.estado_de(vencido.id) is EstadoDeRecordatorio.ENVIADO
    assert e.recordatorios.estado_de(futuro.id) is EstadoDeRecordatorio.PENDIENTE


async def test_uno_ya_enviado_no_sale_dos_veces() -> None:
    e = await Escenario().con_usuario()
    await e.programar("sacar la pizza", AHORA - timedelta(seconds=10))

    await e.caso.ejecutar(AHORA)
    await e.caso.ejecutar(AHORA + timedelta(seconds=30))  # la vuelta siguiente

    assert len(e.mensajero.enviados) == 1


async def test_si_otro_despachador_lo_reclamo_primero_no_lo_manda() -> None:
    """Deploy con dos contenedores: el compare-and-set lo gana uno solo."""
    e = await Escenario().con_usuario()
    recordatorio = await e.programar("sacar la pizza", AHORA - timedelta(seconds=10))
    assert recordatorio.id is not None
    e.recordatorios.reclamos_perdidos.add(recordatorio.id)

    enviados = await e.caso.ejecutar(AHORA)

    assert enviados == 0
    assert e.mensajero.enviados == []


async def test_puntual_no_aclara_la_hora() -> None:
    e = await Escenario().con_usuario()
    await e.programar("sacar la pizza", AHORA - timedelta(seconds=20))

    await e.caso.ejecutar(AHORA)

    assert e.mensajero.textos == ["⏰ Recordatorio: sacar la pizza"]


async def test_con_demora_aclara_para_que_hora_era_en_hora_local() -> None:
    e = await Escenario().con_usuario()
    await e.programar("sacar la pizza", AHORA - timedelta(minutes=10))  # 18:50 en Argentina

    await e.caso.ejecutar(AHORA)

    [texto] = e.mensajero.textos
    assert texto.startswith("⏰ Recordatorio: sacar la pizza\n")
    assert "Era para las 18:50" in texto


# --- Cuando falla ------------------------------------------------------------


async def test_un_rechazo_de_whatsapp_lo_deja_fallido_y_no_se_reintenta() -> None:
    """131047 (fuera de la ventana de 24 h): reintentar daría lo mismo."""
    e = await Escenario(MensajeroFalso(fallar_con=MensajeRechazadoError("131047"))).con_usuario()
    recordatorio = await e.programar("sacar la pizza", AHORA - timedelta(seconds=10))

    await e.caso.ejecutar(AHORA)
    e.mensajero.fallar_con = None
    await e.caso.ejecutar(AHORA + timedelta(seconds=30))

    assert e.recordatorios.estado_de(recordatorio.id) is EstadoDeRecordatorio.FALLIDO
    assert e.mensajero.enviados == []


async def test_un_corte_transitorio_se_reintenta_en_la_vuelta_siguiente() -> None:
    e = await Escenario(MensajeroQueFallaLaPrimeraVez()).con_usuario()
    recordatorio = await e.programar("sacar la pizza", AHORA - timedelta(seconds=10))

    await e.caso.ejecutar(AHORA)
    assert e.recordatorios.estado_de(recordatorio.id) is EstadoDeRecordatorio.PENDIENTE

    await e.caso.ejecutar(AHORA + timedelta(seconds=30))
    assert e.recordatorios.estado_de(recordatorio.id) is EstadoDeRecordatorio.ENVIADO
    assert e.mensajero.textos == ["⏰ Recordatorio: sacar la pizza"]


async def test_con_mas_de_una_hora_de_demora_un_corte_lo_da_por_perdido() -> None:
    falla = MensajeNoEnviadoError("No se pudo contactar a WhatsApp.")
    e = await Escenario(MensajeroFalso(fallar_con=falla)).con_usuario()
    recordatorio = await e.programar("sacar la pizza", AHORA - timedelta(hours=2))

    await e.caso.ejecutar(AHORA)

    assert e.recordatorios.estado_de(recordatorio.id) is EstadoDeRecordatorio.FALLIDO


async def test_un_recordatorio_que_falla_no_frena_a_los_demas() -> None:
    e = await Escenario(MensajeroQueFallaLaPrimeraVez()).con_usuario()
    primero = await e.programar("sacar la pizza", AHORA - timedelta(minutes=1))
    segundo = await e.programar("llamar al banco", AHORA - timedelta(seconds=10))

    enviados = await e.caso.ejecutar(AHORA)

    assert enviados == 1
    assert e.recordatorios.estado_de(primero.id) is EstadoDeRecordatorio.PENDIENTE
    assert e.recordatorios.estado_de(segundo.id) is EstadoDeRecordatorio.ENVIADO


async def test_si_el_usuario_ya_no_existe_queda_fallido() -> None:
    e = await Escenario().con_usuario()
    huerfano = await e.recordatorios.crear(
        Recordatorio(usuario_id=uuid4(), texto="x", momento=AHORA - timedelta(seconds=10))
    )

    await e.caso.ejecutar(AHORA)

    assert e.recordatorios.estado_de(huerfano.id) is EstadoDeRecordatorio.FALLIDO
    assert e.mensajero.enviados == []


# --- El bucle ----------------------------------------------------------------


async def test_una_vuelta_que_falla_no_mata_el_bucle() -> None:
    """Si una base que parpadea matara el bucle, nadie lo levantaría hasta el próximo deploy."""
    e = await Escenario().con_usuario()
    await e.programar("sacar la pizza", datetime.now(UTC) - timedelta(seconds=5))
    e.recordatorios.fallar_con = RepositoryError("la base parpadeó")

    bucle = asyncio.create_task(despachar_recordatorios(e.caso, intervalo=0.01))
    await asyncio.sleep(0.05)  # varias vueltas fallidas
    e.recordatorios.fallar_con = None
    for _ in range(200):
        if e.mensajero.enviados:
            break
        await asyncio.sleep(0.01)
    bucle.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await bucle

    assert e.mensajero.textos == ["⏰ Recordatorio: sacar la pizza"]


# --- La entidad --------------------------------------------------------------


def test_sin_texto_no_es_un_recordatorio() -> None:
    with pytest.raises(InvalidValueError):
        Recordatorio(usuario_id=uuid4(), texto="   ", momento=AHORA)


def test_un_momento_sin_zona_horaria_es_ambiguo() -> None:
    with pytest.raises(InvalidValueError):
        Recordatorio(usuario_id=uuid4(), texto="x", momento=datetime(2026, 10, 1, 9, 0))


def test_un_texto_demasiado_largo_no_entra() -> None:
    with pytest.raises(InvalidValueError):
        Recordatorio(usuario_id=uuid4(), texto="x" * 301, momento=AHORA)


def test_el_texto_no_aparece_en_el_repr() -> None:
    """Es contenido de la persona: no se cuela en un log por accidente (RF-18)."""
    recordatorio = Recordatorio(usuario_id=uuid4(), texto="tomar la pastilla", momento=AHORA)

    assert "pastilla" not in repr(recordatorio)
