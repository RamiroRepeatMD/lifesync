"""Tests del caso de uso que procesa un mensaje entrante (PB-004)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
import structlog

from src.application.dto.mensaje_entrante import MensajeEntrante
from src.application.services.router_de_comandos import AYUDA, NO_ENTIENDO
from src.application.use_cases.procesar_mensaje_entrante import ProcesarMensajeEntrante
from src.domain.exceptions import RepositoryError
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp
from tests.dobles import MensajeroFalso, RepositorioUsuarioEnMemoria
from tests.payloads_meta import TELEFONO_E164, WA_ID

REMITENTE = NumeroWhatsApp.desde_wa_id(WA_ID)


def _mensaje(texto: str = "hola", wamid: str = "wamid.A") -> MensajeEntrante:
    return MensajeEntrante(
        wamid=wamid,
        remitente=REMITENTE,
        texto=texto,
        enviado_en=datetime.now(UTC),
        nombre_perfil="Ramiro",
    )


def _caso(
    usuarios: RepositorioUsuarioEnMemoria | None = None,
    mensajero: MensajeroFalso | None = None,
) -> tuple[ProcesarMensajeEntrante, RepositorioUsuarioEnMemoria, MensajeroFalso]:
    repo = usuarios or RepositorioUsuarioEnMemoria()
    men = mensajero or MensajeroFalso()
    return ProcesarMensajeEntrante(repo, men), repo, men


async def test_da_de_alta_al_usuario_que_escribe_por_primera_vez() -> None:
    caso, repo, _ = _caso()

    await caso.ejecutar(_mensaje())

    usuario = await repo.obtener_por_telefono(TELEFONO_E164)
    assert usuario is not None
    assert usuario.nombre == "Ramiro"


async def test_no_duplica_al_usuario_que_ya_existe() -> None:
    caso, repo, _ = _caso()

    await caso.ejecutar(_mensaje(wamid="wamid.A"))
    primero = await repo.obtener_por_telefono(TELEFONO_E164)
    await caso.ejecutar(_mensaje(wamid="wamid.B"))
    segundo = await repo.obtener_por_telefono(TELEFONO_E164)

    assert primero is not None
    assert segundo is not None
    assert primero.id == segundo.id


async def test_contesta_al_remitente() -> None:
    caso, _, mensajero = _caso()

    await caso.ejecutar(_mensaje())

    assert len(mensajero.enviados) == 1
    destino, _ = mensajero.enviados[0]
    assert destino == REMITENTE


async def test_responde_la_ayuda_al_comando_de_ayuda() -> None:
    caso, _, mensajero = _caso()

    await caso.ejecutar(_mensaje(texto="/ayuda"))

    assert mensajero.textos == [AYUDA]


async def test_responde_el_fallback_a_lenguaje_natural() -> None:
    caso, _, mensajero = _caso()

    await caso.ejecutar(_mensaje(texto="agendame una reunion"))

    assert mensajero.textos == [NO_ENTIENDO]


async def test_si_falla_la_base_no_se_envia_nada() -> None:
    caso, _, mensajero = _caso(usuarios=RepositorioUsuarioEnMemoria(fallar_con=RepositoryError()))

    with pytest.raises(RepositoryError):
        await caso.ejecutar(_mensaje())

    assert mensajero.enviados == []


async def test_el_error_de_envio_se_propaga() -> None:
    """Quien decide qué hacer con el fallo es el borde del webhook."""
    caso, _, _ = _caso(mensajero=MensajeroFalso(fallar_con=RepositoryError()))

    with pytest.raises(RepositoryError):
        await caso.ejecutar(_mensaje())


async def test_avisar_manda_el_texto_sin_tocar_la_base() -> None:
    caso, repo, mensajero = _caso()

    await caso.avisar(REMITENTE, "algo salió mal")

    assert mensajero.textos == ["algo salió mal"]
    assert await repo.obtener_por_telefono(TELEFONO_E164) is None


async def test_no_se_loguea_ni_el_telefono_ni_el_texto() -> None:
    """RF-18: el id del usuario alcanza para correlacionar."""
    caso, _, _ = _caso()

    with structlog.testing.capture_logs() as eventos:
        await caso.ejecutar(_mensaje(texto="mi diagnostico medico"))

    registrado = json.dumps(eventos, default=str)
    assert "mi diagnostico medico" not in registrado
    assert WA_ID not in registrado
    assert TELEFONO_E164 not in registrado
