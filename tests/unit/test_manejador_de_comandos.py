"""Tests de los comandos que no pasan por el modelo (RF-11, RF-12).

Reemplaza a `test_router_de_comandos.py`: en PB-009 la función pura pasó a ser
un servicio con dependencias, porque `/estado` y `/conectar` necesitan datos.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from src.application.services.manejador_de_comandos import (
    AYUDA,
    CONECTAR_ERROR,
    CONECTAR_NO_DISPONIBLE,
    DESCONECTAR_SIN_CUENTA,
    ESTADO_CON_GOOGLE,
    ESTADO_SIN_CUENTAS,
    ManejadorDeComandos,
)
from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.usuario import Usuario
from tests.dobles import AutorizadorFalso, RepositorioOAuthTokenEnMemoria

AHORA = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
ID_USUARIO = uuid4()
USUARIO = Usuario(telefono_whatsapp="+5491160007044", nombre="Ramiro", id=ID_USUARIO)


def _con_google() -> tuple[ManejadorDeComandos, ConectarGoogle]:
    caso = ConectarGoogle(
        RepositorioOAuthTokenEnMemoria(), AutorizadorFalso(usuario_fijo=USUARIO.id)
    )
    return ManejadorDeComandos(caso), caso


def _sin_google() -> ManejadorDeComandos:
    return ManejadorDeComandos(None)


# --- /ayuda ------------------------------------------------------------------


@pytest.mark.parametrize(
    "texto",
    ["/ayuda", "/AYUDA", "  /Ayuda  ", "/ayuda por favor"],
    ids=["exacto", "mayusculas", "con_espacios", "con_cola"],
)
async def test_el_comando_de_ayuda_se_reconoce(texto: str) -> None:
    manejador, _ = _con_google()

    assert await manejador.responder(texto, USUARIO, AHORA) == AYUDA


async def test_la_ayuda_funciona_sin_ninguna_dependencia() -> None:
    """RF-11: es lo único que tiene que contestar siempre, pase lo que pase."""
    assert await _sin_google().responder("/ayuda", USUARIO, AHORA) == AYUDA


async def test_la_ayuda_lista_los_comandos_que_existen() -> None:
    for comando in ("/ayuda", "/estado", "/conectar"):
        assert comando in AYUDA


# --- Lo que no es comando ----------------------------------------------------


@pytest.mark.parametrize(
    "texto",
    ["hola", "agendame una reunion", "", "   ", "ayuda", "/desconocido", "😀"],
    ids=["saludo", "natural", "vacio", "espacios", "sin_barra", "otro", "emoji"],
)
async def test_lo_que_no_es_comando_queda_para_el_agente(texto: str) -> None:
    """None no es "no entendí": es "esto no me toca a mí"."""
    manejador, _ = _con_google()

    assert await manejador.responder(texto, USUARIO, AHORA) is None


# --- /estado (RF-12) ---------------------------------------------------------


async def test_estado_sin_cuentas_conectadas() -> None:
    manejador, _ = _con_google()

    assert await manejador.responder("/estado", USUARIO, AHORA) == ESTADO_SIN_CUENTAS


async def test_estado_despues_de_conectar_google() -> None:
    """Dejó de ser un texto fijo: ahora consulta de verdad."""
    manejador, caso = _con_google()
    await caso.completar("4/codigo", "state", AHORA)

    assert await manejador.responder("/estado", USUARIO, AHORA) == ESTADO_CON_GOOGLE


async def test_estado_sin_oauth_configurado_no_rompe() -> None:
    assert await _sin_google().responder("/estado", USUARIO, AHORA) == ESTADO_SIN_CUENTAS


async def test_el_estado_orienta_hacia_conectar() -> None:
    assert "/conectar" in ESTADO_SIN_CUENTAS


async def test_el_estado_conectado_promete_solo_lo_que_hay() -> None:
    """El prompt le prohíbe al modelo inventar capacidades; los textos fijos igual.

    Desde PB-016: leer, crear y eliminar sí; modificar todavía no. Y la
    confirmación previa se anuncia, porque es la promesa central de RF-08.
    """
    assert "leer tu calendario" in ESTADO_CON_GOOGLE
    assert "crear" in ESTADO_CON_GOOGLE
    assert "modificar" in ESTADO_CON_GOOGLE
    assert "confirmación" in ESTADO_CON_GOOGLE


# --- /conectar ---------------------------------------------------------------


async def test_conectar_entrega_un_enlace() -> None:
    manejador, _ = _con_google()

    respuesta = await manejador.responder("/conectar", USUARIO, AHORA)

    assert respuesta is not None
    assert "https://accounts.google.com" in respuesta


async def test_el_mensaje_avisa_que_el_enlace_vence() -> None:
    """Es un enlace de autorización que queda en un chat para siempre."""
    manejador, _ = _con_google()

    respuesta = await manejador.responder("/conectar", USUARIO, AHORA)

    assert respuesta is not None
    assert "10 minutos" in respuesta


async def test_el_mensaje_aclara_que_solo_es_lectura() -> None:
    """La persona tiene que saber qué está por conceder."""
    manejador, _ = _con_google()

    respuesta = await manejador.responder("/conectar", USUARIO, AHORA)

    assert respuesta is not None
    assert "ver" in respuesta.lower()


async def test_conectar_sin_oauth_configurado_avisa() -> None:
    assert await _sin_google().responder("/conectar", USUARIO, AHORA) == CONECTAR_NO_DISPONIBLE


async def test_si_falla_armar_el_enlace_la_persona_igual_recibe_respuesta() -> None:
    """RF-19: nunca dejar el mensaje sin contestar."""

    class CasoRoto(ConectarGoogle):
        def link_de_autorizacion(self, usuario_id: UUID, ahora: datetime) -> str:
            raise RuntimeError("se rompió algo")

    manejador = ManejadorDeComandos(CasoRoto(RepositorioOAuthTokenEnMemoria(), AutorizadorFalso()))

    assert await manejador.responder("/conectar", USUARIO, AHORA) == CONECTAR_ERROR


async def test_un_usuario_sin_id_no_rompe() -> None:
    """Defensa en profundidad: el caso de uso ya lo descarta antes."""
    manejador, _ = _con_google()
    sin_id = Usuario(telefono_whatsapp="+5491160007044")

    assert await manejador.responder("/conectar", sin_id, AHORA) == CONECTAR_NO_DISPONIBLE


# --- /desconectar (PB-011, RF-12) --------------------------------------------
#
# Acción destructiva sin grafo: la confirmación son dos pasos y el estado vive
# en el texto del segundo comando, no en memoria.


async def _con_cuenta_conectada() -> tuple[ManejadorDeComandos, ConectarGoogle]:
    manejador, caso = _con_google()
    await caso.completar("4/codigo", "state", AHORA)
    return manejador, caso


async def test_desconectar_sin_confirmar_solo_explica() -> None:
    manejador, caso = await _con_cuenta_conectada()

    respuesta = await manejador.responder("/desconectar", USUARIO, AHORA)
    assert respuesta is not None

    assert "/desconectar confirmar" in respuesta
    assert await caso.esta_conectado(ID_USUARIO) is True  # NO borró nada


async def test_desconectar_confirmado_revoca_y_borra() -> None:
    manejador, caso = await _con_cuenta_conectada()

    respuesta = await manejador.responder("/desconectar confirmar", USUARIO, AHORA)
    assert respuesta is not None

    assert await caso.esta_conectado(ID_USUARIO) is False
    assert "revoqué" in respuesta
    assert "/conectar" in respuesta  # el camino de vuelta


async def test_desconectar_sin_cuenta_lo_dice() -> None:
    manejador, _ = _con_google()  # nunca se conectó

    respuesta = await manejador.responder("/desconectar", USUARIO, AHORA)
    assert respuesta is not None

    assert respuesta == DESCONECTAR_SIN_CUENTA


async def test_si_google_no_confirma_la_revocacion_igual_se_borra_y_se_avisa() -> None:
    """Best-effort honesto: el token se va de nuestro lado, y se dice la verdad."""
    autorizador = AutorizadorFalso(usuario_fijo=ID_USUARIO)
    autorizador.revocacion_exitosa = False
    caso = ConectarGoogle(RepositorioOAuthTokenEnMemoria(), autorizador)
    await caso.completar("4/codigo", "state", AHORA)
    manejador = ManejadorDeComandos(caso)

    respuesta = await manejador.responder("/desconectar confirmar", USUARIO, AHORA)
    assert respuesta is not None

    assert await caso.esta_conectado(ID_USUARIO) is False  # borrado igual
    assert "google.com/permissions" in respuesta  # cómo cerrar el círculo


async def test_un_sufijo_que_no_es_confirmar_no_ejecuta() -> None:
    """Sólo la palabra exacta: "/desconectar ya" no borra nada."""
    manejador, caso = await _con_cuenta_conectada()

    respuesta = await manejador.responder("/desconectar ya", USUARIO, AHORA)
    assert respuesta is not None

    assert await caso.esta_conectado(ID_USUARIO) is True
    assert "/desconectar confirmar" in respuesta


async def test_la_ayuda_lista_desconectar() -> None:
    assert "/desconectar" in AYUDA
