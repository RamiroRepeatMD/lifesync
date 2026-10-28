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
    AYUDA_CON_NOTION,
    CONECTAR_ERROR,
    CONECTAR_NO_DISPONIBLE,
    CONECTAR_NOTION_NO_DISPONIBLE,
    DESCONECTAR_GOOGLE_SIN_CUENTA,
    DESCONECTAR_NOTION_LISTO,
    DESCONECTAR_NOTION_PIDE_CONFIRMACION,
    DESCONECTAR_NOTION_SIN_CUENTA,
    DESCONECTAR_NOTION_SIN_REVOCAR,
    DESCONECTAR_PIDE_CONFIRMACION,
    DESCONECTAR_SIN_CUENTA,
    ESTADO_CON_GOOGLE,
    ESTADO_SIN_CUENTAS,
    GOOGLE_SIN_CONECTAR,
    NOTION_CONECTADO,
    NOTION_SIN_CONECTAR,
    PISTA_DE_NOTION,
    ManejadorDeComandos,
)
from src.application.use_cases.conectar_google import ConectarGoogle
from src.application.use_cases.conectar_notion import ConectarNotion
from src.domain.entities.usuario import Usuario
from tests.dobles import AutorizadorFalso, AutorizadorNotionFalso, RepositorioOAuthTokenEnMemoria

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
    assert "archivos en tu Drive" in ESTADO_CON_GOOGLE
    assert "mandarlos a la papelera" in ESTADO_CON_GOOGLE  # PB-039: nunca para siempre
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


async def test_el_mensaje_dice_que_se_pide_y_que_nada_sale_sin_confirmar() -> None:
    """La persona tiene que saber qué está por conceder.

    Desde PB-038 Google le muestra "ver, editar y borrar todos tus archivos de
    Drive": el mensaje nombra cada cosa y la promesa de RF-08.
    """
    manejador, _ = _con_google()

    respuesta = await manejador.responder("/conectar", USUARIO, AHORA)

    assert respuesta is not None
    for pedido in ("calendario", "tareas", "correo", "Drive"):
        assert pedido in respuesta
    assert "sin confirmarte antes" in respuesta


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


# --- Notion (PB-014) -----------------------------------------------------------


async def _con_notion(
    *, google: bool = False, notion: bool = False
) -> tuple[ManejadorDeComandos, ConectarNotion, AutorizadorNotionFalso]:
    """Google y Notion configurados; `google` y `notion` dicen cuáles están conectados."""
    caso_google = ConectarGoogle(
        RepositorioOAuthTokenEnMemoria(), AutorizadorFalso(usuario_fijo=ID_USUARIO)
    )
    autorizador = AutorizadorNotionFalso(usuario_fijo=ID_USUARIO)
    caso_notion = ConectarNotion(RepositorioOAuthTokenEnMemoria(), autorizador)
    if google:
        await caso_google.completar("4/codigo", "state", AHORA)
    if notion:
        await caso_notion.completar("codigo", "state", AHORA)
    return ManejadorDeComandos(caso_google, caso_notion), caso_notion, autorizador


async def test_con_notion_la_ayuda_suma_sus_comandos() -> None:
    manejador, _, _ = await _con_notion()

    respuesta = await manejador.responder("/ayuda", USUARIO, AHORA)

    assert respuesta == AYUDA_CON_NOTION
    assert "/conectar notion" in AYUDA_CON_NOTION
    assert "/desconectar notion" in AYUDA_CON_NOTION


async def test_conectar_notion_manda_el_enlace_y_explica_que_elegir() -> None:
    manejador, _, _ = await _con_notion()

    respuesta = await manejador.responder("/conectar Notion", USUARIO, AHORA)

    assert respuesta is not None
    assert f"https://api.notion.com/v1/oauth/authorize?state=firmado-{ID_USUARIO}" in respuesta
    assert "qué páginas compartir conmigo" in respuesta
    assert "Nunca escribo, muevo ni borro nada sin confirmarte antes." in respuesta


async def test_conectar_notion_sin_configurar_lo_avisa() -> None:
    manejador, _ = _con_google()

    respuesta = await manejador.responder("/conectar notion", USUARIO, AHORA)

    assert respuesta == CONECTAR_NOTION_NO_DISPONIBLE


async def test_conectar_a_secas_sigue_siendo_google_y_suma_la_pista_de_notion() -> None:
    manejador, _, _ = await _con_notion()

    respuesta = await manejador.responder("/conectar", USUARIO, AHORA)

    assert respuesta is not None
    assert "https://accounts.google.com" in respuesta
    assert respuesta.endswith(PISTA_DE_NOTION)


async def test_sin_notion_configurado_conectar_no_suma_la_pista() -> None:
    manejador, _ = _con_google()

    respuesta = await manejador.responder("/conectar", USUARIO, AHORA)

    assert respuesta is not None
    assert PISTA_DE_NOTION not in respuesta


@pytest.mark.parametrize(
    ("google", "notion", "esperado"),
    [
        (False, False, f"{ESTADO_SIN_CUENTAS}\n\n{PISTA_DE_NOTION}"),
        (True, False, f"{ESTADO_CON_GOOGLE}\n\n{NOTION_SIN_CONECTAR}"),
        (False, True, f"{GOOGLE_SIN_CONECTAR}\n\n{NOTION_CONECTADO}"),
        (True, True, f"{ESTADO_CON_GOOGLE}\n\n{NOTION_CONECTADO}"),
    ],
    ids=["ninguna", "solo_google", "solo_notion", "las_dos"],
)
async def test_el_estado_dice_cada_conexion_por_separado(
    google: bool, notion: bool, esperado: str
) -> None:
    """Con Notion conectado, "ninguna cuenta" sería falso."""
    manejador, _, _ = await _con_notion(google=google, notion=notion)

    assert await manejador.responder("/estado", USUARIO, AHORA) == esperado


async def test_desconectar_notion_sin_conexion() -> None:
    manejador, _, _ = await _con_notion()

    respuesta = await manejador.responder("/desconectar notion", USUARIO, AHORA)

    assert respuesta == DESCONECTAR_NOTION_SIN_CUENTA


async def test_desconectar_notion_primero_pide_confirmacion() -> None:
    manejador, caso, autorizador = await _con_notion(notion=True)

    respuesta = await manejador.responder("/desconectar notion", USUARIO, AHORA)

    assert respuesta == DESCONECTAR_NOTION_PIDE_CONFIRMACION
    assert await caso.esta_conectado(ID_USUARIO) is True
    assert autorizador.revocados == []


async def test_desconectar_notion_confirmado_revoca_y_borra() -> None:
    manejador, caso, autorizador = await _con_notion(notion=True)

    respuesta = await manejador.responder("/desconectar notion confirmar", USUARIO, AHORA)

    assert respuesta == DESCONECTAR_NOTION_LISTO
    assert await caso.esta_conectado(ID_USUARIO) is False
    assert autorizador.revocados == ["notion-access-1", "notion-refresh-1"]


async def test_si_notion_no_confirma_la_revocacion_se_dice_como_cerrarla() -> None:
    manejador, caso, autorizador = await _con_notion(notion=True)
    autorizador.revocacion_exitosa = False

    respuesta = await manejador.responder("/desconectar notion confirmar", USUARIO, AHORA)

    assert respuesta == DESCONECTAR_NOTION_SIN_REVOCAR
    assert "Notion" in respuesta and "google" not in respuesta.lower()
    assert await caso.esta_conectado(ID_USUARIO) is False


async def test_una_confirmacion_con_algo_de_mas_no_confirma() -> None:
    manejador, caso, _ = await _con_notion(notion=True)

    respuesta = await manejador.responder("/desconectar notion confirmar ya", USUARIO, AHORA)

    assert respuesta == DESCONECTAR_NOTION_PIDE_CONFIRMACION
    assert await caso.esta_conectado(ID_USUARIO) is True


async def test_desconectar_ya_sigue_siendo_google() -> None:
    """Sólo "notion" elige Notion: lo demás se comporta como antes de PB-014."""
    manejador, caso, _ = await _con_notion(google=True, notion=True)

    respuesta = await manejador.responder("/desconectar ya", USUARIO, AHORA)

    assert respuesta == DESCONECTAR_PIDE_CONFIRMACION
    assert await caso.esta_conectado(ID_USUARIO) is True


async def test_desconectar_google_con_solo_notion_dice_cual_falta() -> None:
    manejador, caso, _ = await _con_notion(notion=True)

    for texto in ("/desconectar", "/desconectar confirmar"):
        assert await manejador.responder(texto, USUARIO, AHORA) == DESCONECTAR_GOOGLE_SIN_CUENTA
    assert await caso.esta_conectado(ID_USUARIO) is True


async def test_sin_notion_configurado_desconectar_notion_no_rompe() -> None:
    manejador, _ = _con_google()

    respuesta = await manejador.responder("/desconectar notion", USUARIO, AHORA)

    assert respuesta == DESCONECTAR_NOTION_SIN_CUENTA
