"""Tests de integración del webhook de WhatsApp (PB-004).

Regla de estos tests: **nunca afirmar sólo sobre el status code**. El webhook
devuelve 200 para casi todo a propósito, así que un test que sólo mira el 200
pasaría aunque no se procesara nada. Las aserciones van sobre los dobles.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.application.services.router_de_comandos import AYUDA
from src.domain.exceptions import RepositoryError
from src.infrastructure.external.whatsapp.firma import HEADER_FIRMA, firmar
from tests.conftest import APP_SECRET, VERIFY_TOKEN
from tests.dobles import AgenteFalso, MensajeroFalso, RepositorioUsuarioEnMemoria
from tests.payloads_meta import (
    PARAM_CHALLENGE,
    PARAM_MODO,
    PARAM_VERIFY_TOKEN,
    TELEFONO_E164,
    WA_ID,
    acuse_de_entrega,
    mensaje_no_texto,
    webhook_fresco,
)

RUTA = "/webhooks/whatsapp"


def _postear(cliente: TestClient, cuerpo: bytes, *, secreto: str = APP_SECRET) -> object:
    """Postea firmando exactamente los bytes que se mandan."""
    return cliente.post(
        RUTA,
        content=cuerpo,
        headers={HEADER_FIRMA: firmar(cuerpo, secreto), "content-type": "application/json"},
    )


# --- Handshake de verificación ----------------------------------------------


def test_el_challenge_vuelve_como_texto_plano(client_con_whatsapp: TestClient) -> None:
    """Si saliera como JSON iría entre comillas y Meta rechazaría el webhook."""
    respuesta = client_con_whatsapp.get(
        RUTA,
        params={
            PARAM_MODO: "subscribe",
            PARAM_VERIFY_TOKEN: VERIFY_TOKEN,
            PARAM_CHALLENGE: "1158201444",
        },
    )

    assert respuesta.status_code == 200
    assert respuesta.text == "1158201444"  # exactamente igual, sin comillas
    assert respuesta.headers["content-type"].startswith("text/plain")


@pytest.mark.parametrize(
    "params",
    [
        {PARAM_MODO: "subscribe", PARAM_VERIFY_TOKEN: "incorrecto", PARAM_CHALLENGE: "1"},
        {PARAM_MODO: "unsubscribe", PARAM_VERIFY_TOKEN: VERIFY_TOKEN, PARAM_CHALLENGE: "1"},
        {PARAM_MODO: "subscribe", PARAM_CHALLENGE: "1"},
        {},
        {PARAM_MODO: "subscribe", PARAM_VERIFY_TOKEN: VERIFY_TOKEN, PARAM_CHALLENGE: "no-numero"},
    ],
    ids=["token_malo", "modo_malo", "sin_token", "sin_nada", "challenge_no_numerico"],
)
def test_una_verificacion_incorrecta_da_403(
    client_con_whatsapp: TestClient, params: dict[str, str]
) -> None:
    assert client_con_whatsapp.get(RUTA, params=params).status_code == 403


def test_sin_verify_token_configurado_da_503(client: TestClient) -> None:
    """No es 403: no es que el token esté mal, es que falta configurarlo."""
    respuesta = client.get(
        RUTA, params={PARAM_MODO: "subscribe", PARAM_VERIFY_TOKEN: "x", PARAM_CHALLENGE: "1"}
    )

    assert respuesta.status_code == 503


# --- Firma ------------------------------------------------------------------


def test_una_firma_valida_procesa_el_mensaje(
    client_con_whatsapp: TestClient, mensajero_falso: MensajeroFalso
) -> None:
    _postear(client_con_whatsapp, webhook_fresco(texto="/ayuda"))

    assert mensajero_falso.textos == [AYUDA]


def test_una_firma_invalida_da_403_y_no_procesa(
    client_con_whatsapp: TestClient, mensajero_falso: MensajeroFalso
) -> None:
    respuesta = _postear(client_con_whatsapp, webhook_fresco(), secreto="secreto-equivocado")

    assert respuesta.status_code == 403  # type: ignore[attr-defined]
    assert mensajero_falso.enviados == []


def test_sin_cabecera_de_firma_da_403(
    client_con_whatsapp: TestClient, mensajero_falso: MensajeroFalso
) -> None:
    respuesta = client_con_whatsapp.post(
        RUTA, content=webhook_fresco(), headers={"content-type": "application/json"}
    )

    assert respuesta.status_code == 403
    assert mensajero_falso.enviados == []


def test_un_cuerpo_alterado_despues_de_firmar_da_403(
    client_con_whatsapp: TestClient, mensajero_falso: MensajeroFalso
) -> None:
    cuerpo = webhook_fresco()
    respuesta = client_con_whatsapp.post(
        RUTA,
        content=cuerpo + b" ",
        headers={HEADER_FIRMA: firmar(cuerpo, APP_SECRET), "content-type": "application/json"},
    )

    assert respuesta.status_code == 403
    assert mensajero_falso.enviados == []


# --- Procesamiento ----------------------------------------------------------


def test_da_de_alta_al_usuario_que_escribe(
    client_con_whatsapp: TestClient, repositorio_usuarios: RepositorioUsuarioEnMemoria
) -> None:
    _postear(client_con_whatsapp, webhook_fresco())

    # El BackgroundTask ya corrió: TestClient los ejecuta antes de devolver.
    assert repositorio_usuarios._por_telefono.get(TELEFONO_E164) is not None


def test_el_lenguaje_natural_llega_al_agente(
    client_con_whatsapp: TestClient,
    mensajero_falso: MensajeroFalso,
    agente_falso: AgenteFalso,
) -> None:
    """El circuito completo de PB-005: webhook → caso de uso → agente → envío."""
    _postear(client_con_whatsapp, webhook_fresco(texto="agendame algo el jueves"))

    assert agente_falso.textos == ["agendame algo el jueves"]
    assert mensajero_falso.textos == [agente_falso.respuesta]


def test_un_acuse_de_entrega_no_genera_respuesta(
    client_con_whatsapp: TestClient, mensajero_falso: MensajeroFalso
) -> None:
    respuesta = _postear(client_con_whatsapp, acuse_de_entrega())

    assert respuesta.status_code == 200  # type: ignore[attr-defined]
    assert mensajero_falso.enviados == []


def test_un_mensaje_no_texto_no_genera_respuesta(
    client_con_whatsapp: TestClient, mensajero_falso: MensajeroFalso
) -> None:
    _postear(client_con_whatsapp, mensaje_no_texto("image"))

    assert mensajero_falso.enviados == []


def test_un_reintento_de_meta_no_duplica_la_respuesta(
    client_con_whatsapp: TestClient, mensajero_falso: MensajeroFalso
) -> None:
    """Es el caso que motiva el deduplicador."""
    cuerpo = webhook_fresco()

    _postear(client_con_whatsapp, cuerpo)
    _postear(client_con_whatsapp, cuerpo)

    assert len(mensajero_falso.enviados) == 1


# --- Robustez: el POST nunca devuelve 5xx -----------------------------------


def test_un_json_invalido_devuelve_200(
    client_con_whatsapp: TestClient, mensajero_falso: MensajeroFalso
) -> None:
    """Un 4xx/5xx haría que Meta reintente durante horas por un payload roto."""
    respuesta = _postear(client_con_whatsapp, b"esto no es json")

    assert respuesta.status_code == 200  # type: ignore[attr-defined]
    assert mensajero_falso.enviados == []


def test_un_payload_vacio_devuelve_200(client_con_whatsapp: TestClient) -> None:
    assert _postear(client_con_whatsapp, b"{}").status_code == 200  # type: ignore[attr-defined]


def test_si_la_base_falla_igual_se_devuelve_200(
    client_con_whatsapp: TestClient,
    repositorio_usuarios: RepositorioUsuarioEnMemoria,
    mensajero_falso: MensajeroFalso,
) -> None:
    """La excepción del BackgroundTask no puede escapar ni volverse un 500."""
    repositorio_usuarios.fallar_con = RepositoryError()

    respuesta = _postear(client_con_whatsapp, webhook_fresco())

    assert respuesta.status_code == 200  # type: ignore[attr-defined]
    # Aunque falló el procesamiento, se le avisó al usuario con el mensaje
    # pensado para él, no con un detalle técnico (RF-19).
    assert mensajero_falso.textos == [RepositoryError.mensaje_usuario]


def test_si_falla_el_procesamiento_el_mensaje_se_puede_reintentar(
    client_con_whatsapp: TestClient,
    repositorio_usuarios: RepositorioUsuarioEnMemoria,
    mensajero_falso: MensajeroFalso,
) -> None:
    """Un fallo transitorio no debe dejar el mensaje sin respuesta para siempre."""
    cuerpo = webhook_fresco()
    repositorio_usuarios.fallar_con = RepositoryError()
    _postear(client_con_whatsapp, cuerpo)

    repositorio_usuarios.fallar_con = None
    mensajero_falso.enviados.clear()
    _postear(client_con_whatsapp, cuerpo)

    assert len(mensajero_falso.textos) == 1


def test_en_modo_degradado_devuelve_200_y_no_explota(client: TestClient) -> None:
    """Sin Supabase ni WhatsApp, el webhook responde igual.

    Un 503 acá le diría a Meta que reintente durante horas contra un entorno
    que sabemos que no puede atenderlo.
    """
    cuerpo = webhook_fresco()
    respuesta = client.post(RUTA, content=cuerpo, headers={"content-type": "application/json"})

    assert respuesta.status_code == 200


# --- RF-18 ------------------------------------------------------------------


def test_no_se_filtra_el_telefono_ni_el_texto_en_los_logs(
    client_con_whatsapp: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """Se usa capsys y no caplog: configure_logging limpia los handlers."""
    _postear(client_con_whatsapp, webhook_fresco(texto="mi informe medico"))

    salida = capsys.readouterr().out
    assert "mi informe medico" not in salida
    assert WA_ID not in salida
    assert TELEFONO_E164 not in salida


def test_no_se_filtra_el_verify_token_en_los_logs(
    client_con_whatsapp: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """uvicorn.access loguearía el query string entero si no lo silenciáramos."""
    client_con_whatsapp.get(
        RUTA,
        params={
            PARAM_MODO: "subscribe",
            PARAM_VERIFY_TOKEN: VERIFY_TOKEN,
            PARAM_CHALLENGE: "1158201444",
        },
    )

    assert VERIFY_TOKEN not in capsys.readouterr().out


def test_la_respuesta_de_error_no_expone_detalles(client_con_whatsapp: TestClient) -> None:
    """Un 403 no debe contarle nada a quien forjó la petición."""
    respuesta = _postear(client_con_whatsapp, webhook_fresco(), secreto="mal")

    assert respuesta.text == ""  # type: ignore[attr-defined]
