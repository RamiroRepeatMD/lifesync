"""Tests del parser de payloads de webhook (PB-004).

El parser es la pieza que absorbe la forma real de la API de Meta. Estos tests
cubren los casos que sólo aparecen en producción: lotes, acuses mezclados con
mensajes, tipos no soportados y payloads deformes.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from src.infrastructure.external.whatsapp.parser import (
    FRESCURA_MAXIMA,
    LARGO_MAXIMO_TEXTO,
    MAXIMO_MENSAJES,
    parsear,
)
from tests.payloads_meta import (
    MOMENTO_FIJO,
    PHONE_NUMBER_ID,
    TELEFONO_E164,
    WA_ID,
    WAMID,
    acuse_de_entrega,
    mensaje_no_texto,
    webhook_con_dos_entradas,
    webhook_con_varios_mensajes,
    webhook_de_texto,
)


def _parsear(crudo: bytes, **kwargs: Any) -> Any:
    """Parsea unos bytes con el momento fijo de los payloads de prueba."""
    kwargs.setdefault("phone_number_id_propio", PHONE_NUMBER_ID)
    kwargs.setdefault("ahora", MOMENTO_FIJO)
    return parsear(json.loads(crudo), **kwargs)


# --- Camino feliz -----------------------------------------------------------


def test_extrae_un_mensaje_de_texto() -> None:
    contenido = _parsear(webhook_de_texto(texto="hola"))

    assert len(contenido.mensajes) == 1
    mensaje = contenido.mensajes[0]
    assert mensaje.texto == "hola"
    assert mensaje.wamid == WAMID
    assert mensaje.remitente.valor == TELEFONO_E164
    assert mensaje.nombre_perfil == "Ramiro Gracia"
    assert mensaje.enviado_en == MOMENTO_FIJO


def test_conserva_los_emoji() -> None:
    contenido = _parsear(webhook_de_texto(texto="hola 😀 qué tal"))

    assert contenido.mensajes[0].texto == "hola 😀 qué tal"


def test_no_pierde_mensajes_de_un_lote() -> None:
    """`messages[]` es un array: devolver sólo el primero perdería el resto."""
    contenido = _parsear(webhook_con_varios_mensajes(3))

    assert len(contenido.mensajes) == 3
    assert [m.texto for m in contenido.mensajes] == ["mensaje 0", "mensaje 1", "mensaje 2"]


def test_recorre_todas_las_entradas() -> None:
    """`entry[]` también es un array."""
    contenido = _parsear(webhook_con_dos_entradas())

    assert {m.wamid for m in contenido.mensajes} == {"wamid.PRIMERO", "wamid.SEGUNDO"}


def test_acota_la_cantidad_de_mensajes_por_request() -> None:
    """Sin tope, un lote de 1000 encadenaría 1000 llamadas a Graph."""
    contenido = _parsear(webhook_con_varios_mensajes(MAXIMO_MENSAJES + 5))

    assert len(contenido.mensajes) == MAXIMO_MENSAJES
    assert contenido.ignorados == 5


# --- Lo que NO es un mensaje ------------------------------------------------


def test_un_acuse_de_entrega_no_es_un_mensaje() -> None:
    """Llega con `field: "messages"`, igual que un mensaje: la trampa clásica."""
    contenido = _parsear(acuse_de_entrega())

    assert contenido.mensajes == ()
    assert contenido.acuses == 1
    assert contenido.hay_trabajo is False


def test_un_mensaje_no_texto_se_descarta() -> None:
    contenido = _parsear(mensaje_no_texto("image"))

    assert contenido.mensajes == ()
    assert contenido.no_soportados == 1


def test_un_valor_desconocido_se_cuenta_y_reporta_sus_claves() -> None:
    crudo = json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [{"changes": [{"value": {"algo_nuevo": [1]}, "field": "messages"}]}],
        }
    ).encode()

    contenido = _parsear(crudo)

    assert contenido.ignorados == 1
    assert "algo_nuevo" in contenido.claves_desconocidas


# --- Robustez: nunca lanza --------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [b"{}", b"[]", b'"texto"', b"null", b'{"entry": "no es lista"}', b'{"entry": [null]}'],
    ids=["vacio", "lista", "string", "null", "entry_no_lista", "entry_con_null"],
)
def test_un_payload_deforme_no_lanza(payload: bytes) -> None:
    """Un 500 acá haría que Meta reintente durante horas."""
    contenido = _parsear(payload)

    assert contenido.mensajes == ()


def test_un_mensaje_roto_no_se_lleva_puestos_a_los_demas() -> None:
    payload = json.loads(webhook_con_varios_mensajes(3))
    payload["entry"][0]["changes"][0]["value"]["messages"][1] = {"type": "text", "roto": True}

    contenido = parsear(payload, phone_number_id_propio=PHONE_NUMBER_ID, ahora=MOMENTO_FIJO)

    assert len(contenido.mensajes) == 2
    assert [m.texto for m in contenido.mensajes] == ["mensaje 0", "mensaje 2"]


def test_un_remitente_invalido_se_descarta_sin_romper() -> None:
    contenido = _parsear(webhook_de_texto(wa_id="0491141234567"))

    assert contenido.mensajes == ()
    assert contenido.sin_remitente == 1


def test_un_mensaje_sin_from_se_descarta() -> None:
    """Meta está migrando a BSUIDs y puede dejar de mandar `from`."""
    payload = json.loads(webhook_de_texto())
    del payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]

    contenido = parsear(payload, phone_number_id_propio=PHONE_NUMBER_ID, ahora=MOMENTO_FIJO)

    assert contenido.mensajes == ()
    assert contenido.sin_remitente == 1


def test_un_texto_vacio_se_descarta() -> None:
    contenido = _parsear(webhook_de_texto(texto="   "))

    assert contenido.mensajes == ()
    assert contenido.ignorados == 1


# --- Seguridad --------------------------------------------------------------


def test_descarta_eventos_de_otro_numero_de_negocio() -> None:
    """La firma prueba que lo mandó Meta, no que sea para nosotros."""
    contenido = _parsear(webhook_de_texto(phone_number_id="999999999"))

    assert contenido.mensajes == ()
    assert contenido.ajenos == 1


def test_un_mensaje_viejo_se_descarta() -> None:
    """La firma de Meta no lleva timestamp: un cuerpo firmado vale para siempre."""
    viejo = int((MOMENTO_FIJO - FRESCURA_MAXIMA - timedelta(minutes=1)).timestamp())

    contenido = _parsear(webhook_de_texto(timestamp=viejo))

    assert contenido.mensajes == ()
    assert contenido.vencidos == 1


def test_un_mensaje_dentro_de_la_ventana_se_acepta() -> None:
    reciente = int((MOMENTO_FIJO - FRESCURA_MAXIMA + timedelta(minutes=1)).timestamp())

    contenido = _parsear(webhook_de_texto(timestamp=reciente))

    assert len(contenido.mensajes) == 1


# --- Nombre de perfil -------------------------------------------------------


def test_sin_contacts_el_nombre_queda_en_none() -> None:
    contenido = _parsear(webhook_de_texto(nombre=None))

    assert contenido.mensajes[0].nombre_perfil is None


def test_el_nombre_se_empareja_por_wa_id_y_no_por_posicion() -> None:
    payload = json.loads(webhook_con_varios_mensajes(2))
    valor = payload["entry"][0]["changes"][0]["value"]
    valor["messages"][0]["from"] = "5491141234567"
    valor["messages"][1]["from"] = "5491199999999"
    valor["contacts"] = [
        {"profile": {"name": "Segundo"}, "wa_id": "5491199999999"},
        {"profile": {"name": "Primero"}, "wa_id": "5491141234567"},
    ]

    contenido = parsear(payload, phone_number_id_propio=PHONE_NUMBER_ID, ahora=MOMENTO_FIJO)

    nombres = {m.remitente.valor: m.nombre_perfil for m in contenido.mensajes}
    assert nombres == {"+5491141234567": "Primero", "+5491199999999": "Segundo"}


def test_no_adivina_el_nombre_cuando_hay_varios_mensajes() -> None:
    """Emparejar mal congelaría el nombre en la fila del usuario equivocado."""
    payload = json.loads(webhook_con_varios_mensajes(2))
    valor = payload["entry"][0]["changes"][0]["value"]
    valor["messages"][1]["from"] = "5491199999999"
    valor["contacts"] = [{"profile": {"name": "Alguien"}}]  # sin wa_id

    contenido = parsear(payload, phone_number_id_propio=PHONE_NUMBER_ID, ahora=MOMENTO_FIJO)

    assert all(m.nombre_perfil is None for m in contenido.mensajes)


# --- Texto largo ------------------------------------------------------------


def test_un_texto_muy_largo_se_trunca() -> None:
    largo = ("palabra " * 1000).strip()

    contenido = _parsear(webhook_de_texto(texto=largo))

    assert contenido.truncados == 1
    assert len(contenido.mensajes[0].texto) <= LARGO_MAXIMO_TEXTO


def test_el_truncado_corta_en_un_espacio() -> None:
    """Cortar por índice partiría un emoji compuesto por la mitad."""
    largo = ("hola " * 2000).strip()

    texto = _parsear(webhook_de_texto(texto=largo)).mensajes[0].texto

    assert not texto.endswith("hol")
    assert texto.endswith("hola")


# --- El DTO no filtra el contenido -----------------------------------------


def test_el_repr_del_dto_no_expone_el_texto() -> None:
    """El contenido del mensaje es privado: no debe salir en un traceback."""
    mensaje = _parsear(webhook_de_texto(texto="mi secreto medico")).mensajes[0]

    assert "mi secreto medico" not in repr(mensaje)
    assert WA_ID in repr(mensaje) or TELEFONO_E164 in repr(mensaje)
