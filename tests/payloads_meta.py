"""Payloads de webhook de Meta para los tests (PB-004).

Los payloads se manejan como **bytes**, no como diccionarios, porque la firma
HMAC va sobre los bytes exactos: si un test construyera el cuerpo con `json=`
y firmara `json.dumps()` del mismo diccionario, los bytes podrían diferir y el
test pasaría o fallaría por la razón equivocada.

Los ejemplos vienen de la documentación oficial de Meta, con el número de
teléfono cambiado por uno de prueba.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

# Números de prueba. El del negocio es el que Meta pone en sus ejemplos.
WA_ID = "5491141234567"
TELEFONO_E164 = "+5491141234567"
PHONE_NUMBER_ID = "106540352242922"
DISPLAY_PHONE_NUMBER = "15550783881"
WAMID = "wamid.HBgLMTY1MDM4Nzk0MzkVAgASGBQzQTRBNjU5OUFFRTAzODEwMTQ0RgA="

# Momento fijo para los tests del parser, que reciben `ahora` explícito.
TIMESTAMP_FIJO = 1749416383
MOMENTO_FIJO = datetime.fromtimestamp(TIMESTAMP_FIJO, tz=UTC)

# Los parámetros del handshake llevan punto en el nombre. Se define una sola
# vez para que ningún test los escriba mal por su cuenta.
PARAM_MODO = "hub.mode"
PARAM_CHALLENGE = "hub.challenge"
PARAM_VERIFY_TOKEN = "hub.verify_token"


def _a_bytes(payload: dict[str, Any]) -> bytes:
    """Serializa de forma estable. Los tests firman exactamente esta salida."""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _envolver(valor: dict[str, Any]) -> dict[str, Any]:
    """Mete un `value` en la estructura entry/changes que usa Meta."""
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "102290129340398", "changes": [{"value": valor, "field": "messages"}]}],
    }


def _metadata(phone_number_id: str = PHONE_NUMBER_ID) -> dict[str, Any]:
    return {
        "display_phone_number": DISPLAY_PHONE_NUMBER,
        "phone_number_id": phone_number_id,
    }


def mensaje_de_texto(
    *,
    texto: str = "Hola, ¿me agendás una reunión?",
    wa_id: str = WA_ID,
    wamid: str = WAMID,
    timestamp: int = TIMESTAMP_FIJO,
    nombre: str | None = "Ramiro Gracia",
    phone_number_id: str = PHONE_NUMBER_ID,
) -> dict[str, Any]:
    """Un `value` con un único mensaje de texto entrante."""
    valor: dict[str, Any] = {
        "messaging_product": "whatsapp",
        "metadata": _metadata(phone_number_id),
        "messages": [
            {
                "from": wa_id,
                "id": wamid,
                "timestamp": str(timestamp),
                "type": "text",
                "text": {"body": texto},
            }
        ],
    }
    if nombre is not None:
        valor["contacts"] = [{"profile": {"name": nombre}, "wa_id": wa_id}]
    return valor


def webhook_de_texto(**kwargs: Any) -> bytes:
    """Payload completo, listo para firmar y postear."""
    return _a_bytes(_envolver(mensaje_de_texto(**kwargs)))


def webhook_fresco(**kwargs: Any) -> bytes:
    """Igual que `webhook_de_texto` pero con timestamp de ahora.

    Lo usan los tests de integración, que pasan por el endpoint real y por lo
    tanto quedan sujetos a la ventana de frescura del parser.
    """
    kwargs.setdefault("timestamp", int(datetime.now(UTC).timestamp()))
    return webhook_de_texto(**kwargs)


def webhook_con_varios_mensajes(cantidad: int, *, timestamp: int = TIMESTAMP_FIJO) -> bytes:
    """Un `value` con varios mensajes, para probar que no se pierde ninguno."""
    valor = mensaje_de_texto(timestamp=timestamp)
    valor["messages"] = [
        {
            "from": WA_ID,
            "id": f"wamid.MENSAJE{indice}",
            "timestamp": str(timestamp),
            "type": "text",
            "text": {"body": f"mensaje {indice}"},
        }
        for indice in range(cantidad)
    ]
    return _a_bytes(_envolver(valor))


def webhook_con_dos_entradas(*, timestamp: int = TIMESTAMP_FIJO) -> bytes:
    """Dos `entry`, cada una con su mensaje. Meta agrupa así los lotes."""
    return _a_bytes(
        {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "102290129340398",
                    "changes": [
                        {
                            "value": mensaje_de_texto(wamid="wamid.PRIMERO", timestamp=timestamp),
                            "field": "messages",
                        }
                    ],
                },
                {
                    "id": "102290129340399",
                    "changes": [
                        {
                            "value": mensaje_de_texto(wamid="wamid.SEGUNDO", timestamp=timestamp),
                            "field": "messages",
                        }
                    ],
                },
            ],
        }
    )


def acuse_de_entrega() -> bytes:
    """Notificación de estado. Llega con `field: "messages"`, igual que un mensaje."""
    return _a_bytes(
        _envolver(
            {
                "messaging_product": "whatsapp",
                "metadata": _metadata(),
                "statuses": [
                    {
                        "id": WAMID,
                        "status": "delivered",
                        "timestamp": "1750263773",
                        "recipient_id": WA_ID,
                        "conversation": {
                            "id": "6ceb9d929c9bdc4f90e967a32f8639b4",
                            "origin": {"type": "service"},
                        },
                        "pricing": {
                            "billable": True,
                            "pricing_model": "CBP",
                            "category": "service",
                        },
                    }
                ],
            }
        )
    )


def mensaje_no_texto(tipo: str = "image", *, timestamp: int = TIMESTAMP_FIJO) -> bytes:
    """Un mensaje que no es de texto: hay que descartarlo sin romperse."""
    return _a_bytes(
        _envolver(
            {
                "messaging_product": "whatsapp",
                "metadata": _metadata(),
                "contacts": [{"profile": {"name": "Ramiro"}, "wa_id": WA_ID}],
                "messages": [
                    {
                        "from": WA_ID,
                        "id": WAMID,
                        "timestamp": str(timestamp),
                        "type": tipo,
                        tipo: {"id": "1234", "mime_type": "image/jpeg"},
                    }
                ],
            }
        )
    )
