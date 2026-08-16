"""Validación de la firma de los webhooks de Meta (RF-18).

Meta firma cada POST con HMAC-SHA256 del cuerpo, usando el **App Secret** de la
aplicación. Dos precisiones que rompen implementaciones enteras si se pasan por
alto:

1. El secreto es el **App Secret**, NO el verify token. El verify token se usa
   sólo en el handshake GET y es un valor que inventamos nosotros.
2. El HMAC va sobre los **bytes crudos** del cuerpo. Parsear el JSON y volver a
   serializarlo cambia el digest: `json.dumps` altera espacios, orden de claves
   y escapado de Unicode, y los mensajes de WhatsApp están llenos de emoji.

No se implementa el `X-Hub-Signature` viejo (SHA-1) a propósito: un camino de
respaldo con un algoritmo más débil es una superficie de downgrade.
"""

from __future__ import annotations

import hmac
from hashlib import sha256

HEADER_FIRMA = "X-Hub-Signature-256"
PREFIJO = "sha256="


def firma_valida(cuerpo: bytes, cabecera: str | None, secreto: str) -> bool:
    """Indica si la firma de la cabecera corresponde al cuerpo recibido.

    Devuelve un bool en vez de lanzar: quien llama decide el código HTTP, y una
    firma inválida no es un error de negocio del que haya que informar a nadie.

    Args:
        cuerpo: Bytes exactos del cuerpo, tal como llegaron.
        cabecera: Valor de `X-Hub-Signature-256`, o None si no vino.
        secreto: App Secret de la aplicación de Meta.
    """
    if not cabecera or not cabecera.startswith(PREFIJO):
        return False

    esperada = hmac.new(secreto.encode("utf-8"), cuerpo, sha256).hexdigest()
    recibida = cabecera[len(PREFIJO) :]

    # compare_digest y no ==: evita filtrar por tiempo cuántos caracteres
    # coincidían, que permitiría construir una firma válida byte a byte.
    return hmac.compare_digest(esperada, recibida)


def firmar(cuerpo: bytes, secreto: str) -> str:
    """Genera la cabecera de firma para un cuerpo dado.

    En producción no se usa —firmar es tarea de Meta— pero es lo que permite
    que los tests construyan peticiones legítimas sin duplicar el algoritmo.
    """
    return PREFIJO + hmac.new(secreto.encode("utf-8"), cuerpo, sha256).hexdigest()
