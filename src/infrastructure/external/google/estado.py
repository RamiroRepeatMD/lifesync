"""Firmado del parámetro `state` del flujo OAuth2 (PB-009).

`state` es lo único que ata la vuelta de Google al usuario que pidió conectar su
cuenta. Sin firmarlo, cualquiera podría fabricar un callback con el id de otra
persona y **dejar su propia cuenta de Google enganchada al usuario de LifeSync
que elija**. Por eso se firma siempre y se verifica siempre.

El formato es deliberadamente simple —no es un JWT, porque lo único que hay que
transportar es un id y un vencimiento—:

    base64url(usuario_id.vencimiento) . base64url(HMAC-SHA256)

**No lleva tabla ni caché.** La firma es lo que lo hace verificable, así que el
callback no necesita consultar nada. Es la misma decisión que en el webhook de
Meta: la validez sale de la firma, no de un registro previo.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import datetime
from uuid import UUID

from src.domain.exceptions import InvalidValueError

# Cuánto vive un link de autorización. Corto a propósito: el link queda escrito
# para siempre en un chat de WhatsApp, así que conviene que caduque solo.
VIGENCIA_SEGUNDOS = 10 * 60

# Etiqueta de derivación. La clave del `state` NO es directamente
# TOKEN_ENCRYPTION_KEY: se deriva de ella con esta etiqueta, para que firmar
# estados y cifrar credenciales usen material distinto. Reutilizar una clave
# para dos propósitos es lo que convierte la filtración de uno en la de los dos.
_ETIQUETA = b"lifesync.oauth.state.v1"

_SEPARADOR = "."
_PARTES_ESPERADAS = 2


def derivar_clave(clave_maestra: str) -> bytes:
    """Deriva la clave de firmado del `state` a partir de la clave de cifrado."""
    return hmac.new(clave_maestra.encode("utf-8"), _ETIQUETA, hashlib.sha256).digest()


def firmar(usuario_id: UUID, clave: bytes, ahora: datetime) -> str:
    """Arma un `state` firmado que vence en `VIGENCIA_SEGUNDOS`."""
    vence_en = int(ahora.timestamp()) + VIGENCIA_SEGUNDOS
    carga = f"{usuario_id}{_SEPARADOR}{vence_en}"
    return f"{_b64(carga.encode('utf-8'))}{_SEPARADOR}{_b64(_firma_de(carga, clave))}"


def verificar(estado: str, clave: bytes, ahora: datetime) -> UUID:
    """Devuelve el id de usuario de un `state` válido.

    Raises:
        InvalidValueError: Si está deformado, mal firmado o vencido. El motivo
            no viaja al usuario: quien manda un state inválido es un atacante o
            alguien con un link viejo, y a ninguno le sirve el detalle.
    """
    partes = estado.split(_SEPARADOR)
    if len(partes) != _PARTES_ESPERADAS:
        raise InvalidValueError("El parámetro state no tiene el formato esperado.")

    try:
        carga = _des_b64(partes[0]).decode("utf-8")
        firma_recibida = _des_b64(partes[1])
    except (ValueError, UnicodeDecodeError):
        raise InvalidValueError("El parámetro state no se pudo decodificar.") from None

    # compare_digest y no ==: comparar firmas byte a byte con salida temprana
    # filtra, por lo que tarda, cuántos bytes acertó quien la fabricó.
    if not hmac.compare_digest(firma_recibida, _firma_de(carga, clave)):
        raise InvalidValueError("La firma del parámetro state no es válida.")

    usuario_id, _, vence_en = carga.partition(_SEPARADOR)
    try:
        vencimiento = int(vence_en)
        identificador = UUID(usuario_id)
    except ValueError:
        raise InvalidValueError("El parámetro state no tiene el formato esperado.") from None

    if vencimiento <= int(ahora.timestamp()):
        raise InvalidValueError("El enlace de autorización venció.")
    return identificador


def _firma_de(carga: str, clave: bytes) -> bytes:
    return hmac.new(clave, carga.encode("utf-8"), hashlib.sha256).digest()


def _b64(crudo: bytes) -> str:
    """base64 urlsafe sin relleno: el `=` se escapa feo en una query string."""
    return base64.urlsafe_b64encode(crudo).decode("ascii").rstrip("=")


def _des_b64(texto: str) -> bytes:
    relleno = "=" * (-len(texto) % 4)
    return base64.urlsafe_b64decode(texto + relleno)
