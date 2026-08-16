"""Cifrado de datos sensibles en reposo (RF-01 + RF-18).

Los tokens OAuth2 se cifran **en la aplicación**, antes de salir hacia
PostgreSQL: a la base sólo llega texto cifrado. Así, ni un dump de la base ni
una fuga de la `service_role key` exponen credenciales de los usuarios.

Se usa Fernet (AES-128-CBC + HMAC-SHA256, con IV aleatorio por mensaje), que
además autentica: si alguien altera una fila, el descifrado falla en vez de
devolver basura.

Generar una clave nueva:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import structlog
from cryptography.fernet import Fernet, InvalidToken

from src.domain.exceptions import EncryptionError

logger = structlog.get_logger(__name__)

AYUDA_CLAVE = (
    "Generá una con: "
    'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
)


class TokenCipher:
    """Cifra y descifra credenciales con una clave simétrica.

    Ningún método de esta clase loguea el texto plano ni el cifrado.
    """

    def __init__(self, clave: str) -> None:
        """Prepara el cifrador.

        La clave se valida acá y no en el primer uso: si es inválida, conviene
        que la aplicación falle al arrancar y no al guardar el primer token.

        Args:
            clave: Clave Fernet (32 bytes en base64 urlsafe).

        Raises:
            EncryptionError: Si la clave no tiene el formato esperado.
        """
        try:
            self._fernet = Fernet(clave.encode("utf-8"))
        except (ValueError, TypeError) as exc:
            raise EncryptionError(f"TOKEN_ENCRYPTION_KEY inválida. {AYUDA_CLAVE}") from exc

    def cifrar(self, texto_plano: str) -> str:
        """Devuelve el texto cifrado, listo para guardar en la base.

        Dos llamadas con el mismo texto devuelven cifrados distintos, porque
        Fernet usa un IV aleatorio en cada operación.
        """
        return self._fernet.encrypt(texto_plano.encode("utf-8")).decode("utf-8")

    def descifrar(self, texto_cifrado: str) -> str:
        """Devuelve el texto original a partir del cifrado.

        Raises:
            EncryptionError: Si el dato fue alterado o se cifró con otra clave.
        """
        try:
            return self._fernet.decrypt(texto_cifrado.encode("utf-8")).decode("utf-8")
        except (InvalidToken, ValueError, TypeError) as exc:
            # Sin detalles del contenido: sólo el hecho de que falló.
            logger.error("cifrado.descifrado_fallido")
            raise EncryptionError(
                "No se pudo descifrar el dato: fue alterado o la TOKEN_ENCRYPTION_KEY "
                "no es la misma con la que se cifró."
            ) from exc
