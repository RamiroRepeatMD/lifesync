"""Tests del cifrado de credenciales en reposo (RF-18)."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from src.domain.exceptions import EncryptionError
from src.infrastructure.persistence.encryption import TokenCipher

TOKEN_DE_PRUEBA = "ya29.a0AfB_byC-token-de-google-de-ejemplo"


def test_cifrar_y_descifrar_devuelve_el_original(cipher: TokenCipher) -> None:
    cifrado = cipher.cifrar(TOKEN_DE_PRUEBA)

    assert cipher.descifrar(cifrado) == TOKEN_DE_PRUEBA


def test_el_texto_cifrado_no_contiene_el_original(cipher: TokenCipher) -> None:
    cifrado = cipher.cifrar(TOKEN_DE_PRUEBA)

    assert TOKEN_DE_PRUEBA not in cifrado


def test_dos_cifrados_del_mismo_valor_son_distintos(cipher: TokenCipher) -> None:
    """Fernet usa un IV aleatorio: por eso la columna cifrada no se puede indexar."""
    primero = cipher.cifrar(TOKEN_DE_PRUEBA)
    segundo = cipher.cifrar(TOKEN_DE_PRUEBA)

    assert primero != segundo
    assert cipher.descifrar(primero) == cipher.descifrar(segundo)


def test_soporta_un_token_largo(cipher: TokenCipher) -> None:
    """Un access_token de Google con muchos scopes ronda los 2 KB."""
    token_largo = "x" * 2048

    assert cipher.descifrar(cipher.cifrar(token_largo)) == token_largo


def test_soporta_caracteres_no_ascii(cipher: TokenCipher) -> None:
    assert cipher.descifrar(cipher.cifrar("ñandú-áéíóú")) == "ñandú-áéíóú"


@pytest.mark.parametrize(
    "clave_invalida",
    ["", "no-es-base64", "Y29ydGE=", "x" * 100],
    ids=["vacia", "no_base64", "muy_corta", "largo_incorrecto"],
)
def test_clave_invalida_falla_al_construir(clave_invalida: str) -> None:
    """Conviene fallar al arrancar la app, no al guardar el primer token."""
    with pytest.raises(EncryptionError, match="TOKEN_ENCRYPTION_KEY"):
        TokenCipher(clave_invalida)


def test_descifrar_texto_corrupto_falla(cipher: TokenCipher) -> None:
    cifrado = cipher.cifrar(TOKEN_DE_PRUEBA)
    corrupto = cifrado[:-4] + "AAAA"

    with pytest.raises(EncryptionError):
        cipher.descifrar(corrupto)


def test_descifrar_con_otra_clave_falla(cipher: TokenCipher) -> None:
    """Fernet autentica: una clave ajena no devuelve basura, falla."""
    cifrado = cipher.cifrar(TOKEN_DE_PRUEBA)
    otro_cipher = TokenCipher(Fernet.generate_key().decode("utf-8"))

    with pytest.raises(EncryptionError):
        otro_cipher.descifrar(cifrado)


def test_descifrar_algo_que_no_es_un_token_falla(cipher: TokenCipher) -> None:
    with pytest.raises(EncryptionError):
        cipher.descifrar("esto no es un token cifrado")


def test_el_mensaje_de_error_no_expone_el_dato(cipher: TokenCipher) -> None:
    """El texto del error va al usuario: no debe filtrar el secreto."""
    cifrado = cipher.cifrar(TOKEN_DE_PRUEBA)
    otro_cipher = TokenCipher(Fernet.generate_key().decode("utf-8"))

    with pytest.raises(EncryptionError) as capturado:
        otro_cipher.descifrar(cifrado)

    assert cifrado not in str(capturado.value)
    assert TOKEN_DE_PRUEBA not in str(capturado.value)
