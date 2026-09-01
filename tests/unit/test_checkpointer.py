"""Tests de la memoria persistida y cifrada del agente (PB-013).

El saver de Postgres es de la librería y no se re-testea; lo nuestro sí: el
cifrado, la derivación de claves, el hardening y el cableado. Todo sin red.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.external.google import estado as estado_oauth
from src.infrastructure.llm.checkpointer import (
    SENTENCIAS_DE_HARDENING,
    TABLAS_DEL_SAVER,
    CifradorFernet,
    crear_serializador_cifrado,
    derivar_clave_de_checkpoints,
)

CLAVE_MAESTRA = "clave-maestra-de-prueba"
CONTENIDO_PRIVADO = "mi diagnostico medico es confidencial"


def _cifrador() -> CifradorFernet:
    return CifradorFernet(derivar_clave_de_checkpoints(CLAVE_MAESTRA))


# --- El cifrador ------------------------------------------------------------


def test_ida_y_vuelta() -> None:
    cifrador = _cifrador()

    nombre, cifrado = cifrador.encrypt(b"hola mundo")

    assert nombre == "fernet"
    assert cifrador.decrypt(nombre, cifrado) == b"hola mundo"


def test_el_cifrado_no_contiene_el_texto_plano() -> None:
    _, cifrado = _cifrador().encrypt(CONTENIDO_PRIVADO.encode())

    assert CONTENIDO_PRIVADO.encode() not in cifrado


def test_rechaza_un_cifrador_desconocido() -> None:
    """Un checkpoint marcado con otro cifrador no se descifra a ciegas."""
    with pytest.raises(ValueError, match="desconocido"):
        _cifrador().decrypt("aes", b"lo-que-sea")


def test_dos_claves_maestras_no_se_leen_entre_si() -> None:
    from cryptography.fernet import InvalidToken

    _, cifrado = _cifrador().encrypt(b"secreto")
    otro = CifradorFernet(derivar_clave_de_checkpoints("otra-clave"))

    with pytest.raises(InvalidToken):
        otro.decrypt("fernet", cifrado)


# --- Derivación de claves ----------------------------------------------------


def test_la_clave_se_deriva_y_no_es_la_maestra() -> None:
    derivada = derivar_clave_de_checkpoints(CLAVE_MAESTRA)

    assert derivada != CLAVE_MAESTRA.encode()
    assert len(derivada) == 44  # 32 bytes en base64url: formato de clave Fernet


def test_cada_proposito_deriva_material_distinto() -> None:
    """Checkpoints y state de OAuth salen de la misma maestra pero no comparten clave."""
    import base64

    material_checkpoints = base64.urlsafe_b64decode(derivar_clave_de_checkpoints(CLAVE_MAESTRA))
    material_oauth = estado_oauth.derivar_clave(CLAVE_MAESTRA)

    assert material_checkpoints != material_oauth


def test_la_derivacion_es_estable() -> None:
    """Si cambiara entre versiones, todos los checkpoints quedarían ilegibles."""
    assert derivar_clave_de_checkpoints(CLAVE_MAESTRA) == derivar_clave_de_checkpoints(
        CLAVE_MAESTRA
    )


# --- El serializer completo, como lo usa el saver ----------------------------


def test_un_mensaje_serializado_queda_ilegible() -> None:
    """Es la afirmación central de RF-18 en este PB: a la base no llega la charla."""
    serde = crear_serializador_cifrado(CLAVE_MAESTRA)

    tipo, datos = serde.dumps_typed(HumanMessage(CONTENIDO_PRIVADO))

    assert tipo.endswith("+fernet")  # marcado como cifrado
    assert CONTENIDO_PRIVADO.encode() not in datos


def test_el_mensaje_se_recupera_intacto() -> None:
    serde = crear_serializador_cifrado(CLAVE_MAESTRA)

    recuperado = serde.loads_typed(serde.dumps_typed(HumanMessage(CONTENIDO_PRIVADO)))

    assert isinstance(recuperado, HumanMessage)
    assert recuperado.content == CONTENIDO_PRIVADO


def test_puede_leer_datos_sin_cifrar() -> None:
    """El serializer tolera checkpoints viejos en claro: lee, aunque ya no escriba así."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = crear_serializador_cifrado(CLAVE_MAESTRA)
    en_claro = JsonPlusSerializer().dumps_typed({"a": 1})

    assert serde.loads_typed(en_claro) == {"a": 1}


# --- El hardening ------------------------------------------------------------


def test_el_hardening_cubre_las_cuatro_tablas() -> None:
    sql = " ".join(SENTENCIAS_DE_HARDENING)

    for tabla in ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"):
        assert f"public.{tabla} ENABLE ROW LEVEL SECURITY" in sql
        assert f"public.{tabla} FROM anon, authenticated" in sql
    assert len(TABLAS_DEL_SAVER) == 4


def test_la_migracion_documentada_no_diverge_del_codigo() -> None:
    """La 003 es una copia del hardening: si se edita una, hay que editar la otra."""
    migracion = Path("db/migrations/003_checkpoints_rls.sql").read_text(encoding="utf-8")

    for sentencia in SENTENCIAS_DE_HARDENING:
        assert sentencia in migracion, f"la migración perdió: {sentencia}"


# --- El cableado -------------------------------------------------------------


def test_settings_sin_url_no_configuran_el_checkpointer() -> None:
    s = Settings(_env_file=None, environment=Environment.TESTING)

    assert s.checkpointer_configurado is False


def test_settings_exigen_tambien_la_clave_de_cifrado() -> None:
    """Persistir la conversación en claro violaría RF-18: sin clave, no hay persistencia."""
    solo_url = Settings(
        _env_file=None,
        environment=Environment.TESTING,
        supabase_db_url="postgresql://x:y@host:5432/db",
    )

    assert solo_url.checkpointer_configurado is False


def test_settings_con_todo_configuran_el_checkpointer() -> None:
    s = Settings(
        _env_file=None,
        environment=Environment.TESTING,
        supabase_db_url="postgresql://x:y@host:5432/db",
        token_encryption_key="clave",
    )

    assert s.checkpointer_configurado is True
    assert "postgresql" not in repr(s)  # la URL lleva el password: no se filtra


def test_una_url_vacia_cuenta_como_ausente() -> None:
    s = Settings(_env_file=None, environment=Environment.TESTING, supabase_db_url="  ")

    assert s.supabase_db_url is None


async def test_el_agente_usa_el_checkpointer_inyectado() -> None:
    """El composition root decide; el grafo usa lo que recibe."""
    from src.infrastructure.llm.agente_gemini import crear_agente_gemini

    inyectado = InMemorySaver()
    s = Settings(
        _env_file=None,
        environment=Environment.TESTING,
        google_api_key="key-de-prueba",
    )

    agente = crear_agente_gemini(s, None, checkpointer=inyectado)

    assert agente._grafo.checkpointer is inyectado
