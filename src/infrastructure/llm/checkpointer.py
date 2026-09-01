"""Memoria conversacional persistida y cifrada en Postgres (PB-013, RF-09 · RF-18).

Es la pieza que hace que un redeploy no borre las conversaciones ni las
confirmaciones pendientes de RF-08: el checkpointer del grafo pasa de la RAM a
la base de Supabase, que es Postgres.

Dos decisiones de este módulo:

- **Los checkpoints se cifran antes de tocar la base.** Llevan la conversación
  completa —lo que la persona escribe puede ser "mi diagnóstico médico"—, así
  que reciben el mismo trato que los tokens OAuth: Fernet, con una clave
  DERIVADA de la maestra con etiqueta propia. En Postgres sólo entra texto
  ilegible.
- **El hardening de RLS lo aplica la app en el arranque.** `setup()` crea sus
  tablas sin RLS, y Supabase por defecto les da acceso a `anon`: quedarían
  legibles vía PostgREST. Como ese DDL no es nuestro, el cierre se ejecuta acá,
  idempotente, justo después. Copia documentada en
  `db/migrations/003_checkpoints_rls.sql` — si editás una, editá la otra (hay
  un test que compara).
"""

from __future__ import annotations

import base64
import hashlib
import hmac

import structlog
from cryptography.fernet import Fernet
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from src.domain.exceptions import ServiceUnavailableError
from src.infrastructure.config.settings import Settings

# El saver exige conexiones con filas-diccionario; el alias evita repetir el
# genérico completo en cada firma.
PoolDeCheckpoints = AsyncConnectionPool[AsyncConnection[DictRow]]

logger = structlog.get_logger(__name__)

# Etiqueta de derivación: firmar states de OAuth, cifrar tokens y cifrar
# checkpoints usan material distinto derivado de la misma maestra. Reutilizar
# una clave para dos propósitos convierte una filtración en dos.
_ETIQUETA = b"lifesync.checkpoint.v1"

# Una instancia, un worker: el pool es chico a propósito. El free tier de
# Supabase admite pocas conexiones y este proceso no necesita más.
POOL_MIN = 1
POOL_MAX = 3
TIMEOUT_CONEXION_SEGUNDOS = 10.0

# Las cuatro tablas que crea el setup() del saver. Si una versión nueva de la
# librería agrega una tabla, hay que sumarla acá y en la migración 003.
TABLAS_DEL_SAVER = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
)

# ENABLE RLS + REVOKE: deny-by-default, igual que usuarios y oauth_tokens.
# La app entra por psycopg como rol dueño, así que a ella no la afecta.
# De a un statement por elemento: psycopg3 usa el protocolo extendido y no
# acepta varios comandos en un mismo execute().
SENTENCIAS_DE_HARDENING = tuple(
    sentencia
    for tabla in TABLAS_DEL_SAVER
    for sentencia in (
        f"ALTER TABLE IF EXISTS public.{tabla} ENABLE ROW LEVEL SECURITY",
        f"REVOKE ALL ON public.{tabla} FROM anon, authenticated",
    )
)


def derivar_clave_de_checkpoints(clave_maestra: str) -> bytes:
    """Deriva la clave Fernet que cifra los checkpoints.

    HMAC-SHA256 de la maestra con la etiqueta del propósito, igual que la clave
    del `state` de OAuth. El resultado se codifica en base64url porque es el
    formato que Fernet exige para sus claves de 32 bytes.
    """
    material = hmac.new(clave_maestra.encode("utf-8"), _ETIQUETA, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(material)


class CifradorFernet:
    """`CipherProtocol` de LangGraph implementado con la Fernet del proyecto.

    Evita sumar pycryptodome: `cryptography` ya está y es la misma primitiva
    que protege los tokens OAuth (RF-18).
    """

    NOMBRE = "fernet"

    def __init__(self, clave: bytes) -> None:
        """Recibe la clave ya derivada (inyección explícita)."""
        self._fernet = Fernet(clave)

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        """Cifra un checkpoint serializado."""
        return self.NOMBRE, self._fernet.encrypt(plaintext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        """Descifra un checkpoint leído de la base."""
        if ciphername != self.NOMBRE:
            raise ValueError(f"Cifrador desconocido en un checkpoint: {ciphername!r}")
        return self._fernet.decrypt(ciphertext)


def crear_serializador_cifrado(clave_maestra: str) -> EncryptedSerializer:
    """Arma el serializer que cifra todo lo que el saver persiste."""
    return EncryptedSerializer(CifradorFernet(derivar_clave_de_checkpoints(clave_maestra)))


async def crear_checkpointer_postgres(
    settings: Settings,
) -> tuple[PoolDeCheckpoints, AsyncPostgresSaver]:
    """Abre el pool, prepara el esquema y devuelve el saver listo.

    Se llama una sola vez, en el `lifespan`. El pool queda a cargo del
    llamador, que debe cerrarlo en el shutdown.

    Raises:
        ServiceUnavailableError: Si falta configuración.
    """
    url = settings.supabase_db_url
    clave_maestra = settings.token_encryption_key
    if url is None or clave_maestra is None:
        raise ServiceUnavailableError(
            "Faltan SUPABASE_DB_URL o TOKEN_ENCRYPTION_KEY para persistir la conversación."
        )

    pool: PoolDeCheckpoints = AsyncConnectionPool(
        url.get_secret_value(),
        min_size=POOL_MIN,
        max_size=POOL_MAX,
        timeout=TIMEOUT_CONEXION_SEGUNDOS,
        open=False,  # se abre explícito: el constructor no debe hacer I/O
        # row_factory=dict_row es lo que el saver espera de sus conexiones;
        # prepare_threshold=0 evita prepared statements, que los poolers de
        # Supabase no siempre toleran.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        connection_class=AsyncConnection[DictRow],
    )
    await pool.open(wait=True, timeout=TIMEOUT_CONEXION_SEGUNDOS)

    saver = AsyncPostgresSaver(
        conn=pool,
        serde=crear_serializador_cifrado(clave_maestra.get_secret_value()),
    )
    # Idempotente: la librería versiona sus migraciones en checkpoint_migrations.
    await saver.setup()
    await _endurecer_permisos(pool)

    logger.info("checkpointer.postgres_listo", tablas=len(TABLAS_DEL_SAVER))
    return pool, saver


async def _endurecer_permisos(pool: PoolDeCheckpoints) -> None:
    """Cierra el acceso PostgREST a las tablas del saver (RF-18).

    Sin esto, `anon` podría leer los checkpoints vía la API REST de Supabase,
    porque las tablas nuevas de `public` nacen con GRANT para ese rol. Es
    idempotente y corre en cada arranque, así el cierre sobrevive a cualquier
    tabla que una versión nueva del saver recree.
    """
    async with pool.connection() as conexion:
        for sentencia in SENTENCIAS_DE_HARDENING:
            await conexion.execute(sentencia.encode("utf-8"))
    logger.info("checkpointer.rls_aplicado", tablas=list(TABLAS_DEL_SAVER))


async def cerrar_pool(pool: PoolDeCheckpoints | None) -> None:
    """Cierra el pool de conexiones. Tolera None y errores de cierre."""
    if pool is None:
        return
    try:
        await pool.close()
    except Exception as exc:  # fallar al apagar no rompe el shutdown
        logger.warning("checkpointer.cierre_con_error", tipo=type(exc).__name__)
