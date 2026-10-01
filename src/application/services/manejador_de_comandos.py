"""Comandos de respuesta directa (RF-11, RF-12).

Reemplaza a `router_de_comandos`, que era una función pura. Dejó de alcanzar en
PB-009: `/estado` tiene que decir qué cuentas hay conectadas de verdad y
`/conectar` tiene que armar un enlace para **esa** persona, y las dos cosas
necesitan datos.

Lo que **no** cambió es por qué los comandos siguen sin pasar por el modelo:

- **RF-11 pide un sistema de ayuda.** Si `/ayuda` dependiera del LLM, cambiaría
  de texto en cada invocación y podría inventar funciones inexistentes.
- **`/conectar` entrega un enlace con credenciales adentro.** Eso no se le
  delega a algo que improvisa: el enlace tiene que salir exacto o no sirve.
- Siguen funcionando aunque falte la API key del modelo.

El contrato se mantiene: devuelve el texto de la respuesta, o `None` para decir
"esto es lenguaje natural, que lo maneje el agente".
"""

from __future__ import annotations

from datetime import datetime

import structlog

from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.usuario import Usuario

logger = structlog.get_logger(__name__)

AYUDA = (
    "Soy LifeSync, tu asistente personal.\n\n"
    "Escribime en lenguaje natural y hago lo que pueda: preguntame la hora, "
    "pedime que te ayude a organizarte o contame qué necesitás.\n\n"
    "Comandos:\n"
    "• /ayuda — esta lista\n"
    "• /conectar — vincular tu cuenta de Google\n"
    "• /desconectar — desvincularla y revocar el permiso\n"
    "• /estado — qué cuentas tenés conectadas\n\n"
    "Cuando conectes tu Google voy a poder consultarte la agenda."
)

ESTADO_SIN_CUENTAS = (
    "Todavía no tenés ninguna cuenta conectada.\n\n"
    "Escribí /conectar para vincular tu Google y que pueda ver tu calendario."
)

# Este texto sólo puede prometer lo que el agente sabe hacer de verdad (la
# regla viene de PB-015, cuando prometió el calendario antes de tiempo).
# Desde PB-017 el calendario está completo: leer, crear, modificar y eliminar.
ESTADO_CON_GOOGLE = (
    "Tenés tu cuenta de Google conectada. ✅\n\n"
    "Puedo leer tu calendario y crear (incluso eventos que se repiten), "
    "modificar o eliminar eventos; también "
    "ver tus tareas pendientes, anotar nuevas, marcarlas como hechas, "
    "cambiarles la fecha o eliminarlas; y buscar, leer y mandar correos de Gmail "
    "—siempre te pido confirmación antes de tocar algo."
)

CONECTAR_NO_DISPONIBLE = (
    "Todavía no puedo conectar cuentas de Google: al asistente le falta esa "
    "configuración.\n\nEscribí /ayuda para ver lo que sí puedo hacer."
)

CONECTAR_ERROR = "No pude generar el enlace para conectar tu cuenta. Probá de nuevo en un minuto."

DESCONECTAR_SIN_CUENTA = "No tenés ninguna cuenta conectada, así que no hay nada que desconectar."

# Confirmación en dos pasos SIN estado: lo pendiente vive en el texto del
# segundo comando, no en memoria. Es la versión para comandos de la regla de
# RF-08 — los comandos no pasan por el grafo, así que acá no hay interrupt.
DESCONECTAR_PIDE_CONFIRMACION = (
    "Vas a desconectar tu cuenta de Google: dejo de poder ver tu calendario y "
    "de gestionar tus eventos.\n\n"
    "Para confirmar, escribí: /desconectar confirmar"
)

DESCONECTAR_LISTO = (
    "Listo, desconecté tu cuenta de Google y revoqué el permiso. "
    "Cuando quieras reconectarla: /conectar"
)

# La revocación es best-effort: si Google no contestó, el token igual se borró
# de nuestro lado, y la persona merece saber cómo cerrar el círculo.
DESCONECTAR_SIN_REVOCAR = (
    "Borré tu cuenta de mi lado, pero no pude confirmar la revocación con "
    "Google. Si querés estar seguro, revisá myaccount.google.com/permissions"
)


def _texto_del_enlace(enlace: str) -> str:
    return (
        "Para conectar tu cuenta de Google, entrá acá y autorizá el acceso:\n\n"
        f"{enlace}\n\n"
        "El enlace vence en 10 minutos. Te voy a pedir permiso para ver tu "
        "calendario y gestionar eventos; nunca toco nada sin confirmarte antes."
    )


class ManejadorDeComandos:
    """Resuelve los comandos que no pasan por el modelo."""

    def __init__(self, conectar_google: ConectarGoogle | None) -> None:
        """Recibe sus dependencias por constructor (inyección explícita).

        `conectar_google` es `None` cuando falta la configuración de OAuth. No
        se usa un *null object* como con el agente porque acá los dos casos
        dicen cosas distintas y ninguna es "responder igual pero peor".
        """
        self._conectar_google = conectar_google

    async def responder(self, texto: str, usuario: Usuario, ahora: datetime) -> str | None:
        """Devuelve la respuesta del comando, o None si no es un comando.

        `None` no significa "no entendí": significa "esto no me toca a mí".
        Devolverlo en vez de un texto de descarte es lo que permite que decida
        el agente y no este manejador.

        El comando se toma del primer token, así que "/Ayuda" y "/ayuda por
        favor" funcionan igual.
        """
        primer_token = texto.strip().lower().split(maxsplit=1)
        comando = primer_token[0] if primer_token else ""

        if comando == "/ayuda":
            return AYUDA
        if comando == "/estado":
            return await self._estado(usuario)
        if comando == "/conectar":
            return await self._conectar(usuario, ahora)
        if comando == "/desconectar":
            resto = texto.strip().lower().split(maxsplit=1)
            confirmado = len(resto) > 1 and resto[1].strip() == "confirmar"
            return await self._desconectar(usuario, confirmado)
        return None

    async def _estado(self, usuario: Usuario) -> str:
        """Reporta qué integraciones tiene conectadas la persona (RF-12)."""
        if self._conectar_google is None or usuario.id is None:
            return ESTADO_SIN_CUENTAS

        conectado = await self._conectar_google.esta_conectado(usuario.id)
        return ESTADO_CON_GOOGLE if conectado else ESTADO_SIN_CUENTAS

    async def _desconectar(self, usuario: Usuario, confirmado: bool) -> str:
        """Desconecta la cuenta de Google, con confirmación en dos pasos (RF-12).

        Es una acción destructiva, así que RF-08 aplica también acá — pero los
        comandos no pasan por el grafo, y el interrupt no existe. La
        confirmación vive en el texto: `/desconectar` sólo explica, y
        `/desconectar confirmar` ejecuta. Determinístico y sin memoria.
        """
        if self._conectar_google is None or usuario.id is None:
            return DESCONECTAR_SIN_CUENTA

        if not await self._conectar_google.esta_conectado(usuario.id):
            return DESCONECTAR_SIN_CUENTA

        if not confirmado:
            return DESCONECTAR_PIDE_CONFIRMACION

        try:
            revocado = await self._conectar_google.desconectar(usuario.id)
        except Exception as exc:  # el comando nunca deja a la persona sin respuesta
            logger.error(
                "google.desconexion_fallida", usuario_id=str(usuario.id), tipo=type(exc).__name__
            )
            return "No pude desconectar tu cuenta ahora mismo. Probá de nuevo en un minuto."

        return DESCONECTAR_LISTO if revocado else DESCONECTAR_SIN_REVOCAR

    async def _conectar(self, usuario: Usuario, ahora: datetime) -> str:
        """Arma el enlace de consentimiento para esta persona."""
        if self._conectar_google is None or usuario.id is None:
            return CONECTAR_NO_DISPONIBLE

        try:
            enlace = self._conectar_google.link_de_autorizacion(usuario.id, ahora)
        except Exception as exc:  # el comando nunca deja a la persona sin respuesta
            logger.error("google.link_fallido", usuario_id=str(usuario.id), tipo=type(exc).__name__)
            return CONECTAR_ERROR

        # El enlace lleva el state firmado: no se loguea entero.
        logger.info("google.link_entregado", usuario_id=str(usuario.id))
        return _texto_del_enlace(enlace)
