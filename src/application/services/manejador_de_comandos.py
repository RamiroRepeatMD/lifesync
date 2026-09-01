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
    "• /estado — qué cuentas tenés conectadas\n\n"
    "Cuando conectes tu Google voy a poder consultarte la agenda."
)

ESTADO_SIN_CUENTAS = (
    "Todavía no tenés ninguna cuenta conectada.\n\n"
    "Escribí /conectar para vincular tu Google y que pueda ver tu calendario."
)

# OJO: este texto no puede prometer lo que el agente todavía no sabe hacer.
# Hoy la conexión está guardada y verificada, pero **no hay ninguna herramienta
# que lea el calendario** (eso es PB-015). Decir acá "preguntame qué tenés
# mañana" mandaba a la persona a un callejón: el agente le contestaba, con
# razón, que no tiene acceso. Actualizarlo es parte de PB-015.
ESTADO_CON_GOOGLE = (
    "Tenés tu cuenta de Google conectada. ✅\n\n"
    "Todavía no puedo leer tu calendario: esa parte está en construcción. "
    "Cuando esté lista, vas a poder preguntarme qué tenés en el día."
)

CONECTAR_NO_DISPONIBLE = (
    "Todavía no puedo conectar cuentas de Google: al asistente le falta esa "
    "configuración.\n\nEscribí /ayuda para ver lo que sí puedo hacer."
)

CONECTAR_ERROR = "No pude generar el enlace para conectar tu cuenta. Probá de nuevo en un minuto."


def _texto_del_enlace(enlace: str) -> str:
    return (
        "Para conectar tu cuenta de Google, entrá acá y autorizá el acceso:\n\n"
        f"{enlace}\n\n"
        "El enlace vence en 10 minutos. Sólo te voy a pedir permiso para *ver* "
        "tu calendario, no para modificarlo."
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
        return None

    async def _estado(self, usuario: Usuario) -> str:
        """Reporta qué integraciones tiene conectadas la persona (RF-12)."""
        if self._conectar_google is None or usuario.id is None:
            return ESTADO_SIN_CUENTAS

        conectado = await self._conectar_google.esta_conectado(usuario.id)
        return ESTADO_CON_GOOGLE if conectado else ESTADO_SIN_CUENTAS

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
