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
from src.application.use_cases.conectar_notion import ConectarNotion
from src.domain.entities.usuario import Usuario

logger = structlog.get_logger(__name__)


def _texto_de_ayuda(con_notion: bool) -> str:
    """La ayuda; con Notion configurado suma sus dos comandos (PB-014)."""
    notion = (
        "• /conectar notion — vincular tu Notion\n• /desconectar notion — desvincularlo\n"
        if con_notion
        else ""
    )
    return (
        "Soy LifeSync, tu asistente personal.\n\n"
        "Escribime en lenguaje natural y hago lo que pueda: preguntame la hora, "
        "pedime que te ayude a organizarte o contame qué necesitás. También te "
        'puedo escribir a una hora para recordarte algo ("recordame en 20 minutos '
        'que saque la pizza"), dentro de las próximas 24 horas.\n\n'
        "Comandos:\n"
        "• /ayuda — esta lista\n"
        "• /conectar — vincular tu cuenta de Google\n"
        "• /desconectar — desvincularla y revocar el permiso\n"
        f"{notion}"
        "• /estado — qué cuentas tenés conectadas\n\n"
        "Cuando conectes tu Google voy a poder consultarte la agenda."
    )


AYUDA = _texto_de_ayuda(con_notion=False)
AYUDA_CON_NOTION = _texto_de_ayuda(con_notion=True)

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
    "cambiarles la fecha o eliminarlas; buscar, leer, mandar, responder y "
    "reenviar correos de Gmail; buscar archivos en tu Drive, subir los que me "
    "mandes por acá, compartirlos, renombrarlos, moverlos o mandarlos a la "
    "papelera; y escribirte a una hora para recordarte "
    "algo. Antes de borrar algo, mandar un correo, compartir un archivo o hacer "
    "varias cosas juntas, te pido confirmación."
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


# --- Notion (PB-014) ---------------------------------------------------------
# Se usan sólo con Notion configurado: sin él, los textos de arriba siguen
# idénticos. Con Notion conectado, "ninguna cuenta" deja de ser cierto, y por
# eso hay versiones que dicen cuál falta.

PISTA_DE_NOTION = "También podés conectar tu Notion: /conectar notion"
GOOGLE_SIN_CONECTAR = "Tu cuenta de Google no está conectada: escribí /conectar para vincularla."
NOTION_CONECTADO = (
    "Tu Notion está conectado. ✅ Puedo buscar y leer las páginas y bases que compartiste conmigo."
)
NOTION_SIN_CONECTAR = "Tu Notion no está conectado: escribí /conectar notion para vincularlo."

CONECTAR_NOTION_NO_DISPONIBLE = (
    "Todavía no puedo conectar Notion: al asistente le falta esa configuración.\n\n"
    "Escribí /ayuda para ver lo que sí puedo hacer."
)

DESCONECTAR_GOOGLE_SIN_CUENTA = (
    "No tenés tu cuenta de Google conectada, así que no hay nada que desconectar."
)
DESCONECTAR_NOTION_SIN_CUENTA = "No tenés Notion conectado, así que no hay nada que desconectar."
DESCONECTAR_NOTION_PIDE_CONFIRMACION = (
    "Vas a desconectar tu Notion: dejo de poder ver las páginas que compartiste conmigo.\n\n"
    "Para confirmar, escribí: /desconectar notion confirmar"
)
DESCONECTAR_NOTION_LISTO = (
    "Listo, desconecté tu Notion y revoqué el permiso. "
    "Cuando quieras reconectarlo: /conectar notion"
)
DESCONECTAR_NOTION_SIN_REVOCAR = (
    "Borré tu Notion de mi lado, pero Notion no confirmó la revocación. Si querés "
    "estar seguro, quitá la conexión de LifeSync desde la configuración de Notion, "
    "en Conexiones."
)


def _texto_del_enlace_de_notion(enlace: str) -> str:
    return (
        "Para conectar tu Notion, entrá acá y autorizá el acceso:\n\n"
        f"{enlace}\n\n"
        "El enlace vence en 10 minutos. Notion te va a preguntar qué páginas compartir "
        "conmigo: elegí las que quieras que lea (las de adentro vienen incluidas). Sólo "
        "leo: no edito ni borro nada."
    )


def _texto_del_enlace(enlace: str) -> str:
    return (
        "Para conectar tu cuenta de Google, entrá acá y autorizá el acceso:\n\n"
        f"{enlace}\n\n"
        "El enlace vence en 10 minutos. Te voy a pedir permiso para tu calendario, "
        "tus tareas, tu correo y tu Drive; nunca borro nada, ni mando un correo, ni "
        "comparto un archivo sin confirmarte antes."
    )


class ManejadorDeComandos:
    """Resuelve los comandos que no pasan por el modelo."""

    def __init__(
        self, conectar_google: ConectarGoogle | None, conectar_notion: ConectarNotion | None = None
    ) -> None:
        """Recibe sus dependencias por constructor (inyección explícita).

        Cada caso de uso es `None` cuando falta la configuración de su OAuth. No
        se usa un *null object* como con el agente porque acá los dos casos
        dicen cosas distintas y ninguna es "responder igual pero peor". Sin
        Notion configurado, todo responde exactamente como antes de PB-014.
        """
        self._conectar_google = conectar_google
        self._conectar_notion = conectar_notion

    async def responder(self, texto: str, usuario: Usuario, ahora: datetime) -> str | None:
        """Devuelve la respuesta del comando, o None si no es un comando.

        `None` no significa "no entendí": significa "esto no me toca a mí".
        Devolverlo en vez de un texto de descarte es lo que permite que decida
        el agente y no este manejador.

        El comando se toma del primer token, así que "/Ayuda" y "/ayuda por
        favor" funcionan igual.
        """
        partes = texto.strip().lower().split()
        comando = partes[0] if partes else ""
        argumentos = partes[1:]

        if comando == "/ayuda":
            return AYUDA if self._conectar_notion is None else AYUDA_CON_NOTION
        if comando == "/estado":
            return await self._estado(usuario)
        if comando == "/conectar":
            # Sólo "notion" elige Notion: cualquier otra cosa sigue siendo
            # Google, como antes de PB-014.
            if argumentos[:1] == ["notion"]:
                return await self._vincular_notion(usuario, ahora)
            return await self._conectar(usuario, ahora)
        if comando == "/desconectar":
            if argumentos[:1] == ["notion"]:
                confirmado = " ".join(argumentos[1:]) == "confirmar"
                return await self._desconectar_notion(usuario, confirmado)
            return await self._desconectar(usuario, " ".join(argumentos) == "confirmar")
        return None

    async def _estado(self, usuario: Usuario) -> str:
        """Reporta qué integraciones tiene conectadas la persona (RF-12)."""
        google = await self._google_conectado(usuario)
        if self._conectar_notion is None or usuario.id is None:
            return ESTADO_CON_GOOGLE if google else ESTADO_SIN_CUENTAS

        notion = await self._conectar_notion.esta_conectado(usuario.id)
        if not google and not notion:
            return f"{ESTADO_SIN_CUENTAS}\n\n{PISTA_DE_NOTION}"
        bloque_de_google = ESTADO_CON_GOOGLE if google else GOOGLE_SIN_CONECTAR
        linea_de_notion = NOTION_CONECTADO if notion else NOTION_SIN_CONECTAR
        return f"{bloque_de_google}\n\n{linea_de_notion}"

    async def _google_conectado(self, usuario: Usuario) -> bool:
        if self._conectar_google is None or usuario.id is None:
            return False
        return await self._conectar_google.esta_conectado(usuario.id)

    async def _notion_conectado(self, usuario: Usuario) -> bool:
        if self._conectar_notion is None or usuario.id is None:
            return False
        return await self._conectar_notion.esta_conectado(usuario.id)

    async def _desconectar(self, usuario: Usuario, confirmado: bool) -> str:
        """Desconecta la cuenta de Google, con confirmación en dos pasos (RF-12).

        Es una acción destructiva, así que RF-08 aplica también acá — pero los
        comandos no pasan por el grafo, y el interrupt no existe. La
        confirmación vive en el texto: `/desconectar` sólo explica, y
        `/desconectar confirmar` ejecuta. Determinístico y sin memoria.
        """
        if self._conectar_google is None or usuario.id is None:
            return await self._sin_google_que_desconectar(usuario)

        if not await self._conectar_google.esta_conectado(usuario.id):
            return await self._sin_google_que_desconectar(usuario)

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

    async def _sin_google_que_desconectar(self, usuario: Usuario) -> str:
        """Con Notion conectado, "ninguna cuenta" sería falso: se dice cuál falta."""
        if await self._notion_conectado(usuario):
            return DESCONECTAR_GOOGLE_SIN_CUENTA
        return DESCONECTAR_SIN_CUENTA

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
        texto = _texto_del_enlace(enlace)
        return texto if self._conectar_notion is None else f"{texto}\n\n{PISTA_DE_NOTION}"

    async def _vincular_notion(self, usuario: Usuario, ahora: datetime) -> str:
        """Arma el enlace de consentimiento de Notion para esta persona (PB-014)."""
        if self._conectar_notion is None or usuario.id is None:
            return CONECTAR_NOTION_NO_DISPONIBLE

        try:
            enlace = self._conectar_notion.link_de_autorizacion(usuario.id, ahora)
        except Exception as exc:  # el comando nunca deja a la persona sin respuesta
            logger.error("notion.link_fallido", usuario_id=str(usuario.id), tipo=type(exc).__name__)
            return CONECTAR_ERROR

        logger.info("notion.link_entregado", usuario_id=str(usuario.id))
        return _texto_del_enlace_de_notion(enlace)

    async def _desconectar_notion(self, usuario: Usuario, confirmado: bool) -> str:
        """Desconecta Notion, con la misma confirmación en dos pasos que Google."""
        if self._conectar_notion is None or usuario.id is None:
            return DESCONECTAR_NOTION_SIN_CUENTA
        if not await self._conectar_notion.esta_conectado(usuario.id):
            return DESCONECTAR_NOTION_SIN_CUENTA
        if not confirmado:
            return DESCONECTAR_NOTION_PIDE_CONFIRMACION

        try:
            revocado = await self._conectar_notion.desconectar(usuario.id)
        except Exception as exc:  # el comando nunca deja a la persona sin respuesta
            logger.error(
                "notion.desconexion_fallida", usuario_id=str(usuario.id), tipo=type(exc).__name__
            )
            return "No pude desconectar tu Notion ahora mismo. Probá de nuevo en un minuto."

        return DESCONECTAR_NOTION_LISTO if revocado else DESCONECTAR_NOTION_SIN_REVOCAR
