"""Comandos de respuesta fija (RF-11).

Con PB-005 el lenguaje natural pasó a manejarlo el agente, pero los comandos
siguen siendo determinísticos, y es una decisión, no un resto del andamiaje
anterior:

- **RF-11 pide un sistema de ayuda.** Si `/ayuda` dependiera del modelo, la
  ayuda cambiaría de texto en cada invocación y podría inventar capacidades que
  el sistema no tiene.
- **Sigue funcionando sin API key.** Es lo único que contesta igual en modo
  degradado.
- No gasta tokens ni latencia en algo cuya respuesta ya conocemos.

Es una **función pura**: sin estado, sin E/S, sin dependencias. Se testea con
una tabla de casos y sin un solo doble.
"""

from __future__ import annotations

AYUDA = (
    "Soy LifeSync, tu asistente personal.\n\n"
    "Escribime en lenguaje natural y hago lo que pueda: preguntame la hora, "
    "pedime que te ayude a organizarte o contame qué necesitás.\n\n"
    "Comandos fijos:\n"
    "• /ayuda — esta lista\n"
    "• /estado — qué cuentas tenés conectadas\n\n"
    "Todavía estoy en construcción: cuando conectes tu cuenta de Google voy a "
    "poder gestionar tu calendario, tus tareas y tu correo."
)

ESTADO = (
    "Todavía no tenés ninguna cuenta conectada.\n\n"
    "En la próxima entrega vas a poder vincular tu cuenta de Google para que "
    "pueda gestionar tu calendario y tu correo."
)

_RESPUESTAS: dict[str, str] = {
    "/ayuda": AYUDA,
    "/estado": ESTADO,
}


def respuesta_fija(texto: str) -> str | None:
    """Devuelve la respuesta canónica de un comando, o None si no es un comando.

    `None` significa "esto es lenguaje natural": el caso de uso se lo pasa al
    agente. Devolver None en vez de un texto de descarte es lo que permite que
    el agente sea el que decide, y no este router.

    El comando se toma del primer token, así que "/Ayuda" y "/ayuda por favor"
    funcionan igual.
    """
    primer_token = texto.strip().lower().split(maxsplit=1)
    if not primer_token:
        return None
    return _RESPUESTAS.get(primer_token[0])
