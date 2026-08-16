"""Router mínimo de comandos (PB-004, base de RF-11).

Es el andamiaje que ocupa el lugar del agente hasta PB-005: reconoce un puñado
de comandos fijos y para todo lo demás responde con honestidad que todavía no
sabe interpretar lenguaje natural.

Es una **función pura**: sin estado, sin E/S, sin dependencias. Eso hace que se
pueda testear con una tabla de casos y sin un solo doble, y que PB-005 la
reemplace cambiando una línea del caso de uso.
"""

from __future__ import annotations

AYUDA = (
    "Soy LifeSync, tu asistente personal.\n\n"
    "Todavía estoy en construcción. Por ahora entiendo:\n"
    "• /ayuda — esta lista\n"
    "• /estado — qué cuentas tenés conectadas\n\n"
    "Muy pronto vas a poder pedirme cosas en lenguaje natural: agendar "
    "reuniones, crear recordatorios y consultar tu correo."
)

ESTADO = (
    "Todavía no tenés ninguna cuenta conectada.\n\n"
    "En la próxima entrega vas a poder vincular tu cuenta de Google para que "
    "pueda gestionar tu calendario y tu correo."
)

NO_ENTIENDO = (
    "Todavía no sé interpretar mensajes libres, pero ya te estoy escuchando.\n\n"
    "Escribí /ayuda para ver lo que puedo hacer hoy."
)


def decidir_respuesta(texto: str) -> str:
    """Devuelve el texto con el que hay que contestar un mensaje entrante.

    El comando se toma del primer token, así que "/Ayuda" y "/ayuda por favor"
    funcionan igual.
    """
    primer_token = texto.strip().lower().split(maxsplit=1)
    comando = primer_token[0] if primer_token else ""

    if comando == "/ayuda":
        return AYUDA
    if comando == "/estado":
        return ESTADO
    return NO_ENTIENDO
