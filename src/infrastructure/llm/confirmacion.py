"""Clasificación determinística de la respuesta a una confirmación (PB-016, RF-08).

Cuando hay una acción pendiente de confirmar, la decisión de ejecutarla **no
puede ser del modelo**: el modelo interpreta, y una interpretación generosa de
un "mmm bueno" no es un consentimiento. Acá se clasifica con reglas puras:

- **Sólo un sí explícito aprueba.** Match exacto contra una lista corta, sobre
  el mensaje normalizado. Conservador a propósito.
- **Todo lo demás cancela.** Un "no", una pregunta, un cambio de tema: nadie
  crea ni borra nada por un mensaje tibio. La diferencia entre la negación
  explícita y "otra cosa" es sólo qué se hace después: la negación termina el
  turno; otra cosa además se procesa como mensaje nuevo.
"""

from __future__ import annotations

import unicodedata
from enum import Enum

# Cuánto vive una confirmación pendiente. Pasado esto se cancela sola: aprobar
# con un "sí" algo que se propuso hace media hora, sin volver a mostrarlo, es
# ejecutar una acción que la persona quizás ya ni recuerda.
VIGENCIA_DE_CONFIRMACION_SEGUNDOS = 10 * 60


class Decision(Enum):
    """Qué expresó la persona frente a la propuesta pendiente."""

    APRUEBA = "aprueba"
    RECHAZA = "rechaza"
    OTRA_COSA = "otra_cosa"


# Match EXACTO sobre el texto normalizado, nunca substring: "si me parece raro"
# no es un sí, y "nova a andar" no contiene un "no" que valga como rechazo.
_AFIRMACIONES = frozenset(
    {
        "si",
        "sí",
        "dale",
        "ok",
        "okey",
        "okay",
        "confirmo",
        "confirmar",
        "confirmado",
        "de una",
        "obvio",
        "si dale",
        "sí dale",
        "dale si",
        "listo",
        "hacelo",
        "mandale",
        "afirmativo",
        "yes",
        "s",
    }
)

_NEGACIONES = frozenset(
    {
        "no",
        "nop",
        "cancelar",
        "cancela",
        "cancelalo",
        "no gracias",
        "mejor no",
        "dejalo",
        "dejálo",
        "olvidalo",
        "olvidálo",
        "negativo",
        "n",
        "para",
        "pará",
    }
)


def clasificar(texto: str) -> Decision:
    """Clasifica la respuesta de la persona a una confirmación pendiente."""
    normalizado = _normalizar(texto)
    if normalizado in {_normalizar(a) for a in _AFIRMACIONES}:
        return Decision.APRUEBA
    if normalizado in {_normalizar(n) for n in _NEGACIONES}:
        return Decision.RECHAZA
    return Decision.OTRA_COSA


_PUNTUACION = str.maketrans(dict.fromkeys(".,;:!?¡¿…", " "))


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, sin puntuación, espacios simples.

    La puntuación se reemplaza en todo el texto y no sólo en los bordes: un
    "No, gracias" tiene que clasificar igual que "no gracias".
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    return " ".join(sin_tildes.translate(_PUNTUACION).split())
