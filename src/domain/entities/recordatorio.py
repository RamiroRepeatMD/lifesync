"""El recordatorio que la persona pide recibir por WhatsApp (PB-030).

Es lo único del sistema que sale sin que nadie haya escrito antes: a la hora
pedida, LifeSync le escribe a la persona. Por eso tiene un estado propio —el
despachador tiene que saber qué ya salió— y el texto no se guarda en claro:
lo cifra el adaptador, porque es contenido de la persona (RF-18).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from src.domain.exceptions import InvalidValueError

MAX_CARACTERES_DE_RECORDATORIO = 300


class EstadoDeRecordatorio(StrEnum):
    """El ciclo de vida: pendiente → enviando → enviado | fallido, o pendiente → cancelado.

    `enviando` es el reclamo. Durante un deploy, Railway superpone un rato el
    contenedor viejo y el nuevo: hay dos despachadores. El que pasa primero el
    recordatorio de `pendiente` a `enviando` lo manda; el otro lo saltea.
    """

    PENDIENTE = "pendiente"
    ENVIANDO = "enviando"
    ENVIADO = "enviado"
    CANCELADO = "cancelado"
    FALLIDO = "fallido"


@dataclass(frozen=True, slots=True)
class Recordatorio:
    """Un aviso que LifeSync le manda a la persona en un momento dado.

    Attributes:
        usuario_id: De quién es, y a quién se le manda.
        texto: Qué hay que recordarle. Fuera del repr: es contenido de la persona.
        momento: Cuándo mandarlo. Siempre con zona horaria.
        estado: En qué punto del ciclo está.
        id: Asignado por la base al persistir; None mientras no exista.
    """

    usuario_id: UUID
    texto: str = field(repr=False)
    momento: datetime
    estado: EstadoDeRecordatorio = EstadoDeRecordatorio.PENDIENTE
    id: UUID | None = None

    def __post_init__(self) -> None:
        """Valida las invariantes de la entidad al construirla."""
        if not self.texto.strip():
            raise InvalidValueError("El recordatorio necesita un texto.")
        if len(self.texto) > MAX_CARACTERES_DE_RECORDATORIO:
            raise InvalidValueError(
                f"El recordatorio no puede pasar de {MAX_CARACTERES_DE_RECORDATORIO} caracteres."
            )
        # Un momento sin zona es ambiguo: las 21:55 de dónde. El despachador
        # compara contra el reloj en UTC y tiene que poder hacerlo sin adivinar.
        if self.momento.tzinfo is None:
            raise InvalidValueError("El momento del recordatorio necesita zona horaria.")
