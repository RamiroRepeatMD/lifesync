"""Deduplicación de mensajes entrantes de WhatsApp (PB-004).

Meta entrega los webhooks **al menos una vez**: reintenta ante cualquier fallo o
demora, durante horas. Sin deduplicar, un reintento produce una segunda
respuesta al usuario por el mismo mensaje.

La deduplicación es en memoria del proceso, con tope duro y vencimiento. Es una
decisión consciente para el MVP: no sobrevive a un reinicio ni sirve con varias
instancias. La red de contención para el reintento viejo es la ventana de
frescura del parser, que descarta mensajes con timestamp antiguo aunque el
caché esté vacío. La versión persistente queda para el Sprint 2.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import structlog

logger = structlog.get_logger(__name__)

CAPACIDAD_POR_DEFECTO = 512
TTL_POR_DEFECTO_SEGUNDOS = 6 * 60 * 60  # 6 h


class DeduplicadorDeMensajes:
    """Recuerda qué `wamid` ya se procesaron, con tope y vencimiento.

    El tope existe para que el caché no sea un vector de agotamiento de
    memoria: un atacante con la firma válida —o Meta en un pico— no puede
    hacerlo crecer sin límite. Al llenarse se desaloja el más viejo (LRU).
    """

    def __init__(
        self,
        capacidad: int = CAPACIDAD_POR_DEFECTO,
        ttl_segundos: float = TTL_POR_DEFECTO_SEGUNDOS,
    ) -> None:
        """Recibe sus parámetros por constructor (inyección explícita)."""
        self._capacidad = capacidad
        self._ttl = ttl_segundos
        self._vistos: OrderedDict[str, float] = OrderedDict()

    def marcar_si_es_nuevo(self, wamid: str) -> bool:
        """Registra el mensaje y dice si es la primera vez que se ve.

        Marca **antes** de procesar, no después: si se marcara al terminar,
        dos entregas solapadas del mismo mensaje pasarían las dos por la
        comprobación antes de que ninguna registre nada.

        Returns:
            True si hay que procesarlo; False si es un duplicado.
        """
        ahora = time.monotonic()
        self._purgar(ahora)

        if wamid in self._vistos:
            self._vistos.move_to_end(wamid)
            return False

        self._vistos[wamid] = ahora
        while len(self._vistos) > self._capacidad:
            self._vistos.popitem(last=False)
            logger.debug("whatsapp.dedup.desalojado", cantidad=len(self._vistos))

        return True

    def olvidar(self, wamid: str) -> None:
        """Borra la marca de un mensaje.

        Se llama cuando el procesamiento falló, para que el reintento de Meta
        tenga otra oportunidad: si no, un fallo transitorio de la base dejaría
        ese mensaje sin respuesta para siempre.
        """
        self._vistos.pop(wamid, None)

    def _purgar(self, ahora: float) -> None:
        """Saca las entradas vencidas. El OrderedDict está por antigüedad."""
        limite = ahora - self._ttl
        while self._vistos:
            wamid, visto_en = next(iter(self._vistos.items()))
            if visto_en > limite:
                return
            self._vistos.pop(wamid, None)

    def __len__(self) -> int:
        """Cantidad de mensajes recordados. Útil para los tests."""
        return len(self._vistos)
