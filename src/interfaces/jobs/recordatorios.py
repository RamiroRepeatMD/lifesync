"""El despachador de recordatorios: un bucle que corre dentro del proceso (PB-030).

Una tarea de `asyncio` que arranca en el `lifespan` y no una dependencia nueva
(APScheduler, un cron de Railway): el servicio corre una sola instancia con un
solo worker (ver docs/CLAUDE.md §5), y lo pendiente vive en la base, así que un
redeploy no pierde nada — el contenedor nuevo retoma en su primera vuelta.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import structlog

from src.application.use_cases.enviar_recordatorios_vencidos import EnviarRecordatoriosVencidos

logger = structlog.get_logger(__name__)

# La precisión del aviso: sale entre 0 y 30 s después de la hora pedida.
INTERVALO_SEGUNDOS = 30.0


async def despachar_recordatorios(
    caso: EnviarRecordatoriosVencidos, intervalo: float = INTERVALO_SEGUNDOS
) -> None:
    """Corre hasta que lo cancelen: cada `intervalo`, manda los vencidos.

    Cada vuelta atrapa todo. Si una base que parpadea matara el bucle, nadie
    lo volvería a levantar hasta el próximo deploy, y los recordatorios
    dejarían de salir en silencio. La cancelación del shutdown sí sale:
    `CancelledError` no es una `Exception`.
    """
    logger.info("recordatorios.despachador_iniciado", intervalo_s=intervalo)
    while True:
        try:
            await caso.ejecutar(datetime.now(UTC))
        except Exception as exc:  # una vuelta rota no puede matar el bucle
            logger.error("recordatorios.vuelta_fallida", tipo=type(exc).__name__)
        await asyncio.sleep(intervalo)
