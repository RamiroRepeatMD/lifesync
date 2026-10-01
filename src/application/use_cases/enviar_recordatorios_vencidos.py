"""Caso de uso: mandar los recordatorios cuya hora llegó (PB-030).

Es lo único del sistema que escribe sin que la persona haya escrito antes. No
lo dispara un webhook sino el reloj: un bucle periódico del proceso
(`src/interfaces/jobs/recordatorios.py`) lo ejecuta cada 30 segundos.

La ventana de 24 h de WhatsApp no se chequea acá: la garantiza la herramienta
que los crea, que sólo acepta avisos dentro de un día desde el "sí" de la
persona. Si igual se escapa uno, Meta lo rechaza con 131047 y queda `fallido`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from uuid import UUID

import structlog

from src.application.ports.whatsapp import MensajeroWhatsApp
from src.domain.entities.recordatorio import EstadoDeRecordatorio, Recordatorio
from src.domain.exceptions import MensajeRechazadoError
from src.domain.repositories.recordatorio_repository import RecordatorioRepository
from src.domain.repositories.usuario_repository import UsuarioRepository
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp

logger = structlog.get_logger(__name__)

# Cuántos se mandan por vuelta, como mucho. Con un bot personal sobra; el tope
# es para que una acumulación (el proceso estuvo caído) no sature a Meta.
LOTE = 20

# Pasado este atraso, un recordatorio que no pudo salir se da por perdido: un
# "sacá la pizza" de hace dos horas confunde más de lo que ayuda.
DEMORA_MAXIMA = timedelta(hours=1)

# Con más atraso que esto, el mensaje aclara para qué hora era.
DEMORA_A_AVISAR = timedelta(minutes=2)

_PENDIENTE = EstadoDeRecordatorio.PENDIENTE
_ENVIANDO = EstadoDeRecordatorio.ENVIANDO


class EnviarRecordatoriosVencidos:
    """Reclama los recordatorios vencidos, los manda y deja asentado el resultado."""

    def __init__(
        self,
        recordatorios: RecordatorioRepository,
        usuarios: UsuarioRepository,
        mensajero: MensajeroWhatsApp,
        zona: tzinfo,
    ) -> None:
        """Recibe sus dependencias por constructor (inyección explícita).

        `zona` es la de la persona, para escribir la hora en el aviso de
        demora. Llega de afuera porque la configuración es infraestructura.
        """
        self._recordatorios = recordatorios
        self._usuarios = usuarios
        self._mensajero = mensajero
        self._zona = zona

    async def ejecutar(self, ahora: datetime) -> int:
        """Manda los vencidos a `ahora`. Devuelve cuántos salieron.

        Cada recordatorio va por su lado: si uno falla, los demás salen igual.
        """
        enviados = 0
        for recordatorio in await self._recordatorios.vencidos(ahora, LOTE):
            if recordatorio.id is None:
                continue  # invariante rota: lo que viene de la base trae id
            try:
                if await self._enviar(recordatorio, recordatorio.id, ahora):
                    enviados += 1
            except Exception as exc:  # aislar: uno roto no frena a los demás
                logger.error(
                    "recordatorio.error",
                    recordatorio_id=str(recordatorio.id),
                    tipo=type(exc).__name__,
                )
        return enviados

    async def _enviar(self, recordatorio: Recordatorio, rid: UUID, ahora: datetime) -> bool:
        """Reclamar → mandar → asentar. True si salió."""
        # Si otro despachador lo reclamó primero (deploy con dos contenedores),
        # es suyo: mandarlo de nuevo sería un aviso duplicado.
        if not await self._recordatorios.cambiar_estado(rid, _PENDIENTE, _ENVIANDO):
            logger.info("recordatorio.ya_reclamado", recordatorio_id=str(rid))
            return False

        demora = ahora - recordatorio.momento
        try:
            usuario = await self._usuarios.obtener_por_id(recordatorio.usuario_id)
            if usuario is None:
                await self._fallar(rid, motivo="usuario_inexistente")
                return False
            await self._mensajero.enviar_texto(
                NumeroWhatsApp(usuario.telefono_whatsapp), self._texto(recordatorio, demora)
            )
        except MensajeRechazadoError:
            # Fuera de la ventana de 24 h, token de Meta vencido, número no
            # autorizado: reintentar daría lo mismo.
            await self._fallar(rid, motivo="rechazado_por_whatsapp")
            return False
        except Exception as exc:  # corte de red, Meta o la base caídos: transitorio
            if demora > DEMORA_MAXIMA:
                await self._fallar(rid, motivo="demora_excesiva")
            else:
                await self._recordatorios.cambiar_estado(rid, _ENVIANDO, _PENDIENTE)
                logger.warning(
                    "recordatorio.reintento", recordatorio_id=str(rid), tipo=type(exc).__name__
                )
            return False

        await self._recordatorios.cambiar_estado(rid, _ENVIANDO, EstadoDeRecordatorio.ENVIADO)
        logger.info(
            "recordatorio.enviado",
            recordatorio_id=str(rid),
            demora_s=int(demora.total_seconds()),
        )
        return True

    async def _fallar(self, rid: UUID, *, motivo: str) -> None:
        await self._recordatorios.cambiar_estado(rid, _ENVIANDO, EstadoDeRecordatorio.FALLIDO)
        logger.warning("recordatorio.fallido", recordatorio_id=str(rid), motivo=motivo)

    def _texto(self, recordatorio: Recordatorio, demora: timedelta) -> str:
        texto = f"⏰ Recordatorio: {recordatorio.texto}"
        if demora > DEMORA_A_AVISAR:
            hora = recordatorio.momento.astimezone(self._zona)
            texto += f"\n(Era para las {hora:%H:%M}: se demoró el envío.)"
        return texto
