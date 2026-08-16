"""Caso de uso: procesar un mensaje que llegó por WhatsApp (PB-004).

Es el corazón del flujo conversacional descrito en la arquitectura, en su
versión mínima: resolver quién escribe, decidir qué contestar y contestar.

Los pasos 3 a 6 de ese flujo (contexto, LangGraph, confirmación, herramientas)
entran en PB-005 reemplazando la llamada a `decidir_respuesta`.
"""

from __future__ import annotations

import structlog

from src.application.dto.mensaje_entrante import MensajeEntrante
from src.application.ports.whatsapp import MensajeroWhatsApp
from src.application.services.router_de_comandos import decidir_respuesta
from src.domain.repositories.usuario_repository import UsuarioRepository
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp

logger = structlog.get_logger(__name__)


class ProcesarMensajeEntrante:
    """Resuelve el usuario, decide la respuesta y la envía."""

    def __init__(
        self,
        usuarios: UsuarioRepository,
        mensajero: MensajeroWhatsApp,
    ) -> None:
        """Recibe sus dependencias por constructor (inyección explícita)."""
        self._usuarios = usuarios
        self._mensajero = mensajero

    async def ejecutar(self, mensaje: MensajeEntrante) -> None:
        """Procesa un mensaje entrante de punta a punta.

        No captura errores: de eso se encarga el borde del webhook, que es
        quien sabe si conviene reintentar y quién le avisa al usuario.
        """
        usuario = await self._usuarios.obtener_o_crear(
            telefono=mensaje.remitente.valor,
            nombre=mensaje.nombre_perfil,
        )
        # Nunca el texto ni el teléfono: el id del usuario alcanza para
        # correlacionar y no es dato personal (RF-18).
        logger.info(
            "whatsapp.mensaje.procesando",
            usuario_id=str(usuario.id),
            wamid=mensaje.wamid,
            largo_texto=len(mensaje.texto),
        )

        respuesta = decidir_respuesta(mensaje.texto)
        await self._mensajero.enviar_texto(mensaje.remitente, respuesta)

    async def avisar(self, destino: NumeroWhatsApp, texto: str) -> None:
        """Le manda un aviso al usuario sin pasar por el flujo completo.

        Lo usa el borde del webhook para contarle que algo falló (RF-19),
        cuando `ejecutar` ya se rompió y no tiene sentido volver a resolver el
        usuario ni a decidir una respuesta.
        """
        await self._mensajero.enviar_texto(destino, texto)
