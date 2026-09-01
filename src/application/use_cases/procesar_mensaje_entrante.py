"""Caso de uso: procesar un mensaje que llegó por WhatsApp (PB-004 · PB-005).

Es el corazón del flujo conversacional descrito en la arquitectura: resolver
quién escribe, decidir qué contestar y contestar.

Con PB-005 el "decidir qué contestar" pasó a ser del agente, salvo los comandos
fijos. Lo que falta del flujo —confirmación de acciones críticas y herramientas
que escriban (RF-08)— entra con las primeras integraciones reales.
"""

from __future__ import annotations

import structlog

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.dto.mensaje_entrante import MensajeEntrante
from src.application.ports.agente import AgenteConversacional
from src.application.ports.whatsapp import MensajeroWhatsApp
from src.application.services.manejador_de_comandos import ManejadorDeComandos
from src.domain.entities.usuario import Usuario
from src.domain.exceptions import RepositoryError
from src.domain.repositories.usuario_repository import UsuarioRepository
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp

logger = structlog.get_logger(__name__)


class ProcesarMensajeEntrante:
    """Resuelve el usuario, decide la respuesta y la envía."""

    def __init__(
        self,
        usuarios: UsuarioRepository,
        mensajero: MensajeroWhatsApp,
        agente: AgenteConversacional,
        comandos: ManejadorDeComandos,
    ) -> None:
        """Recibe sus dependencias por constructor (inyección explícita).

        El agente nunca es `None`: cuando falta la API key, el composition root
        entrega un `AgenteDegradado`. Así no hay una rama de configuración
        acá adentro, que no es asunto de este caso de uso.
        """
        self._usuarios = usuarios
        self._mensajero = mensajero
        self._agente = agente
        self._comandos = comandos

    async def ejecutar(self, mensaje: MensajeEntrante) -> None:
        """Procesa un mensaje entrante de punta a punta.

        No captura errores: de eso se encarga el borde del webhook, que es
        quien sabe si conviene reintentar y quién le avisa al usuario.
        """
        usuario = await self._usuarios.obtener_o_crear(
            telefono=mensaje.remitente.valor,
            nombre=mensaje.nombre_perfil,
        )
        if usuario.id is None:
            # Invariante rota, no un caso normal: el repositorio devuelve la
            # fila persistida y ésa siempre trae id.
            raise RepositoryError("El usuario persistido volvió sin id.")

        # Nunca el texto ni el teléfono: el id del usuario alcanza para
        # correlacionar y no es dato personal (RF-18).
        logger.info(
            "whatsapp.mensaje.procesando",
            usuario_id=str(usuario.id),
            wamid=mensaje.wamid,
            largo_texto=len(mensaje.texto),
        )

        respuesta = await self._decidir_respuesta(mensaje, usuario)
        await self._mensajero.enviar_texto(mensaje.remitente, respuesta)

    async def _decidir_respuesta(self, mensaje: MensajeEntrante, usuario: Usuario) -> str:
        """Comando si lo hay; si no, que lo piense el agente."""
        del_comando = await self._comandos.responder(mensaje.texto, usuario, mensaje.enviado_en)
        if del_comando is not None:
            logger.info("whatsapp.respuesta_de_comando", wamid=mensaje.wamid)
            return del_comando

        # mypy: `ejecutar` ya descartó el id None antes de llegar acá.
        assert usuario.id is not None  # noqa: S101
        return await self._agente.responder(
            ConsultaDelUsuario(
                conversacion_id=usuario.id,
                texto=mensaje.texto,
                nombre_usuario=usuario.nombre,
            )
        )

    async def avisar(self, destino: NumeroWhatsApp, texto: str) -> None:
        """Le manda un aviso al usuario sin pasar por el flujo completo.

        Lo usa el borde del webhook para contarle que algo falló (RF-19),
        cuando `ejecutar` ya se rompió y no tiene sentido volver a resolver el
        usuario ni a decidir una respuesta.
        """
        await self._mensajero.enviar_texto(destino, texto)
