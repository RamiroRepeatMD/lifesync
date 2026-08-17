"""Errores del dominio de LifeSync.

Son independientes de cualquier framework: no saben qué es un código HTTP.
La traducción a respuestas HTTP la hace `src/interfaces/api/errors.py` (RF-19).

Cada error lleva un `mensaje_usuario` pensado para mostrarse tal cual por
WhatsApp: en español, sin jerga técnica y sin filtrar detalles internos.
"""

from __future__ import annotations


class LifeSyncError(Exception):
    """Error base de la aplicación."""

    mensaje_usuario: str = "No pudimos completar la acción solicitada."

    def __init__(self, detalle: str | None = None) -> None:
        self.detalle = detalle or self.mensaje_usuario
        super().__init__(self.detalle)

    @property
    def codigo(self) -> str:
        """Identificador estable del error, útil para logs y clientes."""
        return type(self).__name__


class DomainError(LifeSyncError):
    """Violación de una regla de negocio."""

    mensaje_usuario = "Esa acción no es válida en este momento."


class EntityNotFoundError(DomainError):
    """No existe la entidad solicitada."""

    mensaje_usuario = "No encontré lo que estás buscando."


class InvalidValueError(DomainError):
    """Un value object recibió un valor que no cumple sus invariantes."""

    mensaje_usuario = "Alguno de los datos ingresados no es válido."


class ConfirmationRequiredError(DomainError):
    """Se intentó ejecutar una acción crítica sin confirmación explícita (RF-08).

    La política que decide qué acciones son críticas vive en
    `src/domain/services/`; la enforcement en el grafo llega con PB-005.
    """

    mensaje_usuario = "Necesito que me confirmes antes de hacer este cambio."


class InfrastructureError(LifeSyncError):
    """Falla técnica al hablar con un sistema externo (base de datos, APIs).

    Es hermana de `DomainError`, no hija: el usuario no hizo nada mal, se rompió
    algo nuestro. Vive en el dominio porque las interfaces de repositorio la
    declaran en sus `Raises:`, pero no conoce ninguna tecnología concreta.
    """

    mensaje_usuario = "Estamos con un problema técnico. Probá de nuevo en unos minutos."


class RepositoryError(InfrastructureError):
    """La operación contra el almacén de datos falló."""

    mensaje_usuario = "No pudimos guardar o recuperar tu información en este momento."


class MensajeNoEnviadoError(InfrastructureError):
    """No se pudo entregar un mensaje al usuario por su canal.

    Se distingue de `RepositoryError` a propósito: cuando falla el canal, no
    tiene sentido intentar avisarle al usuario *por ese mismo canal*. Si en
    cambio lo que falló fue la base, el canal sigue sano y el aviso llega.
    """

    mensaje_usuario = "No pude responderte en este momento."


class ServiceUnavailableError(InfrastructureError):
    """Un servicio del que dependemos no está configurado o no responde."""

    mensaje_usuario = "El servicio no está disponible por ahora. Probá más tarde."


class EncryptionError(InfrastructureError):
    """No se pudo cifrar o descifrar un dato sensible (RF-18).

    Suele significar que `TOKEN_ENCRYPTION_KEY` es inválida o que cambió
    respecto de la clave con la que se cifró el dato guardado.
    """

    mensaje_usuario = "Hubo un problema al proteger tus credenciales."
