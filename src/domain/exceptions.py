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


class CuentaNoConectadaError(DomainError):
    """Se pidió algo que necesita una cuenta que la persona no conectó (PB-015).

    No es una falla técnica ni un error de la persona: es un paso que falta.
    Por eso hereda de `DomainError` y no de `InfrastructureError`, y por eso su
    mensaje dice **qué hacer** en vez de disculparse.
    """

    mensaje_usuario = (
        "Todavía no conectaste tu cuenta de Google. Escribí /conectar y te paso el enlace."
    )


class NotionNoConectadoError(CuentaNoConectadaError):
    """Se pidió algo de Notion y la persona no lo conectó (PB-014).

    Hija de la de Google para que todo lo que ya atrapa "falta una cuenta" la
    atrape también; el remedio es otro comando, así que el mensaje es propio.
    """

    mensaje_usuario = (
        "Todavía no conectaste tu Notion. Escribí /conectar notion y te paso el enlace."
    )


class ArchivoNoDisponibleError(EntityNotFoundError):
    """El archivo que mandó la persona ya no se puede bajar del canal (PB-038).

    WhatsApp guarda lo recibido unos días (7, según su documentación): pasado
    ese plazo, o si el id no existe, no hay de dónde sacarlo. El remedio lo
    tiene la persona: mandarlo de nuevo.
    """

    mensaje_usuario = "Ese archivo ya no está disponible en WhatsApp: mandámelo de nuevo y lo subo."


class ArchivoDemasiadoGrandeError(InvalidValueError):
    """El archivo pesa más de lo que se sube desde el chat (PB-038).

    El tope es nuestro, no de Drive: el archivo pasa por la memoria del
    proceso entre la descarga y la subida.
    """

    mensaje_usuario = (
        "Ese archivo es demasiado grande para subirlo desde acá: subilo desde la app de Drive."
    )


class DriveRechazoError(DomainError):
    """Drive no dejó hacer lo pedido sobre un archivo (PB-038).

    Cubre lo que reconectar no arregla: la persona no tiene permiso sobre ese
    archivo, la cuenta (de una organización) no deja compartir afuera, o Drive
    rechazó la dirección. Un 400 de Drive al compartir a veces llega aunque el
    archivo SÍ quedó compartido (no pudo mandar el aviso): por eso el mensaje
    no afirma que no pasó, y manda a mirar Drive antes de repetirlo.
    """

    mensaje_usuario = (
        "Drive no confirmó que se haya hecho: puede que no tengas permiso sobre ese "
        "archivo o que tu cuenta no lo permita. Fijate en Drive antes de pedírmelo de nuevo."
    )


class DriveLlenoError(DomainError):
    """No hay espacio en el Drive de la persona para subir el archivo (PB-038)."""

    mensaje_usuario = (
        "Tu Google Drive está lleno, así que no pude subirlo. Liberá espacio y pedímelo de nuevo."
    )


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


class MensajeRechazadoError(MensajeNoEnviadoError):
    """El canal rechazó el envío por un motivo que no se arregla reintentando.

    El caso que la motivó (PB-030): pasaron más de 24 h desde el último
    mensaje de la persona y WhatsApp sólo acepta plantillas aprobadas (131047).
    El despachador de recordatorios la distingue de un corte de red: un corte
    se reintenta; esto no, porque el resultado sería el mismo.
    """


class ServiceUnavailableError(InfrastructureError):
    """Un servicio del que dependemos no está configurado o no responde."""

    mensaje_usuario = "El servicio no está disponible por ahora. Probá más tarde."


class EnvioInciertoError(ServiceUnavailableError):
    """El proveedor no confirmó un envío que pudo haber salido (PB-035).

    El pedido llegó y la respuesta no (se cortó, venció el tiempo) o el
    proveedor falló a mitad de camino (un 5xx). Se separa de su madre porque el
    remedio es el opuesto: un envío que seguro no salió se puede volver a
    pedir; éste no se reintenta a ciegas, porque podría salir dos veces y un
    correo no se des-envía. Hay que mirar Enviados antes.
    """

    mensaje_usuario = (
        "No sé si el correo salió. Revisá tu carpeta Enviados antes de volver a pedirlo."
    )


class EnvioRechazadoError(ServiceUnavailableError):
    """El proveedor rechazó el envío por lo que se le mandó (PB-035).

    Un 400 de Gmail: una dirección que no acepta, un dato que no valida. El
    correo no salió, y a diferencia de un corte o de un límite de uso,
    reintentar igual da lo mismo: hay que corregir el pedido.
    """

    mensaje_usuario = "Gmail rechazó el correo, así que no salió. Revisá las direcciones."


class AgenteNoDisponibleError(InfrastructureError):
    """El agente conversacional no pudo producir una respuesta (PB-005).

    Cubre todo lo que puede salir mal del lado del modelo: que no responda a
    tiempo, que rechace la petición, que devuelva vacío. Se distingue de
    `MensajeNoEnviadoError` porque el canal de WhatsApp sigue sano: el aviso de
    RF-19 le llega igual al usuario.
    """

    mensaje_usuario = "No pude pensar una respuesta ahora mismo. Probá de nuevo en un minuto."


class CuotaDeAgenteAgotadaError(AgenteNoDisponibleError):
    """Se acabó la cuota de la API del modelo (PB-005).

    Se separa de su madre porque el usuario puede hacer algo al respecto:
    esperar. Decirle "hubo un problema técnico" cuando en realidad hay que
    aguantar un minuto lo deja probando una y otra vez, y cada intento gasta
    más cuota.
    """

    mensaje_usuario = "Estoy al límite de consultas por ahora. Probá de nuevo en un minuto."


class AutorizacionFallidaError(InfrastructureError):
    """No se pudo completar la autorización con un proveedor externo (PB-009).

    Cubre lo que puede fallar del lado del proveedor: que rechace el canje del
    código, que no devuelva credenciales o que se caiga la renovación. No cubre
    un `state` inválido —eso es `InvalidValueError`, porque el problema está en
    lo que llegó, no en el proveedor.
    """

    mensaje_usuario = "No pude completar la conexión con tu cuenta. Probá de nuevo."


class PermisoInsuficienteError(AutorizacionFallidaError):
    """La credencial es válida pero no alcanza para la operación (PB-016).

    El caso concreto: una cuenta conectada antes de PB-016 tiene sólo el scope
    de lectura, y la primera escritura devuelve 403. Se separa de su madre
    porque el remedio es distinto y la persona puede ejecutarlo: volver a
    autorizar con /conectar, que ahora pide también el permiso de escritura.
    """

    mensaje_usuario = (
        "Me falta un permiso de tu cuenta de Google. Escribí /conectar para "
        "actualizar la autorización."
    )


class EncryptionError(InfrastructureError):
    """No se pudo cifrar o descifrar un dato sensible (RF-18).

    Suele significar que `TOKEN_ENCRYPTION_KEY` es inválida o que cambió
    respecto de la clave con la que se cifró el dato guardado.
    """

    mensaje_usuario = "Hubo un problema al proteger tus credenciales."
