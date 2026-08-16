"""Traducción del payload de un webhook de Meta a DTOs propios (PB-004).

Es la pieza que absorbe la forma real de la API de Meta, con todas sus
peculiaridades:

- `entry[]`, `changes[]` y `messages[]` son arrays: un solo webhook puede traer
  hasta 1000 actualizaciones. Devolver "el mensaje" en singular perdería el
  resto en silencio, y Meta no reintentaría porque ya recibió 200.
- El campo `field` vale `"messages"` tanto para mensajes entrantes como para
  acuses de entrega. Hay que ramificar por la presencia de `value.messages`
  frente a `value.statuses` o `value.errors`.
- `contacts[]` no viene en los acuses, y cuando viene hay que emparejarlo por
  `wa_id` y no por posición.

Dos reglas de diseño de este módulo:

1. **Nunca lanza.** Un payload con forma inesperada tiene que producir un log y
   un 200, no un 500 que dispare reintentos de Meta durante horas.
2. **Nunca loguea.** Cuenta y clasifica; el handler emite un único evento con
   los conteos. Así el parser es una función pura y se testea sin capturar logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.application.dto.mensaje_entrante import MensajeEntrante
from src.domain.exceptions import InvalidValueError
from src.domain.value_objects.numero_whatsapp import NumeroWhatsApp

# Un cuerpo firmado por Meta es válido para siempre: la firma no lleva timestamp
# ni nonce. Como el caché de deduplicación se vacía en cada despliegue, se
# descartan los mensajes viejos para acotar la ventana de reproducción.
FRESCURA_MAXIMA = timedelta(hours=12)

# Tope de mensajes atendidos por request. Meta puede mandar hasta 1000, y cada
# uno encadena una llamada a Graph: sin tope, un solo webhook quema el rate limit.
MAXIMO_MENSAJES = 50

TIPO_TEXTO = "text"


@dataclass(frozen=True, slots=True)
class ContenidoWebhook:
    """Lo que se pudo extraer de un webhook, ya clasificado.

    Los contadores existen para que el handler pueda emitir una sola línea de
    log con lo que pasó, sin que el parser tenga que loguear nada.
    """

    mensajes: tuple[MensajeEntrante, ...] = ()
    acuses: int = 0
    errores_de_meta: int = 0
    no_soportados: int = 0
    vencidos: int = 0
    ignorados: int = 0
    truncados: int = 0
    sin_remitente: int = 0
    ajenos: int = 0
    claves_desconocidas: tuple[str, ...] = field(default=())

    @property
    def hay_trabajo(self) -> bool:
        """True si hay al menos un mensaje que procesar."""
        return bool(self.mensajes)


def parsear(
    payload: object,
    *,
    phone_number_id_propio: str | None,
    ahora: datetime | None = None,
) -> ContenidoWebhook:
    """Extrae los mensajes de texto de un payload de webhook.

    Args:
        payload: El JSON ya deserializado. Se acepta `object` porque viene de
            afuera y no hay garantía de que sea un dict.
        phone_number_id_propio: Nuestro número. Los eventos dirigidos a otro
            número se descartan: la firma prueba que lo mandó Meta, no que sea
            para nosotros.
        ahora: Momento de referencia para la ventana de frescura.
    """
    acumulador = _Acumulador(
        phone_number_id_propio=phone_number_id_propio,
        ahora=ahora or datetime.now(UTC),
    )

    if not isinstance(payload, dict):
        return acumulador.resultado(ignorados_extra=1)

    for entrada in _lista(payload.get("entry")):
        for cambio in _lista(entrada.get("changes")):
            valor = cambio.get("value")
            if isinstance(valor, dict):
                acumulador.procesar_valor(valor)
            else:
                acumulador.ignorados += 1

    return acumulador.resultado()


class _Acumulador:
    """Estado mutable del parseo. Encapsulado para que `parsear` sea pura."""

    def __init__(self, *, phone_number_id_propio: str | None, ahora: datetime) -> None:
        self._phone_number_id_propio = phone_number_id_propio
        self._ahora = ahora
        self.mensajes: list[MensajeEntrante] = []
        self.acuses = 0
        self.errores_de_meta = 0
        self.no_soportados = 0
        self.vencidos = 0
        self.ignorados = 0
        self.truncados = 0
        self.sin_remitente = 0
        self.ajenos = 0
        self.claves_desconocidas: set[str] = set()

    def procesar_valor(self, valor: dict[str, Any]) -> None:
        """Clasifica un `changes[].value` y extrae lo que corresponda."""
        if not self._es_para_nosotros(valor):
            self.ajenos += 1
            return

        if "messages" in valor:
            self._procesar_mensajes(valor)
        elif "statuses" in valor:
            self.acuses += len(_lista(valor.get("statuses")))
        elif "errors" in valor:
            self.errores_de_meta += len(_lista(valor.get("errors")))
        else:
            self.ignorados += 1
            self.claves_desconocidas.update(str(clave) for clave in valor)

    def _es_para_nosotros(self, valor: dict[str, Any]) -> bool:
        """Comprueba que el evento sea de nuestro número de negocio."""
        if self._phone_number_id_propio is None:
            return True  # sin configurar: no se puede discriminar
        metadata = valor.get("metadata")
        if not isinstance(metadata, dict):
            return True  # los acuses de algunos tipos no traen metadata
        recibido = metadata.get("phone_number_id")
        return recibido is None or recibido == self._phone_number_id_propio

    def _procesar_mensajes(self, valor: dict[str, Any]) -> None:
        """Extrae los mensajes de texto de un `value.messages`."""
        nombres = _nombres_por_wa_id(valor.get("contacts"))
        crudos = _lista(valor.get("messages"))

        for crudo in crudos:
            if len(self.mensajes) >= MAXIMO_MENSAJES:
                self.ignorados += 1
                continue
            # Un mensaje con forma rara no puede llevarse puestos a los demás.
            try:
                self._procesar_mensaje(crudo, nombres, unico=len(crudos) == 1)
            except Exception:  # el parser jamás propaga
                self.ignorados += 1

    def _procesar_mensaje(
        self, crudo: dict[str, Any], nombres: dict[str, str], *, unico: bool
    ) -> None:
        """Convierte un mensaje individual, o lo contabiliza como descartado."""
        if crudo.get("type") != TIPO_TEXTO:
            self.no_soportados += 1
            return

        wamid = crudo.get("id")
        remitente_crudo = crudo.get("from")
        if not isinstance(wamid, str) or not wamid or not isinstance(remitente_crudo, str):
            # Meta está migrando a IDs de usuario (BSUID) y puede dejar de
            # mandar `from`. Que quede una línea de log y no una caída muda.
            self.sin_remitente += 1
            return

        try:
            remitente = NumeroWhatsApp.desde_wa_id(remitente_crudo)
        except InvalidValueError:
            self.sin_remitente += 1
            return

        enviado_en = _a_datetime(crudo.get("timestamp"))
        if enviado_en is None:
            self.ignorados += 1
            return
        if self._ahora - enviado_en > FRESCURA_MAXIMA:
            self.vencidos += 1
            return

        texto_crudo = _texto_de(crudo)
        if texto_crudo is None:
            self.ignorados += 1
            return

        texto, se_trunco = _truncar(texto_crudo)
        if se_trunco:
            self.truncados += 1

        self.mensajes.append(
            MensajeEntrante(
                wamid=wamid,
                remitente=remitente,
                texto=texto,
                enviado_en=enviado_en,
                nombre_perfil=_nombre_para(remitente_crudo, nombres, unico=unico),
            )
        )

    def resultado(self, *, ignorados_extra: int = 0) -> ContenidoWebhook:
        """Congela el acumulador en el DTO de salida."""
        return ContenidoWebhook(
            mensajes=tuple(self.mensajes),
            acuses=self.acuses,
            errores_de_meta=self.errores_de_meta,
            no_soportados=self.no_soportados,
            vencidos=self.vencidos,
            ignorados=self.ignorados + ignorados_extra,
            truncados=self.truncados,
            sin_remitente=self.sin_remitente,
            ajenos=self.ajenos,
            claves_desconocidas=tuple(sorted(self.claves_desconocidas)),
        )


# --- Ayudas de lectura defensiva --------------------------------------------

# Límite práctico del cuerpo de un mensaje de texto en la Cloud API.
LARGO_MAXIMO_TEXTO = 4096


def _lista(valor: object) -> list[dict[str, Any]]:
    """Devuelve la lista de diccionarios de un campo, o vacía si no lo es."""
    if not isinstance(valor, list):
        return []
    return [elemento for elemento in valor if isinstance(elemento, dict)]


def _texto_de(mensaje: dict[str, Any]) -> str | None:
    """Extrae `text.body`, o None si no está o está vacío."""
    texto = mensaje.get("text")
    if not isinstance(texto, dict):
        return None
    cuerpo = texto.get("body")
    if not isinstance(cuerpo, str) or not cuerpo.strip():
        return None
    return cuerpo


def _truncar(texto: str) -> tuple[str, bool]:
    """Acota el texto al límite, cortando en el último espacio.

    Se corta por espacio y no por índice para no partir un emoji compuesto
    (secuencias con ZWJ o modificadores de tono de piel).
    """
    if len(texto) <= LARGO_MAXIMO_TEXTO:
        return texto, False

    recortado = texto[:LARGO_MAXIMO_TEXTO]
    ultimo_espacio = recortado.rfind(" ")
    if ultimo_espacio > 0:
        recortado = recortado[:ultimo_espacio]
    return recortado, True


def _a_datetime(valor: object) -> datetime | None:
    """Convierte el timestamp de Meta (epoch en segundos, como texto) a datetime.

    Es epoch, no ISO: no sirve `mapeo.a_datetime`, que además pertenece a la
    capa de persistencia y no debería cruzarse hasta acá.
    """
    if not isinstance(valor, str | int):
        return None
    try:
        return datetime.fromtimestamp(int(valor), tz=UTC)
    except (ValueError, OSError, OverflowError):
        return None


def _nombres_por_wa_id(contactos: object) -> dict[str, str]:
    """Arma el índice wa_id -> nombre de perfil a partir de `contacts[]`."""
    indice: dict[str, str] = {}
    for contacto in _lista(contactos):
        wa_id = contacto.get("wa_id")
        perfil = contacto.get("profile")
        if not isinstance(wa_id, str) or not isinstance(perfil, dict):
            continue
        nombre = perfil.get("name")
        if isinstance(nombre, str) and nombre.strip():
            indice[wa_id] = nombre.strip()
    return indice


def _nombre_para(wa_id: str, nombres: dict[str, str], *, unico: bool) -> str | None:
    """Busca el nombre del remitente, emparejando por wa_id y no por posición.

    Si el contacto no trae `wa_id` sólo se acepta cuando hay exactamente un
    contacto y un mensaje: con más de uno, adivinar emparejaría mal y el nombre
    quedaría congelado en la fila del usuario equivocado.
    """
    if wa_id in nombres:
        return nombres[wa_id]
    if unico and len(nombres) == 1:
        return next(iter(nombres.values()))
    return None
