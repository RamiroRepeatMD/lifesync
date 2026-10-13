"""Tests del adaptador de Gmail (PB-033). Sin red: MockTransport.

Los JSON respetan la forma real de la API: `messages.list` devuelve sólo ids,
`messages.get` trae el árbol MIME con los cuerpos en base64url (a veces sin
relleno) y `internalDate` en milisegundos.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import httpx
import pytest

from src.application.use_cases.conectar_google import ConectarGoogle
from src.domain.entities.oauth_token import OAuthToken
from src.domain.exceptions import (
    AutorizacionFallidaError,
    CuentaNoConectadaError,
    EntityNotFoundError,
    EnvioInciertoError,
    EnvioRechazadoError,
    PermisoInsuficienteError,
    ServiceUnavailableError,
)
from src.domain.value_objects.proveedor_oauth import ProveedorOAuth
from src.infrastructure.external.google.gmail import (
    MAX_CARACTERES_DE_CUERPO,
    GmailGoogle,
)
from tests.dobles import AutorizadorFalso, RepositorioOAuthTokenEnMemoria

USUARIO = uuid4()
Manejador = Callable[[httpx.Request], httpx.Response]
# 30/09/2026 14:32 UTC, en milisegundos, como lo manda Gmail.
INTERNA = str(int(datetime(2026, 9, 30, 14, 32, tzinfo=UTC).timestamp() * 1000))


def _b64(texto: str, charset: str = "utf-8") -> str:
    # Gmail usa base64url y muchas veces sin el relleno "=".
    return base64.urlsafe_b64encode(texto.encode(charset)).decode().rstrip("=")


def _parte(tipo: str, texto: str, charset: str = "utf-8") -> dict[str, Any]:
    return {
        "mimeType": tipo,
        "headers": [{"name": "Content-Type", "value": f'{tipo}; charset="{charset}"'}],
        "body": {"data": _b64(texto, charset)},
    }


def _mensaje(payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
    cabeceras = [
        {"name": "From", "value": "Juan Pérez <juan@ejemplo.com>"},
        {"name": "Subject", "value": "Factura de septiembre"},
    ]
    return {
        "id": "m1",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Te adjunto la factura &amp; el detalle",
        "internalDate": INTERNA,
        "payload": {"headers": cabeceras, **payload},
        **extra,
    }


async def _gmail(pedidos: list[httpx.Request], manejador: Manejador) -> GmailGoogle:
    tokens = RepositorioOAuthTokenEnMemoria()
    await tokens.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=ProveedorOAuth.GOOGLE,
            access_token="access-vigente",
            refresh_token="refresh",
            expira_en=datetime.now(UTC) + timedelta(hours=1),
        )
    )

    def interceptar(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        return manejador(pedido)

    http = httpx.AsyncClient(transport=httpx.MockTransport(interceptar))
    return GmailGoogle(http, ConectarGoogle(tokens, AutorizadorFalso(usuario_fijo=USUARIO)))


def _bandeja(*mensajes: dict[str, Any]) -> Manejador:
    """Contesta la lista de ids y cada mensaje por su id."""
    por_id = {m["id"]: m for m in mensajes}

    def responder(pedido: httpx.Request) -> httpx.Response:
        if pedido.url.path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": i} for i in por_id]})
        return httpx.Response(200, json=por_id[pedido.url.path.rsplit("/", 1)[1]])

    return responder


def _abierto(payload: dict[str, Any]) -> Manejador:
    return lambda _: httpx.Response(200, json=_mensaje(payload))


# --- Buscar --------------------------------------------------------------------


async def test_buscar_manda_la_consulta_y_pide_los_metadatos_de_cada_uno() -> None:
    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, _bandeja(_mensaje({}), _mensaje({}, id="m2")))

    correos = await gmail.buscar(USUARIO, "from:juan is:unread", 5)

    lista = parse_qs(urlparse(str(pedidos[0].url)).query)
    assert lista["q"] == ["from:juan is:unread"]
    assert lista["maxResults"] == ["5"]
    detalles = [parse_qs(urlparse(str(p.url)).query) for p in pedidos[1:]]
    assert all(d["format"] == ["metadata"] for d in detalles)
    assert [c.id for c in correos] == ["m1", "m2"]


async def test_un_correo_del_listado_trae_lo_que_hay_que_mostrar() -> None:
    gmail = await _gmail([], _bandeja(_mensaje({})))

    (correo,) = await gmail.buscar(USUARIO, "", 5)

    assert correo.remitente == "Juan Pérez <juan@ejemplo.com>"
    assert correo.asunto == "Factura de septiembre"
    assert correo.fecha == datetime(2026, 9, 30, 14, 32, tzinfo=UTC)
    assert correo.no_leido is True
    assert correo.fragmento == "Te adjunto la factura & el detalle"  # entidades resueltas
    assert correo.cuerpo is None  # el listado no trae cuerpo


async def test_sin_consulta_busca_en_la_bandeja_de_entrada_y_acota_la_cantidad() -> None:
    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, lambda _: httpx.Response(200, json={}))

    assert await gmail.buscar(USUARIO, "  ", 50) == ()

    lista = parse_qs(urlparse(str(pedidos[0].url)).query)
    assert lista["q"] == ["in:inbox"]
    assert lista["maxResults"] == ["10"]
    assert len(pedidos) == 1  # sin ids, no hay nada más que pedir


async def test_un_asunto_codificado_en_rfc_2047_se_decodifica() -> None:
    codificado = "=?UTF-8?B?" + base64.b64encode("Reunión mañana".encode()).decode() + "?="
    mensaje = _mensaje({})
    mensaje["payload"]["headers"][1]["value"] = codificado
    gmail = await _gmail([], _bandeja(mensaje))

    (correo,) = await gmail.buscar(USUARIO, "", 5)

    assert correo.asunto == "Reunión mañana"


# --- Leer: el árbol MIME ---------------------------------------------------------


async def test_con_texto_y_html_gana_el_texto_plano() -> None:
    alternativa = {
        "mimeType": "multipart/alternative",
        "parts": [_parte("text/plain", "Hola José, qué tal"), _parte("text/html", "<b>HTML</b>")],
    }
    gmail = await _gmail([], _abierto(alternativa))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo == "Hola José, qué tal"  # tildes intactas tras base64url


async def test_si_solo_hay_html_se_extrae_el_texto() -> None:
    html = (
        "<html><head><style>p{color:red}</style></head><body>"
        "<p>Hola &amp; chau</p><script>alert(1)</script><div>Segunda línea</div>"
        "</body></html>"
    )
    gmail = await _gmail([], _abierto(_parte("text/html", html)))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo is not None
    assert "Hola & chau" in correo.cuerpo
    assert "Segunda línea" in correo.cuerpo
    assert "<" not in correo.cuerpo
    assert "alert" not in correo.cuerpo and "color" not in correo.cuerpo


async def test_un_mixto_anidado_da_el_cuerpo_y_los_nombres_de_los_adjuntos() -> None:
    mixto = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [_parte("text/plain", "Va la factura"), _parte("text/html", "<p>x</p>")],
            },
            {
                "mimeType": "application/pdf",
                "filename": "factura.pdf",
                "body": {"attachmentId": "a1", "size": 1234},
            },
        ],
    }
    gmail = await _gmail([], _abierto(mixto))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo == "Va la factura"
    assert correo.adjuntos == ("factura.pdf",)


async def test_respeta_el_charset_de_la_parte() -> None:
    gmail = await _gmail([], _abierto(_parte("text/plain", "Año y señal", charset="iso-8859-1")))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo == "Año y señal"


async def test_saca_lo_citado_de_hilos_viejos() -> None:
    texto = "Dale, nos vemos.\n\nEl lunes, Ana escribió:\n> ¿Nos vemos el martes?\n> Avisame"
    gmail = await _gmail([], _abierto(_parte("text/plain", texto)))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo is not None
    assert "Dale, nos vemos." in correo.cuerpo
    assert "martes" not in correo.cuerpo


async def test_un_cuerpo_larguisimo_se_recorta() -> None:
    gmail = await _gmail([], _abierto(_parte("text/plain", "x" * 10_000)))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.cuerpo is not None
    assert correo.cuerpo.count("x") == MAX_CARACTERES_DE_CUERPO
    assert correo.cuerpo.endswith("[El correo sigue: se recortó.]")


# --- Errores ---------------------------------------------------------------------


async def test_un_correo_que_no_existe_es_error_con_nombre() -> None:
    gmail = await _gmail([], lambda _: httpx.Response(404, json={}))

    with pytest.raises(EntityNotFoundError):
        await gmail.leer(USUARIO, "fantasma")


async def test_api_deshabilitada_no_manda_a_reconectar() -> None:
    """La lección de Tasks: el scope solo no alcanza, hay que habilitar la API."""
    cuerpo = {
        "error": {
            "code": 403,
            "message": "Gmail API has not been used in project X before or it is disabled.",
            "details": [{"reason": "SERVICE_DISABLED"}],
        }
    }
    gmail = await _gmail([], lambda _: httpx.Response(403, json=cuerpo))

    with pytest.raises(ServiceUnavailableError):
        await gmail.buscar(USUARIO, "", 5)


async def test_sin_el_scope_de_correo_es_permiso_insuficiente() -> None:
    """Una cuenta conectada antes de PB-033 no tiene gmail.readonly."""
    gmail = await _gmail([], lambda _: httpx.Response(403, json={}))

    with pytest.raises(PermisoInsuficienteError):
        await gmail.buscar(USUARIO, "", 5)


# --- Enviar (PB-032) --------------------------------------------------------------


async def test_enviar_arma_un_correo_valido_con_asunto_en_utf8() -> None:
    import json
    from email import message_from_bytes
    from email.policy import default

    from src.domain.entities.correo import CorreoSaliente

    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, lambda _: httpx.Response(200, json={"id": "enviado-1"}))

    identificador = await gmail.enviar(
        USUARIO,
        CorreoSaliente(
            ("ana@ejemplo.com", "beto@ejemplo.com"), "Reunión mañana", "Llego tarde, perdón."
        ),
    )

    (pedido,) = pedidos
    assert pedido.method == "POST"
    assert pedido.url.path.endswith("/messages/send")
    crudo = json.loads(pedido.content)["raw"]
    correo = message_from_bytes(
        base64.urlsafe_b64decode(crudo + "=" * (-len(crudo) % 4)), policy=default
    )
    assert correo["To"] == "ana@ejemplo.com, beto@ejemplo.com"
    assert correo["Subject"] == "Reunión mañana"  # decodificado: viajó en RFC 2047
    assert correo.get_content().strip() == "Llego tarde, perdón."
    assert "From" not in correo  # lo pone Gmail: la cuenta autenticada
    assert identificador == "enviado-1"


async def test_enviar_sin_el_scope_es_permiso_insuficiente() -> None:
    from src.domain.entities.correo import CorreoSaliente

    gmail = await _gmail([], lambda _: httpx.Response(403, json={}))

    with pytest.raises(PermisoInsuficienteError):
        await gmail.enviar(USUARIO, CorreoSaliente(("a@b.com",), "x", "y"))


# --- Responder y reenviar (PB-034) --------------------------------------------------


def _con_hilo(payload: dict[str, Any]) -> Manejador:
    """Un correo con lo que hace falta para contestarlo dentro de su hilo."""
    mensaje = _mensaje(payload, threadId="hilo-7")
    mensaje["payload"]["headers"] += [
        {"name": "Message-ID", "value": "<abc123@mail.ejemplo.com>"},
        {"name": "References", "value": "<viejo1@x.com>\r\n <viejo2@x.com>"},  # plegado
        {"name": "Reply-To", "value": "Facturación <cobros@ejemplo.com>"},
    ]
    return lambda _: httpx.Response(200, json=mensaje)


def _decodificar_envio(pedido: httpx.Request) -> tuple[Any, dict[str, Any]]:
    import json
    from email import message_from_bytes
    from email.policy import default

    cuerpo = json.loads(pedido.content)
    crudo = cuerpo["raw"]
    correo = message_from_bytes(
        base64.urlsafe_b64decode(crudo + "=" * (-len(crudo) % 4)), policy=default
    )
    return correo, cuerpo


async def test_leer_trae_los_datos_del_hilo() -> None:
    gmail = await _gmail([], _con_hilo(_parte("text/plain", "Hola")))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.hilo_id == "hilo-7"
    assert correo.mensaje_id == "<abc123@mail.ejemplo.com>"
    assert correo.referencias == "<viejo1@x.com> <viejo2@x.com>"  # en una línea
    assert correo.responder_a == ("cobros@ejemplo.com",)


async def test_leer_completo_no_recorta_ni_quita_las_citas() -> None:
    """Lo que se reenvía tiene que ir entero: lo recortado es sólo para el modelo."""
    texto = "Arriba.\n> lo citado\n" + "x" * 10_000
    gmail = await _gmail([], _abierto(_parte("text/plain", texto)))

    correo = await gmail.leer(USUARIO, "m1", completo=True)

    assert correo.cuerpo is not None
    assert "> lo citado" in correo.cuerpo
    assert correo.cuerpo.count("x") == 10_000


async def test_responder_engancha_el_correo_al_hilo() -> None:
    """Lo que pide Gmail para hilar: threadId en el cuerpo, In-Reply-To y References."""
    from src.domain.entities.correo import CorreoSaliente

    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, lambda _: httpx.Response(200, json={"id": "r1"}))

    await gmail.enviar(
        USUARIO,
        CorreoSaliente(
            ("cobros@ejemplo.com",),
            "Re: Factura de septiembre",
            "Gracias, ya la pagué.",
            hilo_id="hilo-7",
            en_respuesta_a="<abc123@mail.ejemplo.com>",
            referencias="<viejo1@x.com> <viejo2@x.com>",
        ),
    )

    correo, cuerpo = _decodificar_envio(pedidos[0])
    assert cuerpo["threadId"] == "hilo-7"
    assert correo["In-Reply-To"] == "<abc123@mail.ejemplo.com>"
    assert correo["References"] == "<viejo1@x.com> <viejo2@x.com> <abc123@mail.ejemplo.com>"


async def test_un_correo_nuevo_no_lleva_nada_del_hilo() -> None:
    from src.domain.entities.correo import CorreoSaliente

    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, lambda _: httpx.Response(200, json={"id": "n1"}))

    await gmail.enviar(USUARIO, CorreoSaliente(("ana@ejemplo.com",), "Hola", "Texto"))

    correo, cuerpo = _decodificar_envio(pedidos[0])
    assert "threadId" not in cuerpo
    assert "In-Reply-To" not in correo
    assert "References" not in correo


def test_un_message_id_con_saltos_de_linea_no_se_acepta() -> None:
    """Viene de un correo ajeno: un salto permitiría inyectar un Bcc."""
    from src.domain.entities.correo import CorreoSaliente
    from src.domain.exceptions import InvalidValueError

    with pytest.raises(InvalidValueError):
        CorreoSaliente(
            ("ana@ejemplo.com",),
            "Re: Hola",
            "Texto",
            en_respuesta_a="<a@x.com>\r\nBcc: atacante@malo.com",
        )


# --- Errores del envío (PB-035): ¿salió o no? ------------------------------------
#
# La regla: `enviar` lanza `EnvioInciertoError` si y sólo si el correo PUDO haber
# salido. Cualquier otra falla garantiza que no salió, y se puede volver a pedir.


async def _enviar_ante(manejador: Manejador) -> None:
    """Manda un correo nuevo contra la respuesta (o la falla) de Gmail que se indique."""
    from src.domain.entities.correo import CorreoSaliente

    gmail = await _gmail([], manejador)
    await gmail.enviar(USUARIO, CorreoSaliente(("ana@ejemplo.com",), "Hola", "Llego tarde"))


def _que_falle(falla: type[httpx.TransportError]) -> Manejador:
    def manejador(pedido: httpx.Request) -> httpx.Response:
        raise falla("simulada", request=pedido)

    return manejador


def _error(status: int, mensaje: str, motivo: str) -> httpx.Response:
    cuerpo = {"error": {"code": status, "message": mensaje, "errors": [{"reason": motivo}]}}
    return httpx.Response(status, json=cuerpo)


@pytest.mark.parametrize("status", [500, 502, 503])
async def test_un_5xx_al_enviar_deja_el_envio_en_duda(status: int) -> None:
    """Gmail falló a mitad de camino: el correo pudo haber salido igual."""
    with pytest.raises(EnvioInciertoError):
        await _enviar_ante(lambda _: _error(status, "Backend Error", "backendError"))


@pytest.mark.parametrize("falla", [httpx.ReadTimeout, httpx.RemoteProtocolError])
async def test_si_se_pierde_la_respuesta_el_envio_queda_en_duda(
    falla: type[httpx.TransportError],
) -> None:
    """El pedido llegó a Gmail y la respuesta no volvió: no hay forma de saber si salió."""
    with pytest.raises(EnvioInciertoError):
        await _enviar_ante(_que_falle(falla))


@pytest.mark.parametrize("falla", [httpx.ConnectError, httpx.ConnectTimeout])
async def test_si_no_se_pudo_conectar_seguro_que_no_salio(
    falla: type[httpx.TransportError],
) -> None:
    """El pedido nunca llegó a Gmail: se puede volver a pedir sin miedo a duplicarlo."""
    with pytest.raises(ServiceUnavailableError) as capturado:
        await _enviar_ante(_que_falle(falla))

    assert not isinstance(capturado.value, EnvioInciertoError)


async def test_un_400_es_un_rechazo_y_el_correo_no_salio() -> None:
    """Reintentar igual da lo mismo: hay que corregir el pedido (una dirección, por ejemplo)."""
    with pytest.raises(EnvioRechazadoError):
        await _enviar_ante(lambda _: _error(400, "Invalid To header", "invalidArgument"))


@pytest.mark.parametrize(
    ("status", "motivo"),
    [(429, "rateLimitExceeded"), (403, "userRateLimitExceeded")],
    ids=["429", "403_por_limite"],
)
async def test_el_limite_de_envios_no_deja_el_envio_en_duda(status: int, motivo: str) -> None:
    """Gmail frenó el pedido antes de mandar nada: no salió, y más tarde se puede."""
    with pytest.raises(ServiceUnavailableError) as capturado:
        await _enviar_ante(lambda _: _error(status, "Rate Limit Exceeded", motivo))

    assert not isinstance(capturado.value, (EnvioInciertoError, EnvioRechazadoError))


async def test_un_5xx_al_leer_no_es_un_envio_en_duda() -> None:
    """Leer es idempotente: la duda existe sólo para lo que pudo haber salido."""
    gmail = await _gmail([], lambda _: httpx.Response(503, json={}))

    with pytest.raises(ServiceUnavailableError) as capturado:
        await gmail.buscar(USUARIO, "", 5)

    assert not isinstance(capturado.value, EnvioInciertoError)


# --- Comprobar la cuenta antes de confirmar un envío (PB-035) -------------------------


async def test_comprobar_acceso_con_el_token_vigente_no_le_pide_nada_a_gmail() -> None:
    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, lambda _: httpx.Response(500))

    await gmail.comprobar_acceso(USUARIO)

    assert pedidos == []


async def test_comprobar_acceso_sin_cuenta_conectada_lo_dice() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    conectar = ConectarGoogle(RepositorioOAuthTokenEnMemoria(), AutorizadorFalso(USUARIO))
    gmail = GmailGoogle(http, conectar)

    with pytest.raises(CuentaNoConectadaError):
        await gmail.comprobar_acceso(USUARIO)


async def test_comprobar_acceso_con_el_permiso_vencido_lo_dice() -> None:
    """El caso del modo Testing: el refresh token dura 7 días y Google lo rechaza."""
    tokens = RepositorioOAuthTokenEnMemoria()
    await tokens.guardar(
        OAuthToken(
            usuario_id=USUARIO,
            proveedor=ProveedorOAuth.GOOGLE,
            access_token="access-viejo",
            refresh_token="refresh-vencido",
            expira_en=datetime.now(UTC) - timedelta(minutes=5),
        )
    )
    autorizador = AutorizadorFalso(USUARIO)
    autorizador.fallar_refresco = AutorizacionFallidaError("invalid_grant")
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    gmail = GmailGoogle(http, ConectarGoogle(tokens, autorizador))

    with pytest.raises(AutorizacionFallidaError):
        await gmail.comprobar_acceso(USUARIO)


# --- Lo que encontró la integración de punta a punta (PB-036) -----------------------


@pytest.mark.parametrize(
    "correo_id",
    ["../profile", "../../settings/filters", "m1/attachments/A1", "m1?format=raw", ""],
    ids=["perfil", "fuera_de_users_me", "adjunto", "con_query", "vacio"],
)
async def test_un_id_que_no_es_de_gmail_no_sale_como_pedido(correo_id: str) -> None:
    """El id lo pasa el modelo: armado en la URL tal cual, "../profile" pedía el perfil."""
    pedidos: list[httpx.Request] = []
    gmail = await _gmail(pedidos, _abierto(_parte("text/plain", "Hola")))

    with pytest.raises(EntityNotFoundError):
        await gmail.leer(USUARIO, correo_id)

    assert pedidos == []


async def test_un_correo_sin_asunto_queda_con_el_asunto_vacio() -> None:
    """El "(sin asunto)" es de la pantalla: en la entidad viajaba al responder."""
    mensaje = _mensaje(_parte("text/plain", "Mirá la foto"))
    mensaje["payload"]["headers"] = [
        h for h in mensaje["payload"]["headers"] if h["name"] != "Subject"
    ]
    gmail = await _gmail([], lambda _: httpx.Response(200, json=mensaje))

    correo = await gmail.leer(USUARIO, "m1")

    assert correo.asunto == ""
