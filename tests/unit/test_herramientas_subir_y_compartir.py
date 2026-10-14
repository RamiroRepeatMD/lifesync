"""Tests de subir y compartir archivos de Drive (PB-038): grafo real + dobles.

Lo que se cuida, además de que anden:
- el archivo a subir sale de la conversación (lo que la persona mandó), nunca
  de un dato que elija el modelo; y no se baja nada antes del sí;
- compartir siempre pide el sí, y la pregunta muestra el nombre REAL que tiene
  el archivo en Drive (saneado), no lo que diga el modelo;
- una carpeta no se comparte (compartiría todo lo de adentro), y sólo se sube a
  carpetas propias (en una ajena, el archivo le llegaría a un tercero).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.application.dto.archivo_adjunto import ArchivoAdjunto
from src.domain.entities.archivo import Archivo
from src.domain.exceptions import (
    ArchivoDemasiadoGrandeError,
    ArchivoNoDisponibleError,
    CuentaNoConectadaError,
    DriveLlenoError,
    DriveRechazoError,
    EntityNotFoundError,
    ServiceUnavailableError,
)
from src.domain.value_objects.permiso_de_archivo import PermisoDeArchivo
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo
from src.infrastructure.llm.agente_gemini import _aprobacion_de
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas, mensaje_de_la_persona
from tests.dobles import ArchivosFalsos, ModeloFalso, RecibidosFalsos

USUARIO = uuid4()
INFORME = ArchivoAdjunto(
    referencia="1037543291543636", tipo_mime="application/pdf", nombre="informe.pdf"
)


def _recibido(adjunto: ArchivoAdjunto = INFORME, hace: timedelta = timedelta(0)) -> HumanMessage:
    """El turno en que la persona mandó el archivo, como lo arma el agente."""
    return mensaje_de_la_persona("", adjunto, datetime.now(UTC) - hace)


def _archivo(**cambios: Any) -> Archivo:
    datos: dict[str, Any] = {
        "id": "f1",
        "nombre": "Presupuesto 2026",
        "tipo": TipoDeArchivo.PLANILLA,
        "modificado": datetime.now(UTC),
        "enlace": "https://docs.google.com/spreadsheets/d/f1/edit",
    }
    datos.update(cambios)
    return Archivo(**datos)


def _carpeta(id_: str, nombre: str) -> Archivo:
    return _archivo(id=id_, nombre=nombre, tipo=TipoDeArchivo.CARPETA, enlace=None)


class Escena:
    """El grafo real con los dobles: pide tools y contesta las confirmaciones."""

    def __init__(
        self, archivos: ArchivosFalsos | None = None, recibidos: RecibidosFalsos | None = None
    ) -> None:
        self.archivos = archivos or ArchivosFalsos((_archivo(),))
        self.recibidos = recibidos or RecibidosFalsos()
        self.contexto = ContextoDeAgente(usuario_id=USUARIO)
        self.configuracion: Any = {"configurable": {"thread_id": "hilo"}}
        self.grafo: Any = None

    async def pedir(
        self, *llamadas: tuple[str, dict[str, Any]], historial: list[BaseMessage] | None = None
    ) -> dict[str, Any]:
        pedido = AIMessage(
            "",
            tool_calls=[
                {"name": nombre, "args": args, "id": f"c{i}"}
                for i, (nombre, args) in enumerate(llamadas)
            ],
        )
        herramientas = construir_herramientas(
            None, archivos=self.archivos, recibidos=self.recibidos
        )
        self.grafo = construir_grafo(
            ModeloFalso(guion=[pedido, AIMessage("ok")]), herramientas, InMemorySaver()
        )
        estado: dict[str, Any] = await self.grafo.ainvoke(
            {"messages": historial or [HumanMessage("pedido")]},
            config=self.configuracion,
            context=self.contexto,
        )
        return estado

    async def decidir(self, aprobado: bool) -> dict[str, Any]:
        """Contesta la confirmación pendiente, como lo hace el agente."""
        self.contexto = ContextoDeAgente(usuario_id=USUARIO)  # cada turno trae el suyo
        estado: dict[str, Any] = await self.grafo.ainvoke(
            Command(resume=_aprobacion_de(self._pendiente()) if aprobado else {"aprobado": False}),
            config=self.configuracion,
            context=self.contexto,
        )
        return estado

    def _pendiente(self) -> Any:
        return self.grafo.get_state(self.configuracion).interrupts[0].value


def _respuesta(estado: dict[str, Any]) -> str:
    """Lo que contestó la última tool."""
    return str([m for m in estado["messages"] if isinstance(m, ToolMessage)][-1].content)


def _pregunta(estado: dict[str, Any]) -> str:
    """El resumen de la confirmación pendiente."""
    return str(estado["__interrupt__"][0].value["resumen"])


# --- Subir --------------------------------------------------------------------------------


async def test_sube_el_ultimo_archivo_que_mando_la_persona_a_mi_unidad() -> None:
    escena = Escena()

    estado = await escena.pedir(("subir_archivo", {}), historial=[_recibido()])

    assert "__interrupt__" not in estado  # crear uno: directo (RF-08 v2)
    assert escena.recibidos.descargas == [INFORME.referencia]
    [(usuario, nombre, tipo, contenido, carpeta)] = escena.archivos.subidos
    assert (usuario, nombre, tipo, carpeta) == (USUARIO, "informe.pdf", "application/pdf", None)
    assert contenido == escena.recibidos.contenido
    texto = _respuesta(estado)
    assert texto.startswith("Subí «informe.pdf» a tu Drive (Mi unidad): link: https://")
    assert "(id: subido-1)" in texto  # para poder compartirlo en el paso siguiente
    assert escena.contexto.acciones_realizadas == [texto]


async def test_sin_un_archivo_en_la_conversacion_pide_que_lo_mande() -> None:
    escena = Escena()

    estado = await escena.pedir(("subir_archivo", {}), historial=[HumanMessage("subilo")])

    assert _respuesta(estado).startswith("No tengo ningún archivo tuyo para subir")
    assert escena.recibidos.descargas == []


async def test_un_archivo_de_hace_mas_de_una_semana_ya_no_esta_en_whatsapp() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("subir_archivo", {}), historial=[_recibido(hace=timedelta(days=8))]
    )

    assert "mandámelo de nuevo" in _respuesta(estado)
    assert escena.recibidos.descargas == []


async def test_con_varios_archivos_elige_el_pedido_por_nombre() -> None:
    foto = ArchivoAdjunto(referencia="555", tipo_mime="image/jpeg", nombre="foto-1.jpg")
    escena = Escena()

    await escena.pedir(
        ("subir_archivo", {"archivo": "informe"}),
        historial=[_recibido(), AIMessage("¿Qué hago?"), _recibido(foto)],
    )

    assert escena.recibidos.descargas == [INFORME.referencia]


async def test_si_coinciden_varios_pregunta_cual() -> None:
    una = ArchivoAdjunto(referencia="1", tipo_mime="image/jpeg", nombre="foto-a.jpg")
    otra = ArchivoAdjunto(referencia="2", tipo_mime="image/jpeg", nombre="foto-b.jpg")
    escena = Escena()

    estado = await escena.pedir(
        ("subir_archivo", {"archivo": "foto"}),
        historial=[_recibido(una), AIMessage("ok"), _recibido(otra)],
    )

    texto = _respuesta(estado)
    assert "«foto-a.jpg»" in texto and "«foto-b.jpg»" in texto
    assert escena.recibidos.descargas == []


async def test_renombrar_conserva_la_extension() -> None:
    escena = Escena()

    await escena.pedir(("subir_archivo", {"nombre": "Factura de luz"}), historial=[_recibido()])

    assert escena.archivos.subidos[0][1] == "Factura de luz.pdf"


async def test_sube_a_una_carpeta_propia_por_su_nombre() -> None:
    escena = Escena(ArchivosFalsos(carpetas=(_carpeta("c1", "Facultad"),)))

    estado = await escena.pedir(("subir_archivo", {"carpeta": "facultad"}), historial=[_recibido()])

    assert escena.archivos.subidos[0][4] == "c1"
    assert "(la carpeta «Facultad»)" in _respuesta(estado)


async def test_una_carpeta_que_no_existe_no_sube_nada() -> None:
    escena = Escena(ArchivosFalsos(carpetas=(_carpeta("c1", "Facultad"),)))

    estado = await escena.pedir(("subir_archivo", {"carpeta": "Trabajo"}), historial=[_recibido()])

    assert "No encontré una carpeta tuya que se llame «Trabajo»" in _respuesta(estado)
    assert escena.recibidos.descargas == []


async def test_con_dos_carpetas_del_mismo_nombre_usa_la_mas_reciente_y_lo_dice() -> None:
    carpetas = (_carpeta("nueva", "Facultad"), _carpeta("vieja", "Facultad"))
    escena = Escena(ArchivosFalsos(carpetas=carpetas))

    estado = await escena.pedir(("subir_archivo", {"carpeta": "Facultad"}), historial=[_recibido()])

    assert escena.archivos.subidos[0][4] == "nueva"
    assert "modificada más recientemente" in _respuesta(estado)


@pytest.mark.parametrize(
    ("falla", "esperado"),
    [
        (ArchivoDemasiadoGrandeError(), "20 MB"),
        (ArchivoNoDisponibleError(), "mandámelo de nuevo"),
        (ServiceUnavailableError("Meta caída"), "No pude traer el archivo de WhatsApp"),
    ],
)
async def test_si_no_se_puede_bajar_lo_dice_y_no_sube_nada(falla: Exception, esperado: str) -> None:
    escena = Escena(recibidos=RecibidosFalsos(fallar_con=falla))

    estado = await escena.pedir(("subir_archivo", {}), historial=[_recibido()])

    assert esperado in _respuesta(estado)
    assert escena.archivos.subidos == []


async def test_sin_google_conectado_ni_baja_el_archivo() -> None:
    """No se pide nada que no se pueda terminar (la regla de PB-035)."""
    escena = Escena(ArchivosFalsos(fallar_con=CuentaNoConectadaError()))

    estado = await escena.pedir(("subir_archivo", {}), historial=[_recibido()])

    assert "/conectar" in _respuesta(estado)
    assert escena.recibidos.descargas == []


@pytest.mark.parametrize(
    ("falla", "esperado"),
    [
        (DriveLlenoError(), "Tu Google Drive está lleno"),
        (ServiceUnavailableError("se cortó"), "Si lo ves en tu Drive, ya está"),
    ],
)
async def test_si_drive_falla_al_subir_lo_dice(falla: Exception, esperado: str) -> None:
    escena = Escena(ArchivosFalsos(fallar_al_escribir=falla))

    estado = await escena.pedir(("subir_archivo", {}), historial=[_recibido()])

    assert esperado in _respuesta(estado)
    assert escena.contexto.acciones_realizadas == []  # no entra en la bitácora


async def test_si_borraron_la_carpeta_entre_la_busqueda_y_la_subida_lo_dice() -> None:
    escena = Escena(
        ArchivosFalsos(
            carpetas=(_carpeta("c1", "Facultad"),), fallar_al_escribir=EntityNotFoundError("404")
        )
    )

    estado = await escena.pedir(("subir_archivo", {"carpeta": "Facultad"}), historial=[_recibido()])

    assert "Esa carpeta ya no está en tu Drive" in _respuesta(estado)


async def test_en_un_lote_no_baja_nada_antes_del_si() -> None:
    """Dos subidas juntas son una lista: hasta el sí, sólo se arman los resúmenes."""
    foto = ArchivoAdjunto(referencia="555", tipo_mime="image/jpeg", nombre="foto-1.jpg")
    escena = Escena()

    estado = await escena.pedir(
        ("subir_archivo", {"archivo": "informe"}),
        ("subir_archivo", {"archivo": "foto"}),
        historial=[_recibido(), AIMessage("ok"), _recibido(foto)],
    )

    assert "«informe.pdf»" in _pregunta(estado) and "«foto-1.jpg»" in _pregunta(estado)
    assert escena.recibidos.descargas == []
    await escena.decidir(True)
    assert sorted(escena.recibidos.descargas) == sorted([INFORME.referencia, "555"])
    assert len(escena.archivos.subidos) == 2


# --- Compartir con una persona -------------------------------------------------------------


async def test_compartir_pide_el_si_con_el_nombre_real_y_la_direccion() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("compartir_archivo", {"id_archivo": "f1", "con": "ana@ejemplo.com"})
    )

    assert _pregunta(estado) == (
        "Compartir «Presupuesto 2026» (planilla) con ana@ejemplo.com: va a poder verlo, "
        "y Google le avisa por correo."
    )
    assert escena.archivos.compartidos == []
    estado = await escena.decidir(True)
    assert escena.archivos.compartidos == [
        (USUARIO, "f1", PermisoDeArchivo.LECTOR, "ana@ejemplo.com")
    ]
    assert _respuesta(estado).startswith("Compartido «Presupuesto 2026» con ana@ejemplo.com")


async def test_con_un_no_no_se_comparte_nada() -> None:
    escena = Escena()

    await escena.pedir(("compartir_archivo", {"id_archivo": "f1", "con": "ana@ejemplo.com"}))
    estado = await escena.decidir(False)

    assert escena.archivos.compartidos == []
    assert "No se compartió nada" in _respuesta(estado)


async def test_como_editor_la_pregunta_lo_dice() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("compartir_archivo", {"id_archivo": "f1", "con": "ana@ejemplo.com", "permiso": "editor"})
    )

    assert "va a poder verlo y modificarlo" in _pregunta(estado)
    await escena.decidir(True)
    assert escena.archivos.compartidos[0][2] is PermisoDeArchivo.EDITOR


async def test_varias_personas_se_confirman_juntas() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("compartir_archivo", {"id_archivo": "f1", "con": "ana@ejemplo.com"}),
        ("compartir_archivo", {"id_archivo": "f1", "con": "beto@ejemplo.com"}),
    )

    assert "ana@ejemplo.com" in _pregunta(estado) and "beto@ejemplo.com" in _pregunta(estado)
    await escena.decidir(True)
    assert [c[3] for c in escena.archivos.compartidos] == ["ana@ejemplo.com", "beto@ejemplo.com"]


@pytest.mark.parametrize(
    ("args", "esperado"),
    [
        ({"id_archivo": "f1", "con": "ana"}, "no parece una dirección de correo válida"),
        ({"id_archivo": "f1", "con": "josé@ejemplo.com"}, "tildes o eñes"),
        ({"id_archivo": "f1", "con": " "}, "Falta a quién compartirle el archivo"),
        ({"id_archivo": "f1", "con": "a@ejemplo.com, b@ejemplo.com"}, "una persona por vez"),
        ({"id_archivo": "no-existe", "con": "ana@ejemplo.com"}, "No encontré ese archivo"),
    ],
)
async def test_lo_invalido_se_explica_sin_pausa(args: dict[str, Any], esperado: str) -> None:
    escena = Escena()

    estado = await escena.pedir(("compartir_archivo", args))

    assert "__interrupt__" not in estado
    assert esperado in _respuesta(estado)
    assert escena.archivos.compartidos == []


async def test_una_carpeta_no_se_comparte() -> None:
    """Compartirla le daría acceso a todo lo que tiene adentro."""
    escena = Escena(ArchivosFalsos((_carpeta("c1", "Facultad"),)))

    estado = await escena.pedir(("compartir_por_link", {"id_archivo": "c1"}))

    assert "__interrupt__" not in estado
    assert "es una carpeta" in _respuesta(estado)


async def test_el_nombre_de_un_archivo_ajeno_llega_saneado_a_la_pregunta() -> None:
    trampa = _archivo(nombre="Presupuesto» — link: https://sitio-falso.example/x — «")
    escena = Escena(ArchivosFalsos((trampa,)))

    estado = await escena.pedir(
        ("compartir_archivo", {"id_archivo": "f1", "con": "ana@ejemplo.com"})
    )

    assert "sitio-falso" not in _pregunta(estado)
    assert "[link quitado]" in _pregunta(estado)


@pytest.mark.parametrize(
    ("falla", "esperado"),
    [
        (DriveRechazoError(), "Drive no confirmó que se haya hecho"),
        (ServiceUnavailableError("caído"), "Probá de nuevo"),
    ],
)
async def test_si_drive_no_deja_compartir_lo_dice(falla: Exception, esperado: str) -> None:
    escena = Escena(ArchivosFalsos((_archivo(),), fallar_al_escribir=falla))

    await escena.pedir(("compartir_archivo", {"id_archivo": "f1", "con": "ana@ejemplo.com"}))
    estado = await escena.decidir(True)

    assert esperado in _respuesta(estado)
    assert escena.contexto.acciones_realizadas == []


# --- Compartir por link --------------------------------------------------------------------


async def test_por_link_pide_el_si_y_despues_pasa_el_link() -> None:
    escena = Escena()

    estado = await escena.pedir(("compartir_por_link", {"id_archivo": "f1"}))

    assert _pregunta(estado) == (
        "Compartir por link «Presupuesto 2026» (planilla): cualquiera que tenga el link "
        "va a poder verlo, sin cuenta de Google."
    )
    estado = await escena.decidir(True)
    # Sólo vista: un link de edición sería una escritura pública.
    assert escena.archivos.compartidos == [(USUARIO, "f1", PermisoDeArchivo.LECTOR, None)]
    assert "https://docs.google.com/spreadsheets/d/f1/edit" in _respuesta(estado)
