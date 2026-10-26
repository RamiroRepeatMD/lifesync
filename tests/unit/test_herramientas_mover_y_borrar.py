"""Tests de renombrar, mover y mandar a la papelera archivos de Drive (PB-039).

Grafo real + dobles. Lo que se cuida, además de que anden:
- sólo se toca lo propio (la marca `me` de Drive), y una carpeta sólo se
  renombra: moverla o borrarla arrastraría lo que tiene adentro;
- borrar es mandar a la papelera, siempre con el sí, y la pregunta distingue
  el archivo (tipo y fecha) de otro que se llame igual;
- el nombre nuevo se ve en la pregunta tal como queda en Drive: sin links ni
  « » que el listado quitaría, y con la extensión sólo si es una conocida;
- subir o mover a una carpeta compartida es darle acceso a esa gente: pide el
  sí aunque la acción se pueda deshacer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.application.dto.archivo_adjunto import ArchivoAdjunto
from src.domain.entities.archivo import Archivo
from src.domain.exceptions import (
    DriveLlenoError,
    DriveRechazoError,
    EntityNotFoundError,
    ServiceUnavailableError,
)
from src.domain.value_objects.tipo_de_archivo import TipoDeArchivo
from src.infrastructure.llm.agente_gemini import _aprobacion_de
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import (
    _nombre_con_extension,
    construir_herramientas,
    mensaje_de_la_persona,
)
from tests.dobles import ArchivosFalsos, ModeloFalso, RecibidosFalsos

USUARIO = uuid4()
# Lunes 12 de octubre de 2026, 18:30 en Buenos Aires.
MODIFICADO = datetime(2026, 10, 12, 21, 30, tzinfo=UTC)


def _archivo(**cambios: Any) -> Archivo:
    datos: dict[str, Any] = {
        "id": "f1",
        "nombre": "Presupuesto 2026",
        "tipo": TipoDeArchivo.PLANILLA,
        "modificado": MODIFICADO,
        "enlace": "https://docs.google.com/spreadsheets/d/f1/edit",
        "propio": True,
    }
    datos.update(cambios)
    return Archivo(**datos)


def _carpeta(id_: str, nombre: str, *, compartido: bool = False) -> Archivo:
    return _archivo(
        id=id_, nombre=nombre, tipo=TipoDeArchivo.CARPETA, enlace=None, compartido=compartido
    )


AJENO = _archivo(id="f2", nombre="Notas de Ana", propio=False, propietario="Ana Gómez")


class Escena:
    """El grafo real con los dobles: pide tools y contesta las confirmaciones."""

    def __init__(self, archivos: ArchivosFalsos | None = None) -> None:
        self.archivos = archivos or ArchivosFalsos((_archivo(), AJENO))
        self.recibidos = RecibidosFalsos()
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
        pendiente = self.grafo.get_state(self.configuracion).interrupts[0].value
        estado: dict[str, Any] = await self.grafo.ainvoke(
            Command(resume=_aprobacion_de(pendiente) if aprobado else {"aprobado": False}),
            config=self.configuracion,
            context=self.contexto,
        )
        return estado


def _respuesta(estado: dict[str, Any]) -> str:
    """Lo que contestó la última tool."""
    return str([m for m in estado["messages"] if isinstance(m, ToolMessage)][-1].content)


def _pregunta(estado: dict[str, Any]) -> str:
    return str(estado["__interrupt__"][0].value["resumen"])


def _despues_de_buscar() -> list[BaseMessage]:
    """Un listado de Drive a la vista: el nombre de un archivo lo pudo escribir otro."""
    return [
        HumanMessage("buscá el presupuesto"),
        AIMessage("", tool_calls=[{"name": "buscar_archivos", "args": {}, "id": "b0"}]),
        ToolMessage(
            "1. «Presupuesto 2026» — planilla — id: f1", tool_call_id="b0", name="buscar_archivos"
        ),
        AIMessage("Lo encontré."),
        HumanMessage("hacelo"),
    ]


# --- El nombre nuevo --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pedido", "original", "esperado"),
    [
        ("Factura de luz", "informe.pdf", "Factura de luz.pdf"),
        ("Gastos", "planilla.XLSX", "Gastos.XLSX"),  # la del original, tal cual
        ("Informe.PDF", "informe.pdf", "Informe.PDF"),  # ya la trae
        ("Informe.docx", "informe.pdf", "Informe.docx"),  # trae otra: se respeta
        # Lo que sigue al último punto no siempre es una extensión.
        ("Informe final", "Informe v1.2", "Informe final"),
        ("Dra. Pérez", "Dr. Pérez", "Dra. Pérez"),
        ("Presupuesto final", "Presupuesto 2026", "Presupuesto final"),
        ("  Factura\nde\tluz  ", "informe.pdf", "Factura de luz.pdf"),
        ("", "informe.pdf", "informe.pdf"),
    ],
)
def test_la_extension_se_agrega_solo_si_es_una_conocida(
    pedido: str, original: str, esperado: str
) -> None:
    assert _nombre_con_extension(pedido, original) == esperado


def test_a_una_carpeta_nunca_se_le_agrega_extension() -> None:
    assert _nombre_con_extension("Viaje", "fotos.jpg", TipoDeArchivo.CARPETA) == "Viaje"


# --- Renombrar --------------------------------------------------------------------------


async def test_renombrar_un_archivo_propio_sale_directo() -> None:
    """Se deshace (`MODIFICAR`): sin texto de terceros a la vista, no pausa."""
    escena = Escena()

    estado = await escena.pedir(
        ("renombrar_archivo", {"id_archivo": "f1", "nombre_nuevo": "Presupuesto final"})
    )

    assert "__interrupt__" not in estado
    assert escena.archivos.renombrados == [(USUARIO, "f1", "Presupuesto final")]
    texto = _respuesta(estado)
    assert texto == "Renombré «Presupuesto 2026»: ahora se llama «Presupuesto final»."
    assert escena.contexto.acciones_realizadas == [texto]


async def test_con_un_listado_de_drive_a_la_vista_renombrar_pide_el_si() -> None:
    """El caso de todos los días: el id sale de buscar_archivos (modo estricto)."""
    escena = Escena()

    estado = await escena.pedir(
        ("renombrar_archivo", {"id_archivo": "f1", "nombre_nuevo": "Presupuesto final"}),
        historial=_despues_de_buscar(),
    )

    assert _pregunta(estado) == "Renombrar «Presupuesto 2026» (planilla) a «Presupuesto final»"
    assert escena.archivos.renombrados == []
    await escena.decidir(True)
    assert escena.archivos.renombrados == [(USUARIO, "f1", "Presupuesto final")]


async def test_una_carpeta_se_renombra_sin_extension() -> None:
    escena = Escena(ArchivosFalsos(carpetas=(_carpeta("c1", "Fotos.jpg"),)))

    await escena.pedir(("renombrar_archivo", {"id_archivo": "c1", "nombre_nuevo": "Viaje"}))

    assert escena.archivos.renombrados == [(USUARIO, "c1", "Viaje")]


@pytest.mark.parametrize(
    ("nombre_nuevo", "esperado"),
    [
        (" \n ", "Falta el nombre nuevo"),
        ("Ver https://sitio-falso.example/x", "no puede tener links"),
        ("Presupuesto sitio-falso.com", "no puede tener links"),
        ("Presupuesto «final»", "ni « »"),
        ("x" * 201, "Ese nombre es muy largo"),
    ],
)
async def test_un_nombre_invalido_se_explica_antes_de_hablar_con_drive(
    nombre_nuevo: str, esperado: str
) -> None:
    """Con Drive caído, la respuesta igual es la del nombre: ni se le preguntó."""
    escena = Escena(ArchivosFalsos((_archivo(),), fallar_con=ServiceUnavailableError("caído")))

    estado = await escena.pedir(
        ("renombrar_archivo", {"id_archivo": "f1", "nombre_nuevo": nombre_nuevo})
    )

    assert "__interrupt__" not in estado
    assert esperado in _respuesta(estado)
    assert escena.archivos.renombrados == []


async def test_con_el_mismo_nombre_no_cambia_nada() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("renombrar_archivo", {"id_archivo": "f1", "nombre_nuevo": " Presupuesto  2026 "})
    )

    assert _respuesta(estado) == "«Presupuesto 2026» ya se llama así: no cambié nada."
    assert escena.archivos.renombrados == []


async def test_cambiar_solo_las_mayusculas_si_es_renombrar() -> None:
    escena = Escena()

    await escena.pedir(
        ("renombrar_archivo", {"id_archivo": "f1", "nombre_nuevo": "PRESUPUESTO 2026"})
    )

    assert escena.archivos.renombrados == [(USUARIO, "f1", "PRESUPUESTO 2026")]


# --- Lo que no es de la persona no se toca ------------------------------------------------

AJENAS = [
    ("renombrar_archivo", {"id_archivo": "f2", "nombre_nuevo": "Mías"}),
    ("mover_archivo", {"id_archivo": "f2", "carpeta": "Mi unidad"}),
    ("borrar_archivo", {"id_archivo": "f2"}),
]


@pytest.mark.parametrize(("nombre", "args"), AJENAS)
async def test_un_archivo_ajeno_se_rechaza_antes_de_preguntar(
    nombre: str, args: dict[str, Any]
) -> None:
    """Un editor podría, pero el archivo es de otro, que no lo pidió."""
    escena = Escena()

    estado = await escena.pedir((nombre, args))

    assert "__interrupt__" not in estado
    texto = _respuesta(estado)
    assert texto.startswith("«Notas de Ana» es de «Ana Gómez», no tuyo: no puedo")
    assert escena.archivos.renombrados == []
    assert escena.archivos.movidos == []
    assert escena.archivos.en_la_papelera == []


async def test_sin_la_marca_de_drive_tampoco_se_toca() -> None:
    """Drive no dijo de quién es: no se presume que es de la persona."""
    sin_dueno = _archivo(propio=False, propietario=None)
    escena = Escena(ArchivosFalsos((sin_dueno,)))

    estado = await escena.pedir(("borrar_archivo", {"id_archivo": "f1"}))

    assert "Drive no confirma que «Presupuesto 2026» sea tuyo" in _respuesta(estado)
    assert escena.archivos.en_la_papelera == []


@pytest.mark.parametrize(
    ("nombre", "args"),
    [
        ("mover_archivo", {"id_archivo": "c1", "carpeta": "Mi unidad"}),
        ("borrar_archivo", {"id_archivo": "c1"}),
    ],
)
async def test_una_carpeta_no_se_mueve_ni_se_borra(nombre: str, args: dict[str, Any]) -> None:
    escena = Escena(ArchivosFalsos(carpetas=(_carpeta("c1", "Facultad"),)))

    estado = await escena.pedir((nombre, args))

    assert "__interrupt__" not in estado
    assert "«Facultad» es una carpeta" in _respuesta(estado)
    assert escena.archivos.movidos == []
    assert escena.archivos.en_la_papelera == []


# --- Mover --------------------------------------------------------------------------------


async def test_mover_a_una_carpeta_propia_por_su_nombre() -> None:
    escena = Escena(ArchivosFalsos((_archivo(),), carpetas=(_carpeta("c1", "Facultad"),)))

    estado = await escena.pedir(("mover_archivo", {"id_archivo": "f1", "carpeta": "facultad"}))

    assert "__interrupt__" not in estado
    assert escena.archivos.movidos == [(USUARIO, "f1", "c1")]
    texto = _respuesta(estado)
    # "Quedó": si ya estaba ahí, Drive no cambia nada y la frase sigue siendo cierta.
    assert texto == "Listo: «Presupuesto 2026» quedó en la carpeta «Facultad»."
    assert escena.contexto.acciones_realizadas == [texto]


@pytest.mark.parametrize("carpeta", ["Mi unidad", " mi   UNIDAD ", "My Drive", "raíz", "Mi drive"])
async def test_mi_unidad_y_sus_alias(carpeta: str) -> None:
    escena = Escena()

    estado = await escena.pedir(("mover_archivo", {"id_archivo": "f1", "carpeta": carpeta}))

    assert escena.archivos.movidos == [(USUARIO, "f1", None)]
    assert _respuesta(estado) == "Listo: «Presupuesto 2026» quedó en Mi unidad."


async def test_sin_carpeta_pregunta_adonde() -> None:
    escena = Escena()

    estado = await escena.pedir(("mover_archivo", {"id_archivo": "f1", "carpeta": " "}))

    assert _respuesta(estado).startswith("Falta adónde")
    assert escena.archivos.movidos == []


async def test_a_una_carpeta_que_no_existe_no_se_mueve() -> None:
    escena = Escena(ArchivosFalsos((_archivo(),), carpetas=(_carpeta("c1", "Facultad"),)))

    estado = await escena.pedir(("mover_archivo", {"id_archivo": "f1", "carpeta": "Trabajo"}))

    assert _respuesta(estado) == (
        "No encontré una carpeta tuya que se llame «Trabajo». ¿Me decís otra, o va a Mi unidad?"
    )
    assert escena.archivos.movidos == []


class _SinCarpetas(ArchivosFalsos):
    """Drive lee el archivo, pero se cae al buscar la carpeta."""

    async def carpetas(self, usuario_id: UUID, nombre: str) -> tuple[Archivo, ...]:
        raise ServiceUnavailableError("Drive no responde")


async def test_si_drive_se_cae_al_buscar_la_carpeta_no_mueve_nada() -> None:
    escena = Escena(_SinCarpetas((_archivo(),)))

    estado = await escena.pedir(("mover_archivo", {"id_archivo": "f1", "carpeta": "Facultad"}))

    assert "__interrupt__" not in estado
    assert _respuesta(estado).startswith("No pude moverlo ahora mismo")
    assert escena.archivos.movidos == []


async def test_a_una_carpeta_compartida_pide_el_si_aunque_se_pueda_deshacer() -> None:
    """Moverlo ahí le da acceso a quienes ven la carpeta: es compartir."""
    carpeta = _carpeta("c1", "Equipo", compartido=True)
    escena = Escena(ArchivosFalsos((_archivo(),), carpetas=(carpeta,)))

    estado = await escena.pedir(("mover_archivo", {"id_archivo": "f1", "carpeta": "Equipo"}))

    assert _pregunta(estado) == (
        "Mover «Presupuesto 2026» (planilla) a la carpeta «Equipo»: esa carpeta está "
        "compartida, así que quienes la ven van a poder ver el archivo"
    )
    assert escena.archivos.movidos == []
    await escena.decidir(True)
    assert escena.archivos.movidos == [(USUARIO, "f1", "c1")]


async def test_con_un_no_no_se_mueve_nada() -> None:
    carpeta = _carpeta("c1", "Equipo", compartido=True)
    escena = Escena(ArchivosFalsos((_archivo(),), carpetas=(carpeta,)))

    await escena.pedir(("mover_archivo", {"id_archivo": "f1", "carpeta": "Equipo"}))
    estado = await escena.decidir(False)

    assert escena.archivos.movidos == []
    assert "No se movió nada" in _respuesta(estado)


async def test_subir_a_una_carpeta_compartida_tambien_pide_el_si() -> None:
    """La corrección de PB-038: antes subía ahí sin preguntar."""
    carpeta = _carpeta("c1", "Equipo", compartido=True)
    escena = Escena(ArchivosFalsos(carpetas=(carpeta,)))
    informe = ArchivoAdjunto(referencia="1037", tipo_mime="application/pdf", nombre="informe.pdf")

    estado = await escena.pedir(
        ("subir_archivo", {"carpeta": "Equipo"}),
        historial=[mensaje_de_la_persona("", informe, datetime.now(UTC))],
    )

    assert "esa carpeta está compartida" in _pregunta(estado)
    assert escena.recibidos.descargas == []  # nada se baja antes del sí
    await escena.decidir(True)
    assert escena.archivos.subidos[0][4] == "c1"


# --- Mandar a la papelera ---------------------------------------------------------------


async def test_borrar_siempre_pide_el_si_con_el_tipo_y_la_fecha() -> None:
    """El tipo y la fecha distinguen dos archivos que se llaman igual."""
    escena = Escena()

    estado = await escena.pedir(("borrar_archivo", {"id_archivo": "f1"}))

    assert _pregunta(estado) == (
        "Mandar a la papelera «Presupuesto 2026» (planilla, modificado el lunes 12 de "
        "octubre, 18:30). Se puede recuperar desde la papelera de Drive durante 30 días."
    )
    assert escena.archivos.en_la_papelera == []
    estado = await escena.decidir(True)
    assert escena.archivos.en_la_papelera == [(USUARIO, "f1")]
    texto = _respuesta(estado)
    assert texto == (
        "Mandé «Presupuesto 2026» a la papelera de tu Drive: se puede recuperar desde ahí "
        "durante 30 días."
    )
    assert escena.contexto.acciones_realizadas == [texto]


async def test_con_un_no_no_se_borra_nada() -> None:
    escena = Escena()

    await escena.pedir(("borrar_archivo", {"id_archivo": "f1"}))
    estado = await escena.decidir(False)

    assert escena.archivos.en_la_papelera == []
    assert "No se borró nada" in _respuesta(estado)
    assert escena.contexto.acciones_realizadas == []


async def test_dos_borrados_se_confirman_en_una_sola_lista() -> None:
    otro = _archivo(id="f3", nombre="Presupuesto 2025")
    escena = Escena(ArchivosFalsos((_archivo(), otro)))

    estado = await escena.pedir(
        ("borrar_archivo", {"id_archivo": "f1"}), ("borrar_archivo", {"id_archivo": "f3"})
    )

    assert _pregunta(estado).startswith("Perfecto, hago esto:\n• Mandar a la papelera")
    assert "«Presupuesto 2025»" in _pregunta(estado)
    await escena.decidir(True)
    assert escena.archivos.en_la_papelera == [(USUARIO, "f1"), (USUARIO, "f3")]


# --- Lo que contesta Drive después del sí ----------------------------------------------------


@pytest.mark.parametrize(
    ("falla", "esperado"),
    [
        (DriveRechazoError(), "Drive no confirmó que se haya hecho"),
        (DriveLlenoError(), "Tu Google Drive está lleno y no dejó mandarlo a la papelera"),
        (EntityNotFoundError("404"), "no hizo falta mandarlo a la papelera"),
    ],
)
async def test_si_drive_no_deja_lo_dice_y_no_entra_en_la_bitacora(
    falla: Exception, esperado: str
) -> None:
    escena = Escena(ArchivosFalsos((_archivo(),), fallar_al_escribir=falla))

    await escena.pedir(("borrar_archivo", {"id_archivo": "f1"}))
    estado = await escena.decidir(True)

    assert esperado in _respuesta(estado)
    assert escena.contexto.acciones_realizadas == []
