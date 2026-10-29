"""Tests de las escrituras en Notion (RF-07): grafo real + dobles.

Lo que se cuida, además de que anden:
- toda escritura pide el sí, también sola y sin texto de terceros a la vista:
  lo que se escribe lo ve quien ve la página (`comparte=True`);
- la pregunta muestra entero lo que se va a escribir, y lo que viene de Notion
  (títulos, nombres, opciones) llega saneado;
- el texto pasa a bloques propios sin lanzar nunca, y los datos de una fila
  sólo toman valores que se entienden sin dudas y opciones que ya existen;
- las bases no se mueven ni se borran, y las filas no salen de su base;
- si Notion no confirmó, no se dice que se hizo, ni se invita a duplicarlo;
- una base nueva se confirma entera (columnas y filas) en una pregunta, y si
  sus filas se cortan a mitad, la base queda contada y no se la pide de nuevo.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.application.dto.archivo_adjunto import ArchivoAdjunto
from src.application.ports.paginas import FilaDeBase, Propiedad, TipoDeBloque, TipoDePropiedad
from src.domain.entities.pagina import Pagina
from src.domain.exceptions import (
    ArchivoNoDisponibleError,
    AutorizacionFallidaError,
    EntityNotFoundError,
    EscrituraEnDudaError,
    NotionRechazoError,
    NotionSinPermisoError,
    ServiceUnavailableError,
)
from src.domain.value_objects.tipo_de_pagina import TipoDePagina
from src.infrastructure.llm.agente_gemini import _aprobacion_de
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import LOTE_DEMASIADO_LARGO, construir_grafo
from src.infrastructure.llm.herramientas import (
    _bloques,
    _como_queda,
    construir_herramientas,
    mensaje_de_la_persona,
)
from tests.dobles import ModeloFalso, PaginasFalsas, RecibidosFalsos

USUARIO = uuid4()
PAGINA_ID = "1f2e3d4c-5b6a-4789-8abc-def012345678"
DESTINO_ID = "0a1b2c3d-4e5f-4607-8899-aabbccddeeff"
BASE_ID = "9a8b7c6d-5e4f-4321-9fed-cba987654321"
FILA_ID = "2b3c4d5e-6f70-4812-93a4-b5c6d7e8f901"
# Lunes 12 de octubre de 2026, 18:30 en Buenos Aires.
EDITADA = datetime(2026, 10, 12, 21, 30, tzinfo=UTC)
LINK = f"https://www.notion.so/Recetas-{PAGINA_ID.replace('-', '')}"
INFORME = ArchivoAdjunto(
    referencia="1037543291543636", tipo_mime="application/pdf", nombre="informe.pdf"
)
DATOS_DE_LA_FILA = (
    Propiedad(id="title", nombre="Concepto", tipo=TipoDePropiedad.TITULO, valor="Luz"),
    Propiedad(id="m1", nombre="Monto", tipo=TipoDePropiedad.NUMERO, valor="1500"),
    Propiedad(id="p1", nombre="Pagado", tipo=TipoDePropiedad.CASILLA, valor="no"),
    Propiedad(id="v1", nombre="Vence", tipo=TipoDePropiedad.FECHA, valor="2026-11-10"),
    Propiedad(
        id="e1",
        nombre="Estado",
        tipo=TipoDePropiedad.ESTADO,
        valor="Pendiente",
        opciones=("Pendiente", "Pagado"),
    ),
    Propiedad(
        id="r1", nombre="Rubro", tipo=TipoDePropiedad.SELECCION, opciones=("Servicios", "Comida")
    ),
    Propiedad(id="n1", nombre="Notas", tipo=TipoDePropiedad.TEXTO),
)


def _pagina(**cambios: Any) -> Pagina:
    datos: dict[str, Any] = {
        "id": PAGINA_ID,
        "titulo": "Recetas",
        "tipo": TipoDePagina.PAGINA,
        "editada": EDITADA,
        "enlace": LINK,
    }
    datos.update(cambios)
    return Pagina(**datos)


def _notion(**opciones: Any) -> PaginasFalsas:
    """Una página, otra de destino, una base y una fila con datos."""
    return PaginasFalsas(
        (
            _pagina(),
            _pagina(id=DESTINO_ID, titulo="Archivo"),
            _pagina(id=BASE_ID, titulo="Gastos", tipo=TipoDePagina.BASE),
            _pagina(id=FILA_ID, titulo="Luz", en_base=True),
        ),
        **{
            "propiedades": {FILA_ID: DATOS_DE_LA_FILA},
            "esquemas": {BASE_ID: DATOS_DE_LA_FILA},
            **opciones,
        },
    )


def _recibido(adjunto: ArchivoAdjunto = INFORME) -> HumanMessage:
    return mensaje_de_la_persona("", adjunto, datetime.now(UTC))


class Escena:
    """El grafo real con los dobles: pide tools y contesta las confirmaciones."""

    def __init__(
        self, paginas: PaginasFalsas | None = None, recibidos: RecibidosFalsos | None = None
    ) -> None:
        self.paginas = paginas or _notion()
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
        herramientas = construir_herramientas(None, paginas=self.paginas, recibidos=self.recibidos)
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
    """El resumen de la confirmación pendiente."""
    return str(estado["__interrupt__"][0].value["resumen"])


# --- El texto, en bloques ---------------------------------------------------------------


def test_cada_marca_es_su_bloque() -> None:
    texto = (
        "# Compras\n## Súper\n### Verdulería\n- [ ] Harina\n- [x] Huevos\n- Leche\n"
        "* Pan\n• Queso\n1. Paso uno\n12. Paso doce\nUn párrafo"
    )

    assert [(b.tipo, b.texto, b.hecho) for b in _bloques(texto)] == [
        (TipoDeBloque.TITULO_1, "Compras", False),
        (TipoDeBloque.TITULO_2, "Súper", False),
        (TipoDeBloque.TITULO_3, "Verdulería", False),
        (TipoDeBloque.TAREA, "Harina", False),
        (TipoDeBloque.TAREA, "Huevos", True),
        (TipoDeBloque.VINETA, "Leche", False),
        (TipoDeBloque.VINETA, "Pan", False),
        (TipoDeBloque.VINETA, "Queso", False),
        (TipoDeBloque.NUMERADO, "Paso uno", False),
        (TipoDeBloque.NUMERADO, "Paso doce", False),
        (TipoDeBloque.PARRAFO, "Un párrafo", False),
    ]


@pytest.mark.parametrize(
    "linea", ["#hashtag", "####  cuatro", "². no numera", "1.5 kg", "-sin espacio", "**negrita**"]
)
def test_lo_que_no_es_una_marca_va_como_parrafo(linea: str) -> None:
    [bloque] = _bloques(linea)

    assert (bloque.tipo, bloque.texto) == (TipoDeBloque.PARRAFO, " ".join(linea.split()))


def test_las_lineas_vacias_se_omiten_y_los_espacios_se_juntan() -> None:
    assert [b.texto for b in _bloques("\n\n  uno   dos \n\t\n tres")] == ["uno dos", "tres"]


def test_la_pregunta_muestra_cada_linea_como_queda() -> None:
    bloques = _bloques("# Título\n- [ ] a\n- [x] b\n- c\n1. d\n7. e\nf\n1. g")

    assert _como_queda(bloques) == "# Título\n☐ a\n☑ b\n• c\n1. d\n2. e\nf\n1. g"


# --- Crear ----------------------------------------------------------------------------------


async def test_crear_pide_el_si_aunque_sea_una_sola_y_muestra_el_texto() -> None:
    escena = Escena()

    estado = await escena.pedir(
        (
            "crear_pagina_en_notion",
            {
                "id_padre": PAGINA_ID,
                "titulo": "Torta",
                "texto": "## Ingredientes\n- [ ] Harina\nPara el domingo",
            },
        )
    )

    assert _pregunta(estado) == (
        "Crear en tu Notion la página «Torta» adentro de «Recetas», con este texto:\n\n"
        "## Ingredientes\n☐ Harina\nPara el domingo"
    )
    assert escena.paginas.escrituras() == 0
    estado = await escena.decidir(True)
    [(usuario, padre, titulo, bloques)] = escena.paginas.creadas
    assert (usuario, padre, titulo) == (USUARIO, PAGINA_ID, "Torta")
    assert [b.tipo for b in bloques] == [
        TipoDeBloque.TITULO_2,
        TipoDeBloque.TAREA,
        TipoDeBloque.PARRAFO,
    ]
    texto = _respuesta(estado)
    assert texto.startswith("Creé la página «Torta» adentro de «Recetas» (id: ")
    assert texto.endswith(" — link: https://www.notion.so/nueva")
    assert escena.contexto.acciones_realizadas == [texto]


async def test_crear_en_una_base_es_una_fila_nueva() -> None:
    escena = Escena()

    estado = await escena.pedir(("crear_pagina_en_notion", {"id_padre": BASE_ID, "titulo": "Gas"}))

    assert _pregunta(estado) == (
        "Crear en tu Notion la página «Gas» como fila nueva de la base «Gastos»"
    )
    await escena.decidir(True)
    [(_, padre, _, bloques)] = escena.paginas.creadas
    assert (padre, bloques) == (BASE_ID, ())


@pytest.mark.parametrize(
    ("args", "esperado"),
    [
        ({"titulo": ""}, "Falta el título"),
        ({"titulo": "Ver https://sitio.example"}, "no puede tener links"),
        ({"titulo": "Mirá sitio-falso.com"}, "no puede tener links"),
        ({"titulo": "«Torta»"}, "ni « »"),
        ({"titulo": "x" * 201}, "muy largo"),
        ({"titulo": "Torta", "texto": "x" * 3001}, "el máximo es 3000"),
        ({"titulo": "Torta", "texto": "\n".join(["- a"] * 101)}, "hasta 100"),
        ({"titulo": "Torta", "id_padre": str(uuid4())}, "compartilo con LifeSync"),
    ],
)
async def test_crear_lo_invalido_se_explica_sin_pausa(args: dict[str, Any], esperado: str) -> None:
    escena = Escena()

    estado = await escena.pedir(("crear_pagina_en_notion", {"id_padre": PAGINA_ID, **args}))

    assert "__interrupt__" not in estado
    assert esperado in _respuesta(estado)
    assert escena.paginas.escrituras() == 0


# --- Agregar ----------------------------------------------------------------------------


async def test_agregar_muestra_lo_que_suma_y_va_al_final() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("agregar_a_pagina_de_notion", {"id_pagina": PAGINA_ID, "texto": "- Leche\n- Pan"})
    )

    assert _pregunta(estado) == "Agregar al final de «Recetas», en tu Notion:\n\n• Leche\n• Pan"
    assert escena.paginas.escrituras() == 0
    estado = await escena.decidir(True)
    [(_, pagina, bloques)] = escena.paginas.agregados
    assert (pagina, [b.texto for b in bloques]) == (PAGINA_ID, ["Leche", "Pan"])
    assert _respuesta(estado) == "Agregué el texto al final de «Recetas», en tu Notion."


async def test_con_un_no_no_se_agrega_nada() -> None:
    escena = Escena()
    await escena.pedir(("agregar_a_pagina_de_notion", {"id_pagina": PAGINA_ID, "texto": "x"}))

    estado = await escena.decidir(False)

    assert _respuesta(estado) == "La persona lo canceló. No se agregó nada."
    assert escena.paginas.escrituras() == 0


async def test_a_una_base_no_se_le_agrega_texto() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("agregar_a_pagina_de_notion", {"id_pagina": BASE_ID, "texto": "x"})
    )

    assert "es una base, no una página" in _respuesta(estado)
    assert "__interrupt__" not in estado


# --- Cambiar un dato --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dato", "valor", "enviado", "como_se_ve"),
    [
        ("Estado", "pagado", "Pagado", "a «Pagado» (ahora: «Pendiente»)"),
        ("estado", "PAGADO", "Pagado", "a «Pagado» (ahora: «Pendiente»)"),
        ("Monto", "1.500,50", 1500.5, "a 1.500,5 (ahora: «1500»)"),
        ("Monto", "$2000", 2000.0, "a 2.000 (ahora: «1500»)"),
        ("Monto", "0.500", 0.5, "a 0,5 (ahora: «1500»)"),
        ("Pagado", "sí", True, "a marcada (ahora: «no»)"),
        ("Vence", "2026-12-01", date(2026, 12, 1), "a martes 1 de diciembre de 2026"),
        ("Rubro", "servicios", "Servicios", "a «Servicios» (ahora está vacío)"),
        ("Notas", "Pagar online", "Pagar online", "a «Pagar online» (ahora está vacío)"),
    ],
)
async def test_cambiar_un_dato_muestra_el_valor_nuevo_y_el_actual(
    dato: str, valor: str, enviado: object, como_se_ve: str
) -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("cambiar_dato_en_notion", {"id_pagina": FILA_ID, "dato": dato, "valor": valor})
    )

    pregunta = _pregunta(estado)
    assert pregunta.startswith("Cambiar «") and "» de «Luz» en tu Notion " in pregunta
    assert como_se_ve in pregunta
    await escena.decidir(True)
    [(_, pagina, _, nuevo)] = escena.paginas.cambios
    assert (pagina, nuevo) == (FILA_ID, enviado)


async def test_se_escribe_por_el_id_del_dato() -> None:
    escena = Escena()
    await escena.pedir(
        ("cambiar_dato_en_notion", {"id_pagina": FILA_ID, "dato": "Estado", "valor": "Pagado"})
    )

    estado = await escena.decidir(True)

    [(_, _, propiedad, _)] = escena.paginas.cambios
    assert propiedad == "e1"
    assert _respuesta(estado) == "Cambié «Estado» de «Luz» a «Pagado», en tu Notion."


async def test_titulo_elige_el_titulo_se_llame_como_se_llame_la_columna() -> None:
    escena = Escena()

    estado = await escena.pedir(
        (
            "cambiar_dato_en_notion",
            {"id_pagina": FILA_ID, "dato": "título", "valor": "Luz de octubre"},
        )
    )

    assert _pregunta(estado) == (
        "Cambiarle el título a «Luz» en tu Notion: pasa a llamarse «Luz de octubre»"
    )
    estado = await escena.decidir(True)
    [(_, _, propiedad, nuevo)] = escena.paginas.cambios
    assert (propiedad, nuevo) == ("title", "Luz de octubre")
    assert _respuesta(estado) == "Le cambié el título a «Luz»: ahora se llama «Luz de octubre»."


async def test_un_nombre_exacto_gana_sobre_titulo() -> None:
    datos = (
        Propiedad(id="title", nombre="Nombre", tipo=TipoDePropiedad.TITULO, valor="Luz"),
        Propiedad(id="t2", nombre="Título", tipo=TipoDePropiedad.TEXTO),
    )
    escena = Escena(_notion(propiedades={FILA_ID: datos}))
    await escena.pedir(
        ("cambiar_dato_en_notion", {"id_pagina": FILA_ID, "dato": "titulo", "valor": "Factura"})
    )

    await escena.decidir(True)

    [(_, _, propiedad, _)] = escena.paginas.cambios
    assert propiedad == "t2"


async def test_renombrar_una_pagina_suelta() -> None:
    escena = Escena()

    estado = await escena.pedir(
        (
            "cambiar_dato_en_notion",
            {"id_pagina": PAGINA_ID, "dato": "título", "valor": "Recetas de la abuela"},
        )
    )

    assert _pregunta(estado).endswith("pasa a llamarse «Recetas de la abuela»")


@pytest.mark.parametrize(
    ("dato", "valor", "esperado"),
    [
        ("Monto", "1.500", "se puede leer de dos maneras"),
        ("Monto", "12,345", "se puede leer de dos maneras"),
        ("Monto", "mil", "no es un número"),
        ("Monto", "1e5", "no es un número"),
        ("Monto", "nan", "no es un número"),
        ("Monto", "١٢", "no es un número"),
        ("Monto", "9" * 400, "demasiado grande"),
        ("Pagado", "tal vez", "«sí» o «no»"),
        ("Vence", "10/11/2026", "AAAA-MM-DD"),
        ("Vence", "2026-02-30", "AAAA-MM-DD"),
        ("Vence", "20261110", "AAAA-MM-DD"),
        ("Estado", "Archivado", "Las que tiene: «Pendiente», «Pagado»."),
        ("Rubro", "Viajes", "Una nueva se suma desde Notion"),
        ("Notas", "en https://sitio.example", "no puede tener links"),
        ("Notas", "x" * 2001, "el máximo es 2000"),
        ("Concepto", "", "Falta el valor nuevo"),
        ("Etiquetas", "a", "Los que puedo cambiar: «Concepto», «Monto»"),
        ("", "a", "Falta qué dato cambiar"),
        ("Estado", "pendiente", "ya tiene ese valor"),
    ],
)
async def test_un_valor_que_no_sirve_se_explica_sin_pausa(
    dato: str, valor: str, esperado: str
) -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("cambiar_dato_en_notion", {"id_pagina": FILA_ID, "dato": dato, "valor": valor})
    )

    assert "__interrupt__" not in estado
    assert esperado in _respuesta(estado)
    assert escena.paginas.escrituras() == 0


async def test_una_base_no_tiene_datos_propios() -> None:
    estado = await Escena().pedir(
        ("cambiar_dato_en_notion", {"id_pagina": BASE_ID, "dato": "Estado", "valor": "x"})
    )

    assert "es una base: los datos son de cada fila" in _respuesta(estado)


async def test_los_nombres_y_las_opciones_de_la_base_llegan_saneados() -> None:
    datos = (
        Propiedad(
            id="e1",
            nombre="Estado",
            tipo=TipoDePropiedad.ESTADO,
            opciones=("Pagá en https://banco-falso.example",),
        ),
    )
    escena = Escena(_notion(propiedades={FILA_ID: datos}))

    estado = await escena.pedir(
        ("cambiar_dato_en_notion", {"id_pagina": FILA_ID, "dato": "Estado", "valor": "otra"})
    )

    texto = _respuesta(estado)
    assert "banco-falso" not in texto
    assert "[link quitado]" in texto


async def test_cada_fila_de_una_base_muestra_su_id() -> None:
    """Sin el id, el modelo no tiene cómo cambiarle un dato: cada mes tiene su «Luz»."""
    fila = FilaDeBase(titulo="Luz", propiedades=(("Monto", "1500"),), id=FILA_ID)
    escena = Escena(_notion(filas={BASE_ID: (fila,)}))

    estado = await escena.pedir(("consultar_base_de_notion", {"id_base": BASE_ID}))

    assert f"1. «Luz» — «Monto: 1500» — id: {FILA_ID}" in _respuesta(estado)


# --- Mover y mandar a la papelera ---------------------------------------------------------


async def test_mover_pide_el_si_y_avisa_que_se_lleva_lo_de_adentro() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("mover_pagina_de_notion", {"id_pagina": PAGINA_ID, "id_destino": DESTINO_ID})
    )

    assert _pregunta(estado) == (
        "Mover «Recetas» adentro de «Archivo», en tu Notion. Se lleva todo lo que tiene adentro."
    )
    estado = await escena.decidir(True)
    assert escena.paginas.movidas == [(USUARIO, PAGINA_ID, DESTINO_ID)]
    assert _respuesta(estado) == "Moví «Recetas» adentro de «Archivo», en tu Notion."


@pytest.mark.parametrize(
    ("args", "esperado"),
    [
        ({"id_pagina": BASE_ID, "id_destino": DESTINO_ID}, "moverla se llevaría todas sus filas"),
        ({"id_pagina": FILA_ID, "id_destino": DESTINO_ID}, "es una fila de una base"),
        ({"id_pagina": PAGINA_ID, "id_destino": BASE_ID}, "no de una base"),
        ({"id_pagina": PAGINA_ID, "id_destino": PAGINA_ID}, "adentro de sí misma"),
        ({"id_pagina": PAGINA_ID, "id_destino": str(uuid4())}, "compartilo con LifeSync"),
    ],
)
async def test_lo_que_no_se_mueve_se_explica_sin_pausa(args: dict[str, Any], esperado: str) -> None:
    escena = Escena()

    estado = await escena.pedir(("mover_pagina_de_notion", args))

    assert "__interrupt__" not in estado
    assert esperado in _respuesta(estado)
    assert escena.paginas.escrituras() == 0


async def test_borrar_pide_el_si_y_dice_que_se_recupera() -> None:
    escena = Escena()

    estado = await escena.pedir(("borrar_pagina_de_notion", {"id_pagina": PAGINA_ID}))

    assert _pregunta(estado) == (
        "Mandar a la papelera de tu Notion la página «Recetas» (editada el lunes 12 de "
        "octubre, 18:30), con todo lo que tiene adentro. Se puede recuperar desde la "
        "papelera de Notion."
    )
    estado = await escena.decidir(True)
    assert escena.paginas.en_la_papelera == [(USUARIO, PAGINA_ID)]
    assert _respuesta(estado) == (
        "Mandé la página «Recetas» a la papelera de tu Notion: se puede recuperar desde ahí."
    )


async def test_una_fila_va_a_la_papelera_como_fila() -> None:
    estado = await Escena().pedir(("borrar_pagina_de_notion", {"id_pagina": FILA_ID}))

    assert _pregunta(estado).startswith("Mandar a la papelera de tu Notion la fila «Luz»")


async def test_con_un_no_no_se_borra_nada() -> None:
    escena = Escena()
    await escena.pedir(("borrar_pagina_de_notion", {"id_pagina": PAGINA_ID}))

    estado = await escena.decidir(False)

    assert _respuesta(estado) == "La persona lo canceló. No se borró nada."
    assert escena.paginas.escrituras() == 0


async def test_una_base_no_se_borra() -> None:
    estado = await Escena().pedir(("borrar_pagina_de_notion", {"id_pagina": BASE_ID}))

    assert "borrarla se llevaría todas sus filas" in _respuesta(estado)
    assert "__interrupt__" not in estado


# --- Subir un archivo ---------------------------------------------------------------------


async def test_subir_no_baja_nada_antes_del_si() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("subir_a_notion", {"id_pagina": PAGINA_ID}), historial=[_recibido()]
    )

    assert _pregunta(estado) == "Subir «informe.pdf» al final de «Recetas», en tu Notion"
    assert escena.recibidos.descargas == []
    estado = await escena.decidir(True)
    [(usuario, pagina, nombre, tipo, contenido)] = escena.paginas.subidos
    assert (usuario, pagina, nombre, tipo) == (
        USUARIO,
        PAGINA_ID,
        "informe.pdf",
        "application/pdf",
    )
    assert contenido == escena.recibidos.contenido
    assert _respuesta(estado) == (
        f"Subí «informe.pdf» al final de «Recetas», en tu Notion — link: {LINK}"
    )


async def test_mas_grande_que_lo_que_acepta_el_espacio_lo_explica() -> None:
    """En el plan gratis de Notion, 5 MB por archivo."""
    escena = Escena(recibidos=RecibidosFalsos(contenido=b"x" * (5 * 1024 * 1024 + 1)))
    await escena.pedir(("subir_a_notion", {"id_pagina": PAGINA_ID}), historial=[_recibido()])

    estado = await escena.decidir(True)

    assert "pesa más de 5 MB, que es lo máximo que tu espacio de Notion acepta" in (
        _respuesta(estado)
    )
    assert escena.paginas.subidos == []


async def test_si_notion_no_dice_su_limite_vale_el_de_siempre() -> None:
    escena = Escena(_notion(limite=None), RecibidosFalsos(contenido=b"x" * (6 * 1024 * 1024)))
    await escena.pedir(("subir_a_notion", {"id_pagina": PAGINA_ID}), historial=[_recibido()])

    await escena.decidir(True)

    assert len(escena.paginas.subidos) == 1


async def test_sin_un_archivo_en_la_conversacion_pide_que_lo_mande() -> None:
    estado = await Escena().pedir(("subir_a_notion", {"id_pagina": PAGINA_ID}))

    assert _respuesta(estado).startswith("No tengo ningún archivo tuyo para subir")


@pytest.mark.parametrize(
    ("falla", "esperado"),
    [
        # Es un EntityNotFoundError: no es "la página ya no está".
        (ArchivoNoDisponibleError(), "mandámelo de nuevo"),
        (ServiceUnavailableError("caído"), "No pude traer el archivo de WhatsApp"),
    ],
)
async def test_si_no_se_puede_bajar_de_whatsapp_lo_dice(falla: Exception, esperado: str) -> None:
    escena = Escena(recibidos=RecibidosFalsos(fallar_con=falla))
    await escena.pedir(("subir_a_notion", {"id_pagina": PAGINA_ID}), historial=[_recibido()])

    estado = await escena.decidir(True)

    assert esperado in _respuesta(estado)
    assert escena.paginas.subidos == []


@pytest.mark.parametrize(
    ("falla", "esperado"),
    [
        (NotionSinPermisoError(), "/conectar notion"),
        (NotionRechazoError(), "el tipo o el tamaño del archivo"),
        (EscrituraEnDudaError(), "así no queda repetido"),
        (EntityNotFoundError(), "Esa página ya no está en tu Notion"),
    ],
)
async def test_si_notion_falla_al_subir_lo_dice(falla: Exception, esperado: str) -> None:
    escena = Escena(_notion(fallar_al_escribir=falla))
    await escena.pedir(("subir_a_notion", {"id_pagina": PAGINA_ID}), historial=[_recibido()])

    estado = await escena.decidir(True)

    assert esperado in _respuesta(estado)
    assert escena.contexto.acciones_realizadas == []


# --- Las fallas después del sí --------------------------------------------------------------

CREAR = ("crear_pagina_en_notion", {"id_padre": PAGINA_ID, "titulo": "Torta"})
AGREGAR = ("agregar_a_pagina_de_notion", {"id_pagina": PAGINA_ID, "texto": "x"})
CAMBIAR = ("cambiar_dato_en_notion", {"id_pagina": FILA_ID, "dato": "Estado", "valor": "Pagado"})
MOVER = ("mover_pagina_de_notion", {"id_pagina": PAGINA_ID, "id_destino": DESTINO_ID})
BORRAR = ("borrar_pagina_de_notion", {"id_pagina": PAGINA_ID})


@pytest.mark.parametrize(
    ("llamada", "falla", "esperado"),
    [
        (CREAR, NotionSinPermisoError(), "volvé a hacer /conectar notion"),
        (CAMBIAR, NotionRechazoError(), "No se hizo nada"),
        (
            CREAR,
            EscrituraEnDudaError(),
            "No sé si se pudo crear la página: Notion no me lo confirmó. Fijate en Notion "
            "antes de pedírmelo de nuevo, así no queda repetido.",
        ),
        (AGREGAR, EscrituraEnDudaError(), "así no queda repetido"),
        (CAMBIAR, EscrituraEnDudaError(), "y, si no quedó, pedímelo de nuevo"),
        (BORRAR, EscrituraEnDudaError(), "y, si no quedó, pedímelo de nuevo"),
        (MOVER, ServiceUnavailableError("caído"), "No pude mover la página ahora mismo"),
        (BORRAR, AutorizacionFallidaError("revocada"), "Escribí /conectar notion"),
        (AGREGAR, EntityNotFoundError(), "Esa página ya no está en tu Notion"),
        (BORRAR, EntityNotFoundError(), "no hizo falta mandarla a la papelera"),
    ],
)
async def test_si_falla_despues_del_si_lo_dice_y_no_lo_anota(
    llamada: tuple[str, dict[str, Any]], falla: Exception, esperado: str
) -> None:
    escena = Escena(_notion(fallar_al_escribir=falla))
    await escena.pedir(llamada)

    estado = await escena.decidir(True)

    assert esperado in _respuesta(estado)
    assert escena.contexto.acciones_realizadas == []


# --- Varias juntas ------------------------------------------------------------------------


async def test_crear_y_agregar_juntos_se_confirman_en_una_lista() -> None:
    escena = Escena()

    estado = await escena.pedir(CREAR, AGREGAR)

    assert _pregunta(estado).startswith("Perfecto, hago esto:\n• Crear en tu Notion")
    assert escena.paginas.escrituras() == 0
    await escena.decidir(True)
    assert escena.paginas.escrituras() == 2


async def test_un_lote_que_no_entra_en_un_mensaje_no_pausa_ni_escribe() -> None:
    """Meta rechaza más de 4096 caracteres: la persona no recibiría la pregunta."""
    escena = Escena()

    estado = await escena.pedir(
        ("agregar_a_pagina_de_notion", {"id_pagina": PAGINA_ID, "texto": "a" * 2000}),
        ("agregar_a_pagina_de_notion", {"id_pagina": DESTINO_ID, "texto": "b" * 2000}),
    )

    assert "__interrupt__" not in estado
    respuestas = [m.content for m in estado["messages"] if isinstance(m, ToolMessage)]
    assert respuestas == [LOTE_DEMASIADO_LARGO, LOTE_DEMASIADO_LARGO]
    assert escena.paginas.escrituras() == 0


# --- Crear una base ----------------------------------------------------------------------

SEMANA: dict[str, Any] = {
    "id_pagina": PAGINA_ID,
    "titulo": "Semana",
    "columnas": [{"nombre": "Fecha", "tipo": "fecha"}],
    "filas": [["Lunes", "2026-11-02"], ["Martes", "2026-11-03"]],
}


def _base_nueva(escena: Escena) -> str:
    return next(
        p.id for p in escena.paginas.paginas if p.tipo is TipoDePagina.BASE and p.titulo == "Semana"
    )


async def test_crear_una_base_muestra_columnas_y_filas_y_pide_el_si() -> None:
    escena = Escena()

    estado = await escena.pedir(("crear_base_en_notion", SEMANA))

    assert _pregunta(estado) == (
        "Crear en tu Notion la base «Semana» adentro de «Recetas», con estas columnas:\n"
        "• Nombre (el nombre de cada fila)\n"
        "• Fecha (fecha)\n"
        "\n"
        "Y estas 2 filas:\n"
        "1. Lunes — Fecha: lunes 2 de noviembre de 2026\n"
        "2. Martes — Fecha: martes 3 de noviembre de 2026"
    )
    assert escena.paginas.escrituras() == 0


async def test_con_el_si_crea_la_base_y_despues_sus_filas_de_la_ultima_a_la_primera() -> None:
    """Notion muestra arriba la última fila creada: así se leen en el orden pedido."""
    escena = Escena()
    await escena.pedir(("crear_base_en_notion", SEMANA))

    estado = await escena.decidir(True)

    [(usuario, padre, titulo, columnas)] = escena.paginas.bases_creadas
    assert (usuario, padre, titulo) == (USUARIO, PAGINA_ID, "Semana")
    assert [(c.nombre, c.tipo) for c in columnas] == [("Fecha", TipoDePropiedad.FECHA)]
    assert escena.paginas.primeras_columnas == ["Nombre"]
    base_id = _base_nueva(escena)
    assert [(padre, titulo) for _, padre, titulo, _ in escena.paginas.creadas] == [
        (base_id, "Martes"),
        (base_id, "Lunes"),
    ]
    assert [[(c.nombre, v) for c, v in datos] for datos in escena.paginas.datos_de_creadas] == [
        [("Fecha", date(2026, 11, 3))],
        [("Fecha", date(2026, 11, 2))],
    ]
    # Con una sola columna no hay nada que acomodar en la vista.
    assert escena.paginas.tablas_ordenadas == []
    texto = _respuesta(estado)
    assert texto == (
        f"Creé la base «Semana» adentro de «Recetas» (id: {base_id}), con 2 filas "
        "— link: https://www.notion.so/base-nueva"
    )
    assert escena.contexto.acciones_realizadas == [texto]


async def test_una_base_sin_filas() -> None:
    escena = Escena()
    await escena.pedir(("crear_base_en_notion", {**SEMANA, "filas": None}))

    estado = await escena.decidir(True)

    assert escena.paginas.creadas == []
    assert ", sin filas — link: " in _respuesta(estado)


async def test_cada_tipo_de_columna_y_las_opciones_que_traen_las_filas() -> None:
    escena = Escena()
    args = {
        "id_pagina": PAGINA_ID,
        "titulo": "Gastos 2027",
        "columnas": [
            {"nombre": "Rubro", "tipo": "selección", "opciones": ["Servicios"]},
            {"nombre": "Monto", "tipo": "numero"},
            {"nombre": "Pagado", "tipo": "Casilla"},
            {"nombre": "Notas", "tipo": "texto"},
        ],
        "filas": [
            ["Luz", "servicios", "1.500,50", "sí", ""],
            ["Súper", "Comida", "$2000", "no", "con tarjeta"],
        ],
    }

    estado = await escena.pedir(("crear_base_en_notion", args))

    lineas = _pregunta(estado).splitlines()
    assert lineas[1:6] == [
        "• Nombre (el nombre de cada fila)",
        "• Rubro (selección: Servicios, Comida)",  # «Comida» la trajo una fila
        "• Monto (número)",
        "• Pagado (casilla)",
        "• Notas (texto)",
    ]
    assert lineas[-2:] == [
        "1. Luz — Rubro: «Servicios» — Monto: 1.500,5 — Pagado: marcada",
        "2. Súper — Rubro: «Comida» — Monto: 2.000 — Pagado: sin marcar — Notas: «con tarjeta»",
    ]
    await escena.decidir(True)
    [(_, _, _, columnas)] = escena.paginas.bases_creadas
    assert columnas[0].opciones == ("Servicios", "Comida")
    assert [[v for _, v in datos] for datos in escena.paginas.datos_de_creadas] == [
        ["Comida", 2000.0, False, "con tarjeta"],
        ["Servicios", 1500.5, True],
    ]
    # Varias columnas: la vista las pone en el orden pedido, sin ordenar las filas.
    [(_, en_orden, por)] = escena.paginas.tablas_ordenadas
    assert (en_orden, por) == (("Rubro", "Monto", "Pagado", "Notas"), None)


def _con(**cambios: Any) -> dict[str, Any]:
    return {**SEMANA, **cambios}


def _columnas(*tipos: str) -> list[dict[str, Any]]:
    return [{"nombre": f"C{n}", "tipo": tipo} for n, tipo in enumerate(tipos)]


@pytest.mark.parametrize(
    ("args", "esperado"),
    [
        (_con(titulo=""), "Falta el nombre de la base"),
        (_con(titulo="Ver https://sitio.example"), "no puede tener links"),
        (_con(columnas=_columnas(*["texto"] * 6), filas=None), "lleva hasta 5"),
        (_con(columnas=[{"nombre": " ", "tipo": "texto"}], filas=None), "una columna sin nombre"),
        (_con(columnas=[{"nombre": "Título", "tipo": "texto"}], filas=None), "No hace falta"),
        (_con(columnas=[{"nombre": "nombre", "tipo": "texto"}], filas=None), "No hace falta"),
        (
            _con(
                columnas=[
                    {"nombre": "Fecha", "tipo": "fecha"},
                    {"nombre": "fecha", "tipo": "texto"},
                ],
                filas=None,
            ),
            "Hay dos columnas que se llaman",
        ),
        (_con(columnas=[{"nombre": "x" * 81, "tipo": "texto"}], filas=None), "hasta 80"),
        (_con(columnas=[{"nombre": "Estado", "tipo": "estado"}], filas=None), "no puedo crear"),
        (_con(columnas=[{"nombre": "Lista", "tipo": "lista"}], filas=None), "no puedo crear"),
        (
            _con(columnas=[{"nombre": "R", "tipo": "selección", "opciones": ["A, B"]}], filas=None),
            "tiene una coma",
        ),
        (
            _con(
                columnas=[{"nombre": "R", "tipo": "selección", "opciones": ["x" * 81]}], filas=None
            ),
            "es muy larga",
        ),
        (
            _con(
                columnas=[
                    {"nombre": "R", "tipo": "selección", "opciones": [f"O{n}" for n in range(21)]}
                ],
                filas=None,
            ),
            "más de 20 opciones",
        ),
        (_con(filas=[[f"F{n}", "2026-11-02"] for n in range(11)]), "con datos van hasta 10"),
        (_con(filas=[[f"F{n}"] for n in range(32)]), "sólo nombres, sin datos: hasta 31"),
        (_con(filas=[["Lunes", "2026-11-02", "extra"]]), "tiene 3 valores"),
        (_con(filas=[["", "2026-11-02"]]), "A la fila 1 le falta el nombre"),
        (_con(filas=[["Lunes", "2 de noviembre"]]), "Fila 1 («Lunes»), «Fecha»: La fecha va en"),
        (
            _con(columnas=[{"nombre": "Monto", "tipo": "número"}], filas=[["Luz", "1.500"]]),
            "se puede leer de dos maneras",
        ),
        (
            _con(columnas=[{"nombre": "Pagado", "tipo": "casilla"}], filas=[["Luz", "tal vez"]]),
            "«sí» o «no»",
        ),
        (_con(id_pagina=BASE_ID), "es una base: una base nueva va adentro de una página"),
        (_con(id_pagina=str(uuid4())), "compartilo con LifeSync"),
        (
            _con(
                columnas=[{"nombre": "Notas", "tipo": "texto"}],
                filas=[["A", "a" * 1990], ["B", "b" * 1990]],
            ),
            "demasiado largo",
        ),
    ],
)
async def test_crear_una_base_invalida_se_explica_sin_pausa(
    args: dict[str, Any], esperado: str
) -> None:
    escena = Escena()

    estado = await escena.pedir(("crear_base_en_notion", args))

    assert "__interrupt__" not in estado
    assert esperado in _respuesta(estado)
    assert escena.paginas.escrituras() == 0


async def test_la_pregunta_cuenta_lo_ya_hecho_en_el_turno() -> None:
    """El agente le antepone lo ya hecho («Listo — …»): todo tiene que entrar en un WhatsApp."""
    escena = Escena()
    # La pregunta sola entra (~250 caracteres); con esto adelante, ya no.
    escena.contexto.acciones_realizadas.append("x" * 3700)

    estado = await escena.pedir(("crear_base_en_notion", SEMANA))

    assert "__interrupt__" not in estado
    assert "demasiado largo" in _respuesta(estado)


TRES_DIAS = _con(
    filas=[["Lunes", "2026-11-02"], ["Martes", "2026-11-03"], ["Miércoles", "2026-11-04"]]
)


async def test_si_se_corta_a_mitad_dice_cuantas_filas_quedaron_y_no_invita_a_repetirla() -> None:
    # Escrituras: la base (1) y el miércoles (2), que va primero; falla el martes, dos veces.
    escena = Escena(
        _notion(fallar_al_escribir=ServiceUnavailableError("caído"), escrituras_antes_de_fallar=2)
    )
    await escena.pedir(("crear_base_en_notion", TRES_DIAS))

    estado = await escena.decidir(True)

    base_id = _base_nueva(escena)
    texto = _respuesta(estado)
    assert texto == (
        f"Creé la base «Semana» (id: {base_id}): ya existe, no hay que crearla de nuevo. "
        "Alcancé a crear 1 de 3 filas: «Miércoles». Faltan «Lunes», «Martes». Se pueden "
        "sumar de a una, como filas nuevas de la base, de la última a la primera: así quedan "
        "en orden."
    )
    # La bitácora dice lo mismo: si el modelo se cae, la persona sabe qué falta.
    assert escena.contexto.acciones_realizadas == [texto]


async def test_una_fila_en_duda_se_dice_aparte_y_no_se_reintenta() -> None:
    """Pudo haberse creado: otra vez la repetiría. Con un reintento, ésta habría andado."""
    escena = Escena(
        _notion(
            fallar_al_escribir=EscrituraEnDudaError(),
            escrituras_antes_de_fallar=2,
            fallar_una_sola_vez=True,
        )
    )
    await escena.pedir(("crear_base_en_notion", TRES_DIAS))

    estado = await escena.decidir(True)

    texto = _respuesta(estado)
    assert "Alcancé a crear 1 de 3 filas: «Miércoles»." in texto
    assert "La fila «Martes» puede haber quedado" in texto
    assert "Falta «Lunes»." in texto
    assert len(escena.paginas.creadas) == 1


async def test_una_falla_pasajera_en_una_fila_se_reintenta_una_vez() -> None:
    escena = Escena(
        _notion(
            fallar_al_escribir=ServiceUnavailableError("429"),
            escrituras_antes_de_fallar=2,
            fallar_una_sola_vez=True,
        )
    )
    await escena.pedir(("crear_base_en_notion", TRES_DIAS))

    estado = await escena.decidir(True)

    assert [titulo for _, _, titulo, _ in escena.paginas.creadas] == [
        "Miércoles",
        "Martes",
        "Lunes",
    ]
    # El reintento lleva los mismos datos que la pregunta mostró.
    assert [[(c.nombre, v) for c, v in datos] for datos in escena.paginas.datos_de_creadas] == [
        [("Fecha", date(2026, 11, 4))],
        [("Fecha", date(2026, 11, 3))],
        [("Fecha", date(2026, 11, 2))],
    ]
    assert ", con 3 filas — link: " in _respuesta(estado)


async def test_sin_ninguna_fila_creada_las_nombra_a_todas() -> None:
    escena = Escena(
        _notion(fallar_al_escribir=ServiceUnavailableError("caído"), escrituras_antes_de_fallar=1)
    )
    await escena.pedir(("crear_base_en_notion", TRES_DIAS))

    estado = await escena.decidir(True)

    texto = _respuesta(estado)
    assert "No alcancé a crear ninguna de sus 3 filas. Faltan «Lunes», «Martes», «Miércoles»." in (
        texto
    )


async def test_sin_permiso_a_mitad_no_invita_a_sumarlas_de_a_una() -> None:
    """Puede ser el límite de bloques del plan gratis: de a una tampoco entrarían."""
    escena = Escena(
        _notion(fallar_al_escribir=NotionSinPermisoError(), escrituras_antes_de_fallar=2)
    )
    await escena.pedir(("crear_base_en_notion", TRES_DIAS))

    estado = await escena.decidir(True)

    texto = _respuesta(estado)
    assert "Faltan «Lunes», «Martes». Notion no me dejó escribir ahí." in texto
    assert "de a una" not in texto


@pytest.mark.parametrize(
    ("falla", "esperado"),
    [
        (NotionSinPermisoError(), "volvé a hacer /conectar notion"),
        (EscrituraEnDudaError(), "No sé si se pudo crear la base"),
        (EntityNotFoundError(), "no creé la base"),
    ],
)
async def test_si_la_base_no_se_crea_no_intenta_las_filas(falla: Exception, esperado: str) -> None:
    escena = Escena(_notion(fallar_al_escribir=falla))
    await escena.pedir(("crear_base_en_notion", SEMANA))

    estado = await escena.decidir(True)

    assert esperado in _respuesta(estado)
    assert escena.paginas.escrituras() == 0
    assert escena.contexto.acciones_realizadas == []


async def test_una_falla_no_prevista_en_las_filas_deja_la_base_en_la_bitacora() -> None:
    """Si el turno se cae, el agente cuenta lo de la bitácora: nadie pide la base otra vez."""
    escena = Escena(_notion(fallar_al_escribir=RuntimeError("bug"), escrituras_antes_de_fallar=1))
    await escena.pedir(("crear_base_en_notion", SEMANA))

    with pytest.raises(RuntimeError):
        await escena.decidir(True)

    [hecho] = escena.contexto.acciones_realizadas
    assert hecho.startswith("Creé la base «Semana» adentro de «Recetas» (id: ")


# --- Una fila nueva con sus datos -------------------------------------------------------


async def test_una_fila_nueva_con_sus_datos_en_una_sola_confirmacion() -> None:
    escena = Escena()
    args = {
        "id_padre": BASE_ID,
        "titulo": "Gas",
        "datos": [
            {"columna": "monto", "valor": "1500"},
            {"columna": "Vence", "valor": "2026-11-10"},
            {"columna": "Estado", "valor": "pagado"},
        ],
    }

    estado = await escena.pedir(("crear_pagina_en_notion", args))

    assert _pregunta(estado) == (
        "Crear en tu Notion la página «Gas» como fila nueva de la base «Gastos», con estos "
        "datos:\n"
        "• Monto: 1.500\n"
        "• Vence: martes 10 de noviembre de 2026\n"
        "• Estado: «Pagado»"
    )
    await escena.decidir(True)
    [datos] = escena.paginas.datos_de_creadas
    assert [(c.id, v) for c, v in datos] == [
        ("m1", 1500.0),
        ("v1", date(2026, 11, 10)),
        ("e1", "Pagado"),
    ]


async def test_datos_y_texto_van_juntos_en_la_pregunta() -> None:
    escena = Escena()
    args = {
        "id_padre": BASE_ID,
        "titulo": "Gas",
        "texto": "- pagar online",
        "datos": [{"columna": "Monto", "valor": "1500"}],
    }

    estado = await escena.pedir(("crear_pagina_en_notion", args))

    assert _pregunta(estado).endswith(
        "con estos datos:\n• Monto: 1.500\n\nY con este texto:\n\n• pagar online"
    )


@pytest.mark.parametrize(
    ("id_padre", "datos", "esperado"),
    [
        (PAGINA_ID, [{"columna": "Monto", "valor": "1"}], "es una página, no una base"),
        (BASE_ID, [{"columna": "Color", "valor": "1"}], "no tiene un dato que se llame «Color»"),
        (
            BASE_ID,
            [{"columna": "Monto", "valor": "1"}, {"columna": "monto", "valor": "2"}],
            "está dos veces",
        ),
        (BASE_ID, [{"columna": "Concepto", "valor": "Gas"}], "va en el título"),
        (BASE_ID, [{"columna": "título", "valor": "Gas"}], "va en el título"),
        (BASE_ID, [{"columna": " ", "valor": "1"}], "Hay un dato sin columna"),
        (BASE_ID, [{"columna": "Estado", "valor": "Archivado"}], "no tiene la opción"),
        (BASE_ID, [{"columna": "Monto", "valor": "mil"}], "«Monto»: «mil» no es un número"),
    ],
)
async def test_datos_que_no_sirven_se_explican_sin_pausa(
    id_padre: str, datos: list[dict[str, str]], esperado: str
) -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("crear_pagina_en_notion", {"id_padre": id_padre, "titulo": "Gas", "datos": datos})
    )

    assert "__interrupt__" not in estado
    assert esperado in _respuesta(estado)
    assert escena.paginas.escrituras() == 0


# --- Una fila por día, la primera columna y sin recortar (QA del 28/10) -------------------

NOVIEMBRE: dict[str, Any] = {
    "id_pagina": PAGINA_ID,
    "titulo": "Calendario Noviembre",
    "columnas": [
        {"nombre": "Descripción", "tipo": "texto"},
        {"nombre": "Check", "tipo": "casilla"},
    ],
    "primera_columna": "Fecha",
    "desde": "2026-11-01",
    "hasta": "2026-11-30",
}


def _base_llamada(escena: Escena, titulo: str) -> str:
    return next(
        p.id for p in escena.paginas.paginas if p.tipo is TipoDePagina.BASE and p.titulo == titulo
    )


async def test_el_pedido_de_noviembre_arma_una_fila_por_dia_con_la_fecha_primero() -> None:
    """El pedido de la QA: la primera columna es la fecha y hay una fila por cada día."""
    escena = Escena()

    estado = await escena.pedir(("crear_base_en_notion", NOVIEMBRE))

    assert _pregunta(estado) == (
        "Crear en tu Notion la base «Calendario Noviembre» adentro de «Recetas», con estas "
        "columnas:\n"
        "• Fecha (texto con el día, como «Domingo 1 de noviembre»: Notion no deja que la "
        "primera sea de tipo fecha)\n"
        "• Descripción (texto)\n"
        "• Check (casilla)\n"
        "\n"
        "Y una fila por día, del domingo 1 al lunes 30 de noviembre de 2026: 30 filas. Las "
        "otras columnas quedan vacías.\n"
        "\n"
        "Puede tardar cerca de medio minuto."
    )
    assert escena.paginas.escrituras() == 0

    estado = await escena.decidir(True)

    base_id = _base_llamada(escena, "Calendario Noviembre")
    assert escena.paginas.primeras_columnas == ["Fecha"]
    [(_, _, _, columnas)] = escena.paginas.bases_creadas
    assert [c.nombre for c in columnas] == ["Descripción", "Check"]
    titulos = [titulo for _, _, titulo, _ in escena.paginas.creadas]
    assert len(titulos) == 30
    assert (titulos[0], titulos[-1]) == ("Lunes 30 de noviembre", "Domingo 1 de noviembre")
    # Nada inventado: ni descripciones ni casillas.
    assert all(datos == () for datos in escena.paginas.datos_de_creadas)
    assert escena.paginas.tablas_ordenadas == [(base_id, ("Descripción", "Check"), None)]
    assert _respuesta(estado) == (
        f"Creé la base «Calendario Noviembre» adentro de «Recetas» (id: {base_id}), con 30 "
        "filas — link: https://www.notion.so/base-nueva"
    )


async def test_con_una_columna_fecha_lleva_el_dia_y_la_tabla_se_ordena_por_ella() -> None:
    escena = Escena()
    args = {
        "id_pagina": PAGINA_ID,
        "titulo": "Turnos",
        "columnas": [{"nombre": "Fecha", "tipo": "fecha"}, {"nombre": "Notas", "tipo": "texto"}],
        "desde": "2026-11-01",
        "hasta": "2026-11-03",
    }

    estado = await escena.pedir(("crear_base_en_notion", args))

    assert _pregunta(estado).splitlines()[1:] == [
        "• Día (texto con el día, como «Domingo 1 de noviembre»: Notion no deja que la primera "
        "sea de tipo fecha)",
        "• Fecha (fecha)",
        "• Notas (texto)",
        "",
        "Y una fila por día, del domingo 1 al martes 3 de noviembre de 2026: 3 filas, con el "
        "día también en «Fecha», y ordenadas por «Fecha». Las otras columnas quedan vacías.",
    ]
    await escena.decidir(True)
    assert escena.paginas.primeras_columnas == ["Día"]
    assert [[(c.nombre, v) for c, v in datos] for datos in escena.paginas.datos_de_creadas] == [
        [("Fecha", date(2026, 11, 3))],
        [("Fecha", date(2026, 11, 2))],
        [("Fecha", date(2026, 11, 1))],
    ]
    [(_, columnas, por)] = escena.paginas.tablas_ordenadas
    assert (columnas, por) == (("Fecha", "Notas"), "Fecha")


async def test_una_columna_de_fecha_con_el_nombre_de_la_primera_se_saca() -> None:
    """Lo más probable del modelo con el pedido de la QA: la fecha primero, y también aparte."""
    escena = Escena()
    columnas = [{"nombre": "fecha", "tipo": "date"}, *NOVIEMBRE["columnas"]]

    estado = await escena.pedir(("crear_base_en_notion", {**NOVIEMBRE, "columnas": columnas}))

    assert "• fecha (fecha)" not in _pregunta(estado)
    await escena.decidir(True)
    [(_, _, _, creadas)] = escena.paginas.bases_creadas
    assert [c.nombre for c in creadas] == ["Descripción", "Check"]


async def test_otra_fecha_no_lleva_el_dia_ni_ordena() -> None:
    """«Vence» es otra fecha: queda vacía, y la tabla no se ordena por ella."""
    escena = Escena()
    args = {
        **NOVIEMBRE,
        "hasta": "2026-11-02",
        "columnas": [{"nombre": "Vence", "tipo": "fecha"}],
    }

    estado = await escena.pedir(("crear_base_en_notion", args))

    assert _pregunta(estado).endswith(
        "Y una fila por día, del domingo 1 al lunes 2 de noviembre de 2026: 2 filas. Las otras "
        "columnas quedan vacías."
    )
    await escena.decidir(True)
    assert all(datos == () for datos in escena.paginas.datos_de_creadas)
    assert escena.paginas.tablas_ordenadas == []


@pytest.mark.parametrize(
    ("desde", "hasta", "esperado"),
    [
        ("2026-11-15", "2026-11-31", "del domingo 15 al lunes 30 de noviembre de 2026: 16 filas"),
        (
            "2026-11-20",
            "2026-12-05",
            "del viernes 20 de noviembre al sábado 5 de diciembre de 2026: 16 filas",
        ),
        (
            "2026-12-15",
            "2027-01-10",
            "del martes 15 de diciembre de 2026 al domingo 10 de enero de 2027: 27 filas",
        ),
        ("2026-11-01", "2026-11-01", "el domingo 1 de noviembre de 2026: una fila"),
        (
            "2026-11-01",
            "2026-12-01",
            "del domingo 1 de noviembre al martes 1 de diciembre de 2026: 31 filas",
        ),
    ],
    ids=["fin_de_mes", "dos_meses", "cruza_el_año", "un_dia", "tope"],
)
async def test_el_periodo_se_muestra_sin_repetir_mes_ni_año(
    desde: str, hasta: str, esperado: str
) -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("crear_base_en_notion", {**NOVIEMBRE, "desde": desde, "hasta": hasta})
    )

    assert f"Y una fila por día, {esperado}." in _pregunta(estado)


@pytest.mark.parametrize(
    ("cambios", "esperado"),
    [
        ({"hasta": ""}, "hacen falta las dos fechas"),
        ({"desde": ""}, "hacen falta las dos fechas"),
        ({"desde": "1 de noviembre"}, "«desde» va en AAAA-MM-DD"),
        ({"hasta": "2026-13-01"}, "«hasta» no es una fecha"),
        ({"desde": "2026-12-20", "hasta": "2026-01-10"}, "«hasta» es del año siguiente"),
        ({"hasta": "2026-12-02"}, "Son 32 días"),
        ({"filas": [["Lunes"]]}, "las dos cosas juntas no"),
        ({"primera_columna": "Ver sitio.example"}, "primera columna no puede tener links"),
        ({"primera_columna": "x" * 81}, "va hasta 80 caracteres"),
        (
            {"primera_columna": "", "columnas": [{"nombre": "Día", "tipo": "texto"}]},
            "No hace falta la columna «Día»: el nombre de cada fila ya va en la primera "
            "columna, «Día»",
        ),
        (
            {"columnas": [{"nombre": "Fecha", "tipo": "texto"}]},
            "pasá otro nombre en primera_columna",
        ),
    ],
)
async def test_un_rango_que_no_sirve_se_explica_sin_pausa(
    cambios: dict[str, Any], esperado: str
) -> None:
    escena = Escena()

    estado = await escena.pedir(("crear_base_en_notion", {**NOVIEMBRE, **cambios}))

    assert "__interrupt__" not in estado
    assert esperado in _respuesta(estado)
    assert escena.paginas.escrituras() == 0


async def test_sin_rango_una_fecha_que_se_llama_como_la_primera_choca() -> None:
    escena = Escena()
    args = _con(primera_columna="Fecha", filas=None)

    estado = await escena.pedir(("crear_base_en_notion", args))

    assert "No hace falta la columna «Fecha»" in _respuesta(estado)


async def test_con_otra_primera_columna_una_que_se_llame_nombre_vale() -> None:
    escena = Escena()
    args = {
        **_con(primera_columna="Tarea", filas=None),
        "columnas": [{"nombre": "Nombre", "tipo": "texto"}],
    }

    estado = await escena.pedir(("crear_base_en_notion", args))

    assert _pregunta(estado).splitlines()[1:3] == [
        "• Tarea (el nombre de cada fila)",
        "• Nombre (texto)",
    ]


async def test_una_fila_puede_traer_solo_su_nombre() -> None:
    escena = Escena()

    estado = await escena.pedir(
        ("crear_base_en_notion", _con(filas=[["Leche"], ["Pan", "2026-11-02"]]))
    )

    assert _pregunta(estado).endswith(
        "Y estas 2 filas:\n1. Leche\n2. Pan — Fecha: lunes 2 de noviembre de 2026"
    )
    await escena.decidir(True)
    assert [len(datos) for datos in escena.paginas.datos_de_creadas] == [1, 0]


async def test_en_el_tope_la_pregunta_dice_que_es_lo_maximo() -> None:
    """Si el modelo recortó la lista justo en el tope, la persona lo ve antes del sí."""
    escena = Escena()
    filas = [[f"F{n}", "2026-11-02"] for n in range(10)]

    estado = await escena.pedir(("crear_base_en_notion", _con(filas=filas)))

    assert "Y estas 10 filas (es lo máximo de una vez):" in _pregunta(estado)


async def test_hasta_31_filas_si_solo_llevan_el_nombre() -> None:
    escena = Escena()
    filas = [[f"Tarea {n}"] for n in range(1, 32)]

    estado = await escena.pedir(("crear_base_en_notion", _con(filas=filas)))

    pregunta = _pregunta(estado)
    assert "Y estas 31 filas (es lo máximo de una vez):" in pregunta
    assert pregunta.endswith("31. Tarea 31\n\nPuede tardar cerca de medio minuto.")


async def test_con_mas_filas_de_las_que_entran_nunca_dice_que_se_recorten() -> None:
    """«Creala con las primeras» hizo que el modelo cortara en 10 sin avisar (QA del 28/10)."""
    escena = Escena()
    filas = [[f"F{n}", "2026-11-02"] for n in range(14)]

    estado = await escena.pedir(("crear_base_en_notion", _con(filas=filas)))

    texto = _respuesta(estado)
    assert "las primeras" not in texto
    assert (
        "NO las recortes ni les saques datos para que entren: decile a la persona cuántas "
        "entran y preguntale cómo seguir."
    ) in texto
    # Partirla, sólo si la persona lo pide, y con las últimas: así se leen en orden.
    assert "Recién si la persona te dice que la partas: creá la base con las ÚLTIMAS" in texto


@pytest.mark.parametrize("tipo", ["check", "Tilde", "checkbox"])
async def test_check_y_tilde_son_casillas(tipo: str) -> None:
    escena = Escena()
    args = _con(columnas=[{"nombre": "Hecho", "tipo": tipo}], filas=None)

    estado = await escena.pedir(("crear_base_en_notion", args))

    assert "• Hecho (casilla)" in _pregunta(estado)


async def test_si_se_corta_un_rango_dice_que_dias_faltan() -> None:
    # Escrituras: la base (1), el 5 (2) y el 4 (3); falla el 3, dos veces.
    escena = Escena(
        _notion(fallar_al_escribir=ServiceUnavailableError("caído"), escrituras_antes_de_fallar=3)
    )
    await escena.pedir(("crear_base_en_notion", {**NOVIEMBRE, "hasta": "2026-11-05"}))

    estado = await escena.decidir(True)

    assert (
        "Alcancé a crear 2 de 5 filas: las del miércoles 4 al jueves 5 de noviembre de 2026. "
        "Faltan las del domingo 1 al martes 3 de noviembre de 2026."
    ) in _respuesta(estado)


async def test_si_no_se_puede_ordenar_por_la_fecha_prometida_lo_dice() -> None:
    escena = Escena(_notion(fallar_al_ordenar=ServiceUnavailableError("sin vistas")))
    args = {
        **NOVIEMBRE,
        "hasta": "2026-11-02",
        "columnas": [{"nombre": "Fecha", "tipo": "fecha"}],
        "primera_columna": "",
    }
    await escena.pedir(("crear_base_en_notion", args))

    estado = await escena.decidir(True)

    texto = _respuesta(estado)
    assert ", con 2 filas (no pude ordenarlas por «Fecha»: se puede desde Notion) — link: " in (
        texto
    )
    assert escena.contexto.acciones_realizadas == [texto]


async def test_si_no_se_pueden_acomodar_las_columnas_no_se_dice_nada() -> None:
    """Nadie lo prometió: la base quedó bien igual."""
    escena = Escena(_notion(fallar_al_ordenar=ServiceUnavailableError("sin vistas")))
    await escena.pedir(("crear_base_en_notion", NOVIEMBRE))

    estado = await escena.decidir(True)

    assert _respuesta(estado).endswith(", con 30 filas — link: https://www.notion.so/base-nueva")


TURNOS: dict[str, Any] = {
    **NOVIEMBRE,
    "hasta": "2026-11-03",
    "primera_columna": "",
    "columnas": [{"nombre": "Fecha", "tipo": "fecha"}],
}


async def test_si_las_filas_se_cortan_igual_ordena_y_si_no_puede_lo_dice() -> None:
    """La pregunta prometió el orden por «Fecha»: con filas a medias, también."""
    escena = Escena(
        _notion(fallar_al_escribir=ServiceUnavailableError("caído"), escrituras_antes_de_fallar=2)
    )
    await escena.pedir(("crear_base_en_notion", TURNOS))

    estado = await escena.decidir(True)

    texto = _respuesta(estado)
    assert texto.endswith(
        "de la última a la primera: así quedan en orden. No pude ordenarlas por «Fecha»: se "
        "puede desde Notion."
    )
    assert escena.contexto.acciones_realizadas == [texto]


async def test_con_filas_a_medias_la_tabla_queda_ordenada_si_se_puede() -> None:
    """Ordenada por la fecha, lo que se sume después cae en su lugar."""
    escena = Escena(
        _notion(
            fallar_al_escribir=EscrituraEnDudaError(),
            escrituras_antes_de_fallar=2,
            fallar_una_sola_vez=True,
        )
    )
    await escena.pedir(("crear_base_en_notion", TURNOS))

    estado = await escena.decidir(True)

    [(_, _, por)] = escena.paginas.tablas_ordenadas
    assert por == "Fecha"
    texto = _respuesta(estado)
    assert "puede haber quedado" in texto
    assert "ordenarlas" not in texto
    # Ordenada por la fecha, lo que se sume sin su día iría al final.
    assert texto.endswith(
        "La tabla quedó ordenada por «Fecha»: al sumar las que faltan, poné su día también "
        "en «Fecha»."
    )


async def test_un_orden_en_duda_no_se_da_por_fallido() -> None:
    """Un 5xx al ordenar pudo haber quedado: no se dice «no pude»."""
    escena = Escena(_notion(fallar_al_ordenar=EscrituraEnDudaError()))
    await escena.pedir(("crear_base_en_notion", TURNOS))

    estado = await escena.decidir(True)

    assert ", con 3 filas (no sé si quedaron ordenadas por «Fecha»: Notion no lo confirmó) — " in (
        _respuesta(estado)
    )


async def test_una_primera_columna_fecha_sin_rango_avisa_que_es_texto() -> None:
    escena = Escena()
    args = _con(primera_columna="Fecha", columnas=[{"nombre": "Monto", "tipo": "número"}])
    args["filas"] = [["2026-11-02", "1500"]]

    estado = await escena.pedir(("crear_base_en_notion", args))

    assert _pregunta(estado).splitlines()[1] == (
        "• Fecha (el nombre de cada fila, en texto: Notion no deja que la primera sea de tipo "
        "fecha)"
    )
