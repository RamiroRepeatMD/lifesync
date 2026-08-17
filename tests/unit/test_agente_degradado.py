"""Tests del agente de reemplazo sin LLM (PB-005)."""

from __future__ import annotations

import ast
from pathlib import Path
from uuid import uuid4

from src.application.dto.consulta_del_usuario import ConsultaDelUsuario
from src.application.ports.agente import AgenteConversacional
from src.application.services import agente_degradado
from src.application.services.agente_degradado import SIN_AGENTE, AgenteDegradado


def _consulta(texto: str = "agendame algo") -> ConsultaDelUsuario:
    return ConsultaDelUsuario(conversacion_id=uuid4(), texto=texto)


async def test_contesta_el_aviso_en_vez_de_fallar() -> None:
    """Dejar el mensaje sin respuesta sería peor que decir que hoy no se puede."""
    assert await AgenteDegradado().responder(_consulta()) == SIN_AGENTE


async def test_contesta_siempre_lo_mismo() -> None:
    agente = AgenteDegradado()

    primera = await agente.responder(_consulta("hola"))
    segunda = await agente.responder(_consulta("¿qué hora es?"))

    assert primera == segunda


def test_cumple_el_puerto() -> None:
    assert isinstance(AgenteDegradado(), AgenteConversacional)


def test_orienta_hacia_los_comandos_que_si_funcionan() -> None:
    """RF-11: en modo degradado son lo único que contesta de verdad."""
    assert "/ayuda" in SIN_AGENTE


def test_no_depende_del_stack_de_langchain() -> None:
    """Es el camino que tiene que andar justamente cuando ese stack no está.

    Si alguien importara LangChain acá, el modo degradado se caería con un
    ImportError en el peor momento posible. Se mira el árbol sintáctico y no
    los nombres del módulo, porque un `import langgraph` sin usar no dejaría
    rastro en `vars()` y rompería igual.
    """
    fuente = Path(agente_degradado.__file__).read_text(encoding="utf-8")
    arbol = ast.parse(fuente)

    modulos: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos.update(alias.name for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module)

    prohibidos = [m for m in modulos if m.startswith(("langchain", "langgraph", "google"))]
    assert prohibidos == []
