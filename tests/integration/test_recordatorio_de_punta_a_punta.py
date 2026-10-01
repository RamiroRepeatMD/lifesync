"""Un recordatorio de punta a punta, sin red (PB-030).

La persona lo pide al agente (grafo real, tool real, modelo falso), lo
confirma, y cuando llega la hora el despachador lo manda con el cliente de
WhatsApp real: sólo el HTTP hacia Meta es simulado. Se aserta el cuerpo
EXACTO que recibe Meta — incluido el 9 de los móviles argentinos, que en un
mensaje que inicia el bot también hay que sacar.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.application.use_cases.enviar_recordatorios_vencidos import EnviarRecordatoriosVencidos
from src.domain.entities.recordatorio import EstadoDeRecordatorio
from src.domain.entities.usuario import Usuario
from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.config.zona import ZONA_HORARIA
from src.infrastructure.external.whatsapp.cliente import ClienteWhatsApp
from src.infrastructure.llm.contexto import ContextoDeAgente
from src.infrastructure.llm.grafo import construir_grafo
from src.infrastructure.llm.herramientas import construir_herramientas
from tests.dobles import ModeloFalso, RecordatoriosEnMemoria, RepositorioUsuarioEnMemoria

TELEFONO = "+5491123456789"  # como lo entrega Meta en el webhook: con el 9


async def test_pedido_por_el_agente_y_entregado_por_whatsapp_a_la_hora() -> None:
    usuarios = RepositorioUsuarioEnMemoria()
    usuario = await usuarios.crear(Usuario(telefono_whatsapp=TELEFONO))
    assert usuario.id is not None
    recordatorios = RecordatoriosEnMemoria()

    # 1. La persona lo pide y lo confirma.
    momento = (datetime.now(ZONA_HORARIA) + timedelta(minutes=20)).replace(second=0, microsecond=0)
    modelo = ModeloFalso(
        guion=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "crear_recordatorio",
                        "id": "r1",
                        "args": {
                            "texto": "sacar la pizza",
                            "fecha": f"{momento:%Y-%m-%d}",
                            "hora": f"{momento:%H:%M}",
                        },
                    }
                ],
            ),
            AIMessage("Listo, te aviso."),
        ]
    )
    grafo: Any = construir_grafo(
        modelo, construir_herramientas(None, recordatorios=recordatorios), InMemorySaver()
    )
    config: Any = {"configurable": {"thread_id": "hilo"}}
    contexto = ContextoDeAgente(usuario_id=usuario.id)
    await grafo.ainvoke(
        {"messages": [HumanMessage("recordame en 20 minutos que saque la pizza")]},
        config=config,
        context=contexto,
    )
    await grafo.ainvoke(Command(resume={"aprobado": True}), config=config, context=contexto)

    # 2. Llega la hora: el despachador lo manda por el cliente real.
    pedidos: list[httpx.Request] = []

    def meta(pedido: httpx.Request) -> httpx.Response:
        pedidos.append(pedido)
        return httpx.Response(200, json={"messages": [{"id": "wamid.X"}]})

    settings = Settings(
        _env_file=None,
        environment=Environment.TESTING,
        whatsapp_token="token-de-prueba",
        whatsapp_phone_number_id="123456",
    )
    cliente = ClienteWhatsApp(httpx.AsyncClient(transport=httpx.MockTransport(meta)), settings)
    caso = EnviarRecordatoriosVencidos(recordatorios, usuarios, cliente, zona=ZONA_HORARIA)

    antes = await caso.ejecutar(momento - timedelta(seconds=1))
    a_la_hora = await caso.ejecutar(momento + timedelta(seconds=15))

    assert antes == 0  # ni un segundo antes
    assert a_la_hora == 1
    [pedido] = pedidos
    assert json.loads(pedido.content) == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "+541123456789",  # sin el 9: lo que Meta exige al enviar
        "type": "text",
        "text": {"preview_url": False, "body": "⏰ Recordatorio: sacar la pizza"},
    }
    [guardado] = recordatorios.guardados.values()
    assert guardado.estado is EstadoDeRecordatorio.ENVIADO
