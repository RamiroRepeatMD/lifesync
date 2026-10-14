"""El cableado de lo que recibe archivos (PB-038), en el composition root.

`app.state.whatsapp` es el cliente httpx crudo (con el token por defecto), no
un `ClienteWhatsApp`: pasarlo tal cual a las tools tipaba —`app.state` no tiene
tipos— y la primera subida en producción rompía con AttributeError.
"""

from __future__ import annotations

import httpx

from src.infrastructure.config.settings import Environment, Settings
from src.infrastructure.external.whatsapp.cliente import ClienteWhatsApp
from src.interfaces.api.app import _recibidos_de, create_app


def _app() -> object:
    return create_app(Settings(_env_file=None, environment=Environment.TESTING))


async def test_las_tools_reciben_el_cliente_de_whatsapp_y_no_el_httpx_crudo() -> None:
    app = _app()
    app.state.whatsapp = httpx.AsyncClient()  # type: ignore[attr-defined]

    recibidos = _recibidos_de(app)  # type: ignore[arg-type]

    assert isinstance(recibidos, ClienteWhatsApp)
    await app.state.whatsapp.aclose()  # type: ignore[attr-defined]


def test_sin_whatsapp_no_hay_de_donde_bajar_archivos() -> None:
    assert _recibidos_de(_app()) is None  # type: ignore[arg-type]
