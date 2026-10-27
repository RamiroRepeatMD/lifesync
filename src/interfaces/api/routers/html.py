"""Páginas HTML mínimas para lo que abre una persona en el navegador (PB-009 · PB-014).

Las vueltas de OAuth de Google y de Notion las ve alguien en la pantalla del
teléfono, no un cliente de API: un `{"error": {...}}` no le dice nada.
"""

from __future__ import annotations

from fastapi.responses import HTMLResponse

_ESTILO = (
    "font-family:system-ui,-apple-system,sans-serif;max-width:32rem;margin:15vh auto;"
    "padding:0 1.5rem;text-align:center;line-height:1.6;color:#1a1a1a"
)


def pagina(titulo: str, detalle: str, *, icono: str, codigo: int = 200) -> HTMLResponse:
    """Arma una página mínima y autocontenida para el navegador.

    `titulo` y `detalle` son textos nuestros, nunca datos que vengan del
    proveedor o de la persona: no se escapan.
    """
    return HTMLResponse(
        status_code=codigo,
        content=(
            f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>LifeSync</title></head>"
            f'<body style="{_ESTILO}">'
            f'<p style="font-size:3rem;margin:0">{icono}</p>'
            f"<h1>{titulo}</h1><p>{detalle}</p>"
            f"</body></html>"
        ),
    )
