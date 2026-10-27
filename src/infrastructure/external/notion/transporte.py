"""Lo que comparten el OAuth y la API de Notion (PB-014).

La versión de la API va en TODO pedido, también los de OAuth (la referencia
la marca obligatoria; verificado el 27/10/2026). Los errores traen
`{status, code, message}` en la API y `{error}` en el estilo OAuth: de los dos
se loguea sólo el código, nunca el mensaje, que puede repetir lo que se pidió.
"""

from __future__ import annotations

from typing import Any

import httpx

VERSION_DE_NOTION = "2026-03-11"
ENCABEZADOS = {"Notion-Version": VERSION_DE_NOTION}


def codigo_de(respuesta: httpx.Response) -> str | None:
    """El código del error: `code` en la API de Notion, `error` en el estilo OAuth."""
    datos = json_o_vacio(respuesta)
    for clave in ("code", "error"):
        valor = datos.get(clave)
        if isinstance(valor, str):
            return valor
    return None


def json_o_vacio(respuesta: httpx.Response) -> dict[str, Any]:
    """El cuerpo como dict, o vacío si no es JSON con esa forma."""
    try:
        datos = respuesta.json()
    except ValueError:
        return {}
    return datos if isinstance(datos, dict) else {}
