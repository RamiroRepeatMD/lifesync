"""Punto de entrada ASGI de LifeSync.

Uso en desarrollo:
    python -m src.main
    uvicorn src.main:app --reload

Uso en producción (PB-007):
    uvicorn src.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

from src.interfaces.api.app import create_app

app = create_app()


def main() -> None:
    """Levanta el servidor leyendo host/puerto de la configuración."""
    import uvicorn

    from src.infrastructure.config.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        # Nunca autorecarga en producción: `.env.example` trae RELOAD=true, y si
        # ese valor llega al contenedor arrancaría un supervisor de recarga que
        # duplica la memoria y rompe el apagado ordenado.
        reload=settings.reload and not settings.is_production,
        # None: que uvicorn no pise la configuración de structlog.
        log_config=None,
    )


if __name__ == "__main__":
    main()
