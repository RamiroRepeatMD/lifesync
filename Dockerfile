# syntax=docker/dockerfile:1.9
# ============================================================================
# LifeSync — imagen de producción (PB-007)
#
# Multi-stage siguiendo el patrón oficial de uv. El stage `builder` resuelve e
# instala las dependencias; la imagen final sólo recibe el entorno virtual ya
# armado y el código, sin uv, sin cachés y sin herramientas de desarrollo.
#
# Construir y probar en local:
#   docker build -t lifesync .
#   docker run --rm -p 8000:8000 --env-file .env lifesync
# ============================================================================

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.12.5


# ---------------------------------------------------------------------------
# Stage 1 — dependencias
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# uv fijado por versión y no `:latest`: las plataformas cachean imágenes base
# públicas y un tag mutable puede darte una versión distinta sin avisar.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

# copy: evita los avisos de hardlink al cruzar el montaje de caché.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Las dependencias primero, en su propia capa y sin el proyecto. Así un cambio
# en src/ no reinstala nada: sólo se rehace la capa siguiente.
#
# --locked falla si uv.lock no coincide con pyproject.toml en vez de resolver
# de nuevo en silencio. Es lo que hace que versionar el lock sirva de algo.
# --no-dev saca 16 paquetes (mypy, ruff, pytest…) que no van a producción.
#
# El `id=` del cache mount NO es opcional: el builder de Railway rechaza el
# Dockerfile sin él ("flag is missing an id argument"). Docker en local lo
# infiere del target, así que sin id funciona acá y falla allá.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# README.md hace falta: pyproject.toml lo declara como `readme` y hatchling lo
# lee al construir el paquete.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Ahora sí el proyecto. --no-editable lo instala como paquete real y no como
# enlace a un directorio. Mismo `id` que arriba: comparten el caché de uv.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev


# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim

# Sin privilegios: si algo se escapa, que no sea como root.
RUN useradd --create-home --uid 1000 lifesync

WORKDIR /app

COPY --from=builder --chown=lifesync:lifesync /app/.venv /app/.venv
COPY --chown=lifesync:lifesync src/ ./src/

ENV PATH="/app/.venv/bin:${PATH}" \
    # Sin esto los logs salen con retraso en el panel de la plataforma, que es
    # justo cuando estás depurando por qué no arranca un deploy.
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # .env.example trae RELOAD=true. Acá se neutraliza el default; el código
    # además lo ignora cuando ENVIRONMENT=production.
    RELOAD=false

USER lifesync

# Documental: la plataforma inyecta PORT y rutea sola. `settings.port` ya lee
# PORT, así que no hay que expandir variables en el CMD.
EXPOSE 8000

# Forma exec, sin shell: el proceso es PID 1 y recibe el SIGTERM del redeploy,
# de modo que corre el shutdown ordenado del lifespan (cierre del cliente de
# Supabase y del de WhatsApp). Con `sh -c` la shell sería PID 1 y se comería la
# señal.
#
# `python -m src.main` y no el CLI de uvicorn porque `main()` pasa
# log_config=None, que es lo que evita que uvicorn pise la config de structlog.
CMD ["python", "-m", "src.main"]
