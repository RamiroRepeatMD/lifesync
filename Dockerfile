# ============================================================================
# LifeSync — imagen de producción (PB-007)
#
# Multi-stage con uv. El stage `builder` resuelve e instala las dependencias;
# la imagen final sólo recibe el entorno virtual ya armado y el código, sin uv
# y sin herramientas de desarrollo.
#
# Deliberadamente NO se usan `--mount=type=cache` ni `--mount=type=bind`: son
# extensiones de BuildKit y cada plataforma les pone requisitos distintos (el
# builder de Railway exige un `id` con un prefijo `cacheKey` propio). Sólo
# aceleraban el build; el caché por capas de Docker cubre el mismo caso al
# copiar `pyproject.toml` y `uv.lock` antes que el código. Este Dockerfile usa
# únicamente instrucciones estándar y construye en cualquier builder.
#
# Construir y probar en local:
#   docker build -t lifesync .
#   docker run --rm -p 8000:8000 --env-file .env lifesync
# ============================================================================

ARG PYTHON_VERSION=3.13


# ---------------------------------------------------------------------------
# Stage 1 — dependencias
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# uv fijado por versión y no `:latest`: las plataformas cachean imágenes base
# públicas y un tag mutable puede darte una versión distinta sin avisar.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Sólo el manifiesto y el lock: esta capa se rehace únicamente si cambian esos
# dos archivos, así que tocar src/ no reinstala las dependencias.
COPY pyproject.toml uv.lock ./

# --locked falla si uv.lock no coincide con pyproject.toml en vez de resolver
# de nuevo en silencio. Es lo que hace que versionar el lock sirva de algo.
# --no-dev saca 16 paquetes (mypy, ruff, pytest…) que no van a producción.
# --no-install-project instala sólo las dependencias, todavía no el proyecto.
RUN uv sync --locked --no-install-project --no-dev

# Ahora el código. README.md hace falta porque pyproject.toml lo declara como
# `readme` y hatchling lo lee al construir el paquete.
COPY README.md ./
COPY src/ ./src/

# El proyecto. --no-editable lo instala como paquete real, no como enlace.
RUN uv sync --locked --no-editable --no-dev


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
