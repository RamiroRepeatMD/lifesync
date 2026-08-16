"""Puertos de servicios externos que necesitan los casos de uso.

A diferencia de `domain/repositories/` (persistencia), aca van interfaces de
capacidades: enviar un mensaje, invocar al LLM, publicar un recordatorio.
Sus implementaciones viven en `src/infrastructure/external/` y `.../llm/`.
"""
