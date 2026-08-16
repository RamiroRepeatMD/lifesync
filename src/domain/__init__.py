"""Capa de dominio: el corazón del negocio.

No importa NADA de las otras capas ni de frameworks externos (ni FastAPI, ni
Supabase, ni LangGraph). Si algo de acá necesita hablar con el mundo exterior,
se declara como una interfaz en `repositories/` y la implementa `infrastructure/`.
"""
