"""Persistencia: implementaciones de los repositorios del dominio (PB-003).

Adaptadores contra Supabase (PostgreSQL vía PostgREST). El borde del cifrado
vive acá: los tokens OAuth2 se cifran antes de salir hacia la base y se
descifran al leerlos, de modo que el dominio siempre los ve en texto plano
(RF-01 + RF-18).

Módulos:
    encryption          TokenCipher, cifrado simétrico con Fernet.
    supabase_client     Ciclo de vida del cliente asíncrono y ping de salud.
    error_translation   Errores de PostgREST/httpx -> excepciones del proyecto.
    mapeo               Conversión entre filas JSON y tipos de Python.
    supabase_*_repository  Implementaciones de los puertos del dominio.
"""
