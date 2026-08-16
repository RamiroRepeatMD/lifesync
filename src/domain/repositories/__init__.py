"""Interfaces (puertos) de persistencia declaradas por el dominio.

Son clases abstractas puras: definen QUE se necesita guardar o recuperar, nunca
COMO. La implementacion concreta contra Supabase vive en
`src/infrastructure/persistence/` (PB-003).
"""
