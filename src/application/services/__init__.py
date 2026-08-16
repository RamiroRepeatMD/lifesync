"""Servicios de aplicación: lógica de interacción que no es regla de negocio.

Se distingue de `domain/services/`, que aloja invariantes del negocio (como la
política de acciones críticas de RF-08). Acá va lo que decide *cómo* conversa
el sistema, que es andamiaje reemplazable: el router de comandos de PB-004 lo
sustituye el grafo de LangGraph en PB-005.
"""
