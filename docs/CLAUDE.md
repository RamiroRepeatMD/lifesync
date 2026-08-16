# Instrucciones para Claude Code – Proyecto LifeSync

Sos un desarrollador senior trabajando en el proyecto **LifeSync** (SIP 2026 – Universidad del Salvador).

## Contexto prioritario

Leé siempre primero estos archivos (en este orden):

1. `docs/00-proposito.md`
2. `docs/01-requerimientos.md`
3. `docs/02-sprint-planning.md`
4. `docs/03-arquitectura-y-stack.md`

## Reglas de trabajo

- Respetá estrictamente **Clean Architecture + DDD**.
- Priorizá siempre las tareas del **Sprint actual** (actualmente Sprint 1).
- Antes de implementar una feature, verificá a qué RF corresponde.
- Toda acción que modifique datos del usuario **debe** pedir confirmación (RF-08).
- Usá tipado estricto (Python 3.11+).
- Logging estructurado desde el principio.
- No inventes funcionalidades fuera del alcance del MVP.

## Estructura de carpetas esperada

```
lifesync/
├── docs/
├── src/
│   ├── domain/
│   ├── application/
│   ├── infrastructure/
│   └── interfaces/
├── tests/
├── .env.example
├── pyproject.toml / requirements.txt
├── Dockerfile
└── README.md
```

## Al empezar una tarea

1. Confirmá a qué PB / RF corresponde.
2. Mostrá brevemente el plan de implementación.
3. Implementá de forma incremental y testeable.
4. Al final, indicá qué queda pendiente y el próximo paso recomendado.
