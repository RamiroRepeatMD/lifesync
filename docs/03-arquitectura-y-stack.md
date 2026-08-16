# Arquitectura y Stack Tecnológico – LifeSync

## Stack Oficial

| Capa                  | Tecnología                          | Notas |
|-----------------------|-------------------------------------|-------|
| Canal de chat         | WhatsApp Cloud API (Meta)           | Principal |
| Backend               | Python 3.11+ + **FastAPI**          | — |
| Agente IA             | **LangGraph** + LangChain           | Tool calling, memoria, flujos controlados |
| LLM                   | **Google Gemini 1.5 Flash**         | Prioridad por costo y tool-calling |
| Base de datos + Auth  | **Supabase** (PostgreSQL + Auth)    | Tokens OAuth2 cifrados |
| Integraciones         | Google APIs + Notion API            | OAuth2 |
| Hosting               | Railway / Render / Fly.io           | — |
| Contenedores          | Docker                              | — |
| Testing               | pytest                              | — |
| Control de versiones  | Git + GitHub                        | — |

---

## Arquitectura

Se utiliza **Clean Architecture** combinada con principios de **Domain-Driven Design (DDD)**.

### Capas recomendadas

```
src/
├── domain/                 # Entidades, Value Objects, Interfaces de repositorios, Domain Services
├── application/            # Casos de uso (Use Cases / Application Services)
├── infrastructure/         # Implementaciones concretas (Supabase, Google APIs, WhatsApp, LLM)
│   ├── persistence/
│   ├── external/
│   │   ├── google/
│   │   ├── whatsapp/
│   │   └── notion/
│   └── llm/
└── interfaces/             # Adaptadores de entrada (API FastAPI, Webhooks)
    ├── api/
    └── webhooks/
```

### Principios que siempre deben respetarse

1. **Dependencias apuntan hacia adentro** (domain no conoce infrastructure).
2. **Confirmación explícita** antes de cualquier side-effect (RF-08).
3. **Tokens OAuth2 siempre cifrados** en Supabase.
4. **Logging estructurado** desde el día 1.
5. **Manejo de errores amigable** + recuperación de contexto (RF-19).
6. **Tool calling** controlado a través de LangGraph (no llamadas directas descontroladas al LLM).

---

## Flujo conversacional de alto nivel

1. Usuario envía mensaje por WhatsApp.
2. Webhook recibe el mensaje → FastAPI.
3. Se carga / actualiza el estado de la conversación (contexto).
4. LangGraph decide qué herramientas usar (Calendar, Gmail, Tasks, etc.).
5. Si la acción es crítica → se pide confirmación al usuario.
6. Se ejecuta la acción a través de los adaptadores de infrastructure.
7. Se responde al usuario por WhatsApp.
8. Se registra la acción (auditoría).

---

## Variables de entorno mínimas (Sprint 1)

```env
# App
ENVIRONMENT=development
LOG_LEVEL=INFO

# Supabase
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_JWT_SECRET=

# WhatsApp Cloud API
WHATSAPP_TOKEN=
WHATSAPP_PHONE_NUMBER_ID=
WHATSAPP_VERIFY_TOKEN=

# Google OAuth
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=

# Gemini
GOOGLE_API_KEY=   # o GEMINI_API_KEY
```

---

## Decisiones de diseño importantes

- **Un solo agente LangGraph** con tools bien definidas (no múltiples agentes al inicio).
- **Estado de conversación** persistido en Supabase.
- **Human-in-the-loop** explícito para acciones destructivas o de envío.
- Preferir **Gemini 1.5 Flash** por su excelente relación costo/capacidad de tool-calling.
