# Sprint Planning – LifeSync (SP)

**Total de esfuerzo planificado:** 210 horas  
**Cantidad de sprints:** 6  
**Duración aproximada por sprint:** 2 semanas

---

## Épicas (orden de prioridad MoSCoW)

| Épica | Nombre                                      | Prioridad     | Estimación aprox. |
|-------|---------------------------------------------|---------------|-------------------|
| 0     | Infraestructura, DevOps y Base del Agente   | Must          | 38 h              |
| 1     | Autenticación y Gestión de Cuentas          | Must          | 26 h              |
| 2     | Núcleo Conversacional y Control Transversal | Must          | —                 |
| 3     | Gestión de Agenda y Calendario              | Must          | —                 |
| 4     | Gestión de Tareas y Recordatorios           | Must          | —                 |
| 5     | Gestión de Correo (Gmail)                   | Should (alta) | 14 h              |
| 6     | Google Drive                                | Could         | 14 h              |
| 7     | Notion                                      | Could / Post-MVP | 12 h           |
| 8     | Calidad, Documentación y Gestión            | Must (continuo) | 24 h            |

---

## Sprint 1 (Entrega: 19/08/2026 23:59)

**Esfuerzo planificado:** 37 horas  
**Objetivo:** Establecer la base técnica del sistema (infraestructura, WhatsApp Cloud API y estructura del agente) y comenzar la autenticación segura con Google.

### Tareas del Sprint 1

| ID     | Tarea                                                      | Est. (h) | Épica / Prioridad |
|--------|------------------------------------------------------------|----------|-------------------|
| PB-001 | Configurar repositorio GitHub + estructura Clean Architecture / DDD | 4        | Infra (Must)     |
| PB-002 | Setup proyecto FastAPI + dependencias + configuración por entornos | 3        | Infra (Must)     |
| PB-003 | Configurar Supabase (PostgreSQL + Auth + storage de tokens) | 4        | Infra (Must)     |
| PB-004 | Integración WhatsApp Cloud API (webhook + envío/recepción) | 8        | Infra (Must)     |
| PB-005 | Setup LangGraph/LangChain + Gemini 1.5 Flash + tool-calling base | 7        | Infra (Must)     |
| PB-006 | Logging estructurado + manejo básico de errores + health checks | 3        | Infra (Must)     |
| PB-007 | Despliegue inicial (Railway/Render) + variables de entorno seguras | 5        | Infra (Must)     |
| PB-009 | Flujo OAuth2 completo con Google (inicio – scopes y autorización) | 3        | Auth (Must)      |

**Incremento esperado del Sprint 1:**  
Sistema desplegado, capaz de recibir y responder mensajes básicos por WhatsApp, con estructura de agente lista y OAuth2 iniciado.

---

## Resumen de Sprints

| Sprint | Fecha de entrega     | Horas | Enfoque principal                                      |
|--------|----------------------|-------|--------------------------------------------------------|
| 1      | 19/08/2026           | 37    | Infraestructura + WhatsApp + LangGraph base + OAuth2 inicio |
| 2      | 02/09/2026           | 36    | Completar OAuth2 + Núcleo conversacional + inicio Calendar |
| 3      | —                    | 35    | Calendar + Tasks                                      |
| 4      | —                    | 35    | Gmail + refinamiento                                   |
| 5      | —                    | 34    | Drive / Notion (si hay capacidad) + calidad            |
| 6      | —                    | 33    | Cierre MVP + documentación + demos                     |

---

## Criterios de aceptación generales del MVP

- El usuario puede autenticarse con Google vía OAuth2.
- Puede hablar por WhatsApp en español natural.
- El sistema mantiene contexto de la conversación.
- Solicita confirmación antes de acciones que modifican datos.
- Puede gestionar eventos de calendario y tareas básicas.
- Logging y manejo de errores básicos funcionando.
- Código organizado según Clean Architecture.
