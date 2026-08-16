# LifeSync

**Asistente personal digital conversacional por WhatsApp.**

Proyecto **P18** – Seminario de Integración Profesional (SIP)
Ingeniería Informática – Universidad del Salvador – 2026
Alumno: Ramiro Gracia · Docente: Lic. Christian López Pasarón

---

## Qué es

LifeSync centraliza la gestión de **tareas, eventos, recordatorios, correos y notas** en un
único espacio conversacional. El usuario escribe en **lenguaje natural en español** por
WhatsApp y el sistema interpreta el mensaje y lo transforma en acciones concretas sobre
Google Workspace (Calendar, Gmail, Drive, Tasks) y Notion.

No toma decisiones por el usuario: ejecuta y organiza, y **pide confirmación explícita antes
de cualquier acción que modifique datos** (RF-08).

La documentación completa está en [`docs/`](docs/).

---

## Stack

| Capa | Tecnología |
|------|------------|
| Canal de chat | WhatsApp Cloud API (Meta) |
| Backend | Python 3.11+ · FastAPI |
| Agente IA | LangGraph + LangChain |
| LLM | Google Gemini 1.5 Flash |
| Base de datos + Auth | Supabase (PostgreSQL) · tokens OAuth2 cifrados |
| Integraciones | Google APIs · Notion API (OAuth2) |
| Logging | structlog (estructurado, JSON en producción) |
| Testing | pytest |
| Hosting | Railway / Render |

---

## Arquitectura

**Clean Architecture + DDD.** La regla de dependencia apunta siempre hacia adentro:

```
interfaces ──▶ infrastructure ──▶ application ──▶ domain
                                                    ▲
                        (nadie sale del centro hacia afuera)
```

```
src/
├── main.py                     # Entrypoint ASGI
├── domain/                     # Entidades, value objects, interfaces de repos, reglas
├── application/                # Casos de uso, DTOs, puertos de servicios externos
├── infrastructure/             # Config, persistencia, APIs externas, LLM
│   ├── config/                 #   settings.py · logging.py
│   ├── persistence/            #   Supabase + cifrado de tokens
│   ├── external/{google,whatsapp,notion}/
│   └── llm/                    #   LangGraph + Gemini    (PB-005)
└── interfaces/                 # Adaptadores de entrada
    ├── api/                    #   app.py · routers · middleware · errores
    └── webhooks/               #   WhatsApp              (PB-004)

db/migrations/                  # SQL para aplicar en Supabase
```

Principios que no se negocian:

1. `domain/` no importa infraestructura ni frameworks.
2. Confirmación explícita antes de todo side-effect (RF-08).
3. Tokens OAuth2 siempre cifrados en Supabase.
4. Logging estructurado desde el día 1.
5. Errores amigables + recuperación de contexto (RF-19).
6. Tool calling controlado por LangGraph.

Detalle completo en [`docs/03-arquitectura-y-stack.md`](docs/03-arquitectura-y-stack.md).

---

## Puesta en marcha

**Requisitos:** [uv](https://docs.astral.sh/uv/). La versión de Python la fija `.python-version`
(3.13) y uv la instala solo si no la tenés.

```bash
uv sync
cp .env.example .env
```

Levantar el servidor:

```bash
uv run python -m src.main
```

La app arranca **sin credenciales de Supabase**, en modo degradado: `/health` responde 200 y
`/health/ready` responde 503. Para habilitar la persistencia, ver
[Base de datos](#base-de-datos).

Verificar que responde:

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "service": "LifeSync",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-08-16T14:32:07.481920Z"
}
```

Documentación interactiva de la API (deshabilitada en producción): <http://localhost:8000/docs>

---

## Base de datos

### 1. Crear el proyecto y aplicar el esquema

Creá un proyecto en [supabase.com](https://supabase.com), abrí **SQL Editor**, pegá el contenido
de [`db/migrations/001_usuarios_y_oauth_tokens.sql`](db/migrations/001_usuarios_y_oauth_tokens.sql)
y ejecutalo. Crea dos tablas —`usuarios` y `oauth_tokens`— con RLS activado y sin políticas
permisivas (deny-by-default). El script es idempotente: se puede volver a correr.

### 2. Generar la clave de cifrado

Los tokens OAuth2 se cifran en la aplicación antes de llegar a PostgreSQL (RF-18):

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

> Si perdés esta clave, todos los tokens guardados quedan ilegibles y cada usuario tiene que
> volver a conectar sus cuentas. Guardala en el gestor de secretos del hosting, nunca en el repo.

### 3. Completar el `.env`

En **Project Settings → API** del dashboard están la URL y las keys:

```bash
SUPABASE_URL=https://<tu-proyecto>.supabase.co
SUPABASE_KEY=<service_role key>
TOKEN_ENCRYPTION_KEY=<la clave generada en el paso 2>
```

> **Va la `service_role` key, no la `anon`.** El backend escribe tokens de todos los usuarios y
> necesita saltear RLS. Es equivalente a la contraseña de la base: nunca la expongas en un
> cliente ni la commitees.

### 4. Verificar

```bash
curl http://localhost:8000/health/ready
```

Debe devolver 200 con `"status": "ready"`. Si devuelve 503, el campo `detail` dice por qué.

Como evidencia de RF-18, abrí la tabla `oauth_tokens` en el Table Editor: la columna
`access_token_cifrado` tiene que ser ilegible y empezar con `gA`.

### Modelo de datos

| Tabla | Contenido |
|-------|-----------|
| `usuarios` | Personas que usan LifeSync. Identidad natural: `telefono_whatsapp` (único). |
| `oauth_tokens` | Credenciales OAuth2, **cifradas**. Único por `(usuario_id, proveedor)`; se borran en cascada con el usuario. |

El dominio nunca ve texto cifrado: `OAuthTokenRepository` recibe y devuelve tokens en claro, y el
cifrado ocurre dentro del adaptador de `infrastructure/persistence/`.

---

## WhatsApp

El webhook vive en `POST /webhooks/whatsapp` (y `GET` para el handshake).

### Configurar en Meta

1. Dashboard de Meta → tu app → WhatsApp → **Configuration** → Edit webhook.
2. Callback URL: `https://<tu-dominio>/webhooks/whatsapp` (requiere HTTPS público:
   en desarrollo, un túnel; en producción, lo de PB-007).
3. Verify token: el mismo valor que pusiste en `WHATSAPP_VERIFY_TOKEN`.
4. Suscribite al campo **messages**.

> **Trampa conocida con números argentinos.** En modo desarrollo, la lista de destinatarios
> permitidos de Meta matchea **sin** el 9 (`5411…`), pero el webhook entrega el número **con** el 9
> (`54911…`). Si sólo cargaste una forma, la primera respuesta falla con el error `131030`. Se
> arregla registrando el número en el dashboard en las dos formas — **no** es algo para corregir en
> el código.

### Qué contesta hoy

Un router mínimo de comandos: `/ayuda`, `/estado`, y para cualquier otra cosa un mensaje honesto de
"todavía no sé interpretar lenguaje natural". Es el andamiaje que PB-005 reemplaza por el grafo de
LangGraph.

### Cómo funciona por dentro

```
POST → valida firma HMAC sobre los bytes crudos → responde 200 → BackgroundTask:
       deduplica por wamid → busca o crea el Usuario → decide respuesta → envía por Graph API
```

Tres decisiones que conviene conocer antes de tocarlo:

- **El POST sólo devuelve 200 o 403.** Cualquier otro código hace que Meta reintente durante horas.
  Un payload deforme se loguea y se acepta.
- **El trabajo pesado va en segundo plano** y nunca deja escapar una excepción: corre con la
  respuesta ya enviada, así que un error que escape se pierde y corta la conexión.
- **La deduplicación es en memoria** (512 mensajes, 6 h). No sobrevive a un reinicio ni sirve con
  varias instancias; la red de contención es una ventana de frescura de 12 h sobre el timestamp del
  mensaje. La versión persistente es deuda anotada para el Sprint 2.

### Deuda técnica anotada

- La identidad del usuario se apoya en el teléfono. Meta está migrando a IDs de usuario (BSUID) y
  algún día los webhooks pueden llegar sin `wa_id`; el parser ya lo detecta y lo loguea.
- No se manejan mensajes que no son de texto: se descartan con `whatsapp.tipo_no_soportado`.
- Fuera de la ventana de 24 h de atención hay que usar plantillas. Hoy sólo se loguea el error 131047.

---

## Despliegue

El sistema se despliega en **Railway** desde el `Dockerfile` del repo.

### Por qué Railway

Render duerme los servicios gratuitos a los 15 minutos de inactividad y tarda ~60 s en despertar.
Para un webhook de WhatsApp eso significa que el primer mensaje después de cada pausa se pierde y
Meta reintenta. Railway no duerme salvo que actives Serverless a mano: el trial da USD 5 por 30
días y después Hobby son USD 5/mes (la app consume ~USD 2 de ese crédito).

### 1. Aplicar las migraciones

Antes del primer deploy, correr en el SQL Editor de Supabase los archivos de
[`db/migrations/`](db/migrations/) en orden. Ver [Base de datos](#base-de-datos).

### 2. Crear el servicio

1. [railway.com](https://railway.com) → **New Project** → **Deploy from GitHub repo**.
   Verificá la cuenta con GitHub: un trial sin verificar restringe la salida de red y Supabase
   podría no responder.
2. Railway detecta el `Dockerfile` solo. `railway.json` ya declara el health check en `/health`,
   una réplica y la política de reinicio.
3. **Settings → Networking → Generate Domain.** Sin esto el servicio **no es público** — es el
   tropiezo más común del primer deploy.

### 3. Cargar las variables

En **Variables**, y **las nueve juntas**: el validador corta en el primer grupo que falla, así que
de a una necesitarías cuatro deploys para descubrir las cuatro que faltan.

| Variable | Valor |
|---|---|
| `ENVIRONMENT` | `production` |
| `SUPABASE_URL` | URL del proyecto |
| `SUPABASE_KEY` | **service_role** key |
| `TOKEN_ENCRYPTION_KEY` | la clave Fernet |
| `WHATSAPP_TOKEN` | token de Graph API |
| `WHATSAPP_PHONE_NUMBER_ID` | ID de tu número |
| `WHATSAPP_VERIFY_TOKEN` | el que pusiste en Meta |
| `WHATSAPP_APP_SECRET` | App Secret de Meta |
| `LOG_LEVEL` | `INFO` |

Marcá como **Sealed** las cinco sensibles: una vez selladas, Railway no vuelve a mostrar el valor
ni por la UI ni por la API.

**No definas `PORT`** (lo inyecta Railway), ni `RELOAD`, ni `LOG_JSON` (en producción el formato ya
es JSON).

> Una variable declarada **pero vacía** cuenta como ausente y hace fallar el arranque. Es a
> propósito: si `WHATSAPP_APP_SECRET=` pasara como configurada, el webhook validaría las firmas
> contra un secreto vacío, que cualquiera puede reproducir.

### 4. Verificar

```bash
curl https://<tu-servicio>.up.railway.app/health
curl https://<tu-servicio>.up.railway.app/health/ready
```

`/health` debe dar 200 y `/health/ready` debe dar 200 con las dos dependencias en `ready: true`.
`/docs` debe dar **404** — en producción la documentación no se publica.

### 5. Apuntar el webhook de Meta

Callback URL: `https://<tu-servicio>.up.railway.app/webhooks/whatsapp`, con el mismo verify token.
Ver [WhatsApp → Configurar en Meta](#configurar-en-meta).

**Rotá el verify token después de configurarlo.** Viaja en el query string del handshake GET, y
aunque nuestro logger lo silencia, el proxy de Railway loguea las peticiones por su cuenta y sus
docs no aclaran si incluyen el query string. El riesgo es bajo (ese token sólo gatea el handshake;
lo que autentica los POST es la firma HMAC), pero rotarlo es gratis.

### Probar el contenedor localmente

Antes de desplegar, el mismo contenedor que va a correr en Railway:

```bash
docker build -t lifesync . && docker run --rm -p 8000:8000 --env-file .env lifesync
```

El `.env` se pasa en tiempo de ejecución con `--env-file`; **nunca entra a la imagen**, porque
[`.dockerignore`](.dockerignore) lo excluye. Sin ese archivo el contexto de build sería de 169 MB
e incluiría tus credenciales y el `.venv` de macOS.

### Limitaciones conocidas

- **Una sola instancia, un solo worker.** El deduplicador de mensajes vive en memoria del proceso:
  con dos instancias, un reintento de Meta cae en la otra y el usuario recibe la respuesta
  duplicada. No agregar réplicas hasta que el dedup sea persistente (Sprint 2).
- **Railway sólo chequea el health al desplegar, nunca después.** Si el proceso se cuelga sin
  morir, hay que reiniciarlo a mano.
- **Un redeploy mata las tareas en vuelo.** Meta ya recibió el 200, así que ese mensaje no se
  responde ni se reintenta. No redeployar durante una demo.

---

## Desarrollo

```bash
uv run pytest                # tests (no necesitan red ni base de datos)
uv run pytest --cov=src      # tests con cobertura
uv run ruff check .          # linting
uv run ruff format .         # formato
uv run mypy src              # tipado estricto
```

Los mismos checks corren en CI en cada push
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), más un build del `Dockerfile` para que no
se rompa sin que nos enteremos. CI no despliega: de eso se encarga Railway al detectar el push.

### Configuración

Toda la configuración se lee de variables de entorno mediante `pydantic-settings`
(ver [`.env.example`](.env.example) y `src/infrastructure/config/settings.py`).

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` · `testing` · `staging` · `production` |
| `LOG_LEVEL` | `INFO` | Nivel mínimo de log |
| `LOG_JSON` | *(según entorno)* | `true` fuerza logs JSON; consola en desarrollo |
| `HOST` | `0.0.0.0` | Interfaz de escucha |
| `PORT` | `8000` | Puerto HTTP |
| `RELOAD` | `false` | Autorecarga de uvicorn (sólo desarrollo) |
| `SUPABASE_URL` | — | URL del proyecto Supabase |
| `SUPABASE_KEY` | — | **service_role** key (ver [Base de datos](#base-de-datos)) |
| `SUPABASE_JWT_SECRET` | — | Se lee pero todavía no se usa; entra en PB-009 |
| `TOKEN_ENCRYPTION_KEY` | — | Clave Fernet para cifrar tokens en reposo |
| `WHATSAPP_TOKEN` | — | Token de acceso a la Graph API |
| `WHATSAPP_PHONE_NUMBER_ID` | — | ID de nuestro número de negocio |
| `WHATSAPP_VERIFY_TOKEN` | — | Handshake GET del webhook (lo inventás vos) |
| `WHATSAPP_APP_SECRET` | — | Firma HMAC de los POST. **No es el verify token** |

Las tres últimas de Supabase son opcionales fuera de producción (modo degradado) y
**obligatorias** con `ENVIRONMENT=production`: sin ellas el arranque falla, para no desplegar
nunca sin cifrado. Los secretos de WhatsApp, Google y Gemini están documentados en
`.env.example` y se activan en la tarea que los consume.

### Logging

Un evento por línea, con `request_id` propagado automáticamente a todo el request:

```
2026-08-16T14:32:07Z [info] http.request  request_id=3f2a… method=GET path=/health status_code=200 duracion_ms=1.84
```

En producción la misma línea sale como JSON, lista para ingestar en cualquier colector.

---

## Estado del proyecto

**Sprint 1** (entrega 19/08/2026) – Infraestructura + WhatsApp + LangGraph base + OAuth2 inicio.

| Tarea | Descripción | Estado |
|-------|-------------|--------|
| PB-001 | Repositorio + estructura Clean Architecture / DDD | ✅ |
| PB-002 | Setup FastAPI + dependencias + config por entornos | ✅ |
| PB-003 | Supabase (PostgreSQL + Auth + storage de tokens) | ✅ |
| PB-004 | Integración WhatsApp Cloud API (webhook + envío/recepción) | ✅ |
| PB-005 | LangGraph/LangChain + Gemini 1.5 Flash + tool-calling base | ⬜ |
| PB-006 | Logging estructurado + errores + health checks | 🟡 base lista |
| PB-007 | Despliegue inicial (Railway/Render) + variables seguras | ✅ |
| PB-009 | Flujo OAuth2 con Google (inicio) | ⬜ |

Planificación completa en [`docs/02-sprint-planning.md`](docs/02-sprint-planning.md).
