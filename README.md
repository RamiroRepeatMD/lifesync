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
| Agente IA | LangGraph + langchain-core |
| LLM | Google **Gemini 3.5 Flash Lite** (la cuota gratuita es por modelo: lite da 500 req/día vs. 20 del flash) |
| Base de datos + Auth | Supabase (PostgreSQL) · tokens OAuth2 **cifrados** · conversación **cifrada** (checkpointer) |
| Integraciones | Google Calendar (completa) · **Google Tasks** (listar/crear/completar) · Notion API (pendiente) |
| Logging | structlog (estructurado, JSON en producción) |
| Testing | pytest · ruff · mypy strict — 602 tests |
| Hosting | Railway (desde Dockerfile) |

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

Creá un proyecto en [supabase.com](https://supabase.com), abrí **SQL Editor** y ejecutá en orden
los archivos de [`db/migrations/`](db/migrations/) (001 y 002; la 003 no hace falta aplicarla a
mano — la aplica la app en cada arranque y el archivo existe como documentación). Crean
`usuarios` y `oauth_tokens` con RLS activado y sin políticas permisivas (deny-by-default).
Los scripts son idempotentes: se pueden volver a correr.

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

### Memoria conversacional persistida (PB-013)

Con `SUPABASE_DB_URL` configurada, la conversación de cada usuario vive en Postgres
**cifrada** (Fernet, clave derivada de `TOKEN_ENCRYPTION_KEY`): sobrevive a los redeploys,
igual que una confirmación pendiente de "¿Confirmás?". Sin la variable, el bot funciona
igual pero la memoria vive en RAM y muere con cada deploy.

- El valor es la URI del **Session pooler** (Supabase → Connect → Session pooler) — no la
  conexión directa (IPv6-only, Railway no la garantiza) ni el transaction pooler (rompe con
  los prepared statements de psycopg).
- La app se conecta con un **rol dedicado de mínimo privilegio** (`lifesync_checkpointer`),
  no con el password maestro de la base.
- Las tablas del checkpointer se crean solas en el primer arranque, y la app les aplica
  RLS + revocación de permisos en cada arranque (`db/migrations/003` es la copia documentada).
- El modelo recibe una ventana de los **últimos 20 mensajes**; lo demás queda guardado pero
  fuera del contexto. Sin límite de almacenamiento todavía (deuda de Sprint 3).

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

Lenguaje natural, con el agente de LangGraph + Gemini (PB-005), y **gestión completa del
calendario** de quien escribe si conectó su cuenta de Google (RF-03): consultas ("¿qué
tengo hoy?"), creación, **modificación** ("cambiale la hora al dentista") y eliminación —
toda escritura pasa por una **confirmación obligatoria** que el modelo no puede saltear
(RF-08: es una pausa del grafo, no una instrucción del prompt). Y si el modelo falla justo
después de ejecutar una acción, el bot igual cuenta lo que hizo: así nadie repite —y duplica—
algo que ya quedó hecho. Desde el Sprint 3 también
gestiona **tareas de Google Tasks**: listarlas, anotar nuevas ("acordate que tengo que…"),
marcarlas como hechas, posponerlas y eliminarlas — el criterio es simple: con hora es un evento,
sin hora es una tarea.

Los comandos siguen siendo determinísticos y no pasan por el modelo: `/ayuda`, `/estado`,
`/conectar` y `/desconectar` (en dos pasos, con revocación real del permiso en Google).

Que la ayuda no dependa del LLM es a propósito: RF-11 pide un sistema de ayuda, y uno que cambia
de texto en cada invocación —o que inventa funciones que no existen— no lo cumple. Además es lo
único que sigue contestando si falta la API key.

### Cómo funciona por dentro

```
POST → valida firma HMAC sobre los bytes crudos → responde 200 → BackgroundTask:
       deduplica por wamid → busca o crea el Usuario → comando fijo o agente → envía por Graph API
```

Tres decisiones que conviene conocer antes de tocarlo:

- **El POST sólo devuelve 200 o 403.** Cualquier otro código hace que Meta reintente durante horas.
  Un payload deforme se loguea y se acepta.
- **El trabajo pesado va en segundo plano** y nunca deja escapar una excepción: corre con la
  respuesta ya enviada, así que un error que escape se pierde y corta la conexión.
- **La deduplicación es en memoria** (512 mensajes, 6 h). No sobrevive a un reinicio ni sirve con
  varias instancias; la red de contención es una ventana de frescura de 12 h sobre el timestamp del
  mensaje. La versión persistente es deuda anotada para el Sprint 3.

### Deuda técnica anotada

- La identidad del usuario se apoya en el teléfono. Meta está migrando a IDs de usuario (BSUID) y
  algún día los webhooks pueden llegar sin `wa_id`; el parser ya lo detecta y lo loguea.
- No se manejan mensajes que no son de texto: se descartan con `whatsapp.tipo_no_soportado`.
- Fuera de la ventana de 24 h de atención hay que usar plantillas. Hoy sólo se loguea el error 131047.

---

## Conectar Google (PB-009)

La persona conecta su cuenta desde WhatsApp: escribe `/conectar`, el bot le manda
un enlace firmado que vence en 10 minutos, autoriza en Google y vuelve al chat.
Se piden tres permisos: lectura de calendario (`calendar.readonly`), gestión de
eventos (`calendar.events`) y tareas (`tasks`). Para desvincular: `/desconectar` —
pide confirmación y **revoca el permiso en Google de verdad**, no sólo borra la
copia local.

### Configurar en Google Cloud Console

1. **Habilitar la Google Calendar API** en el proyecto.
2. **Pantalla de consentimiento de OAuth** → External, en modo *Testing*, y
   agregarte a vos mismo como *usuario de prueba*.
3. **Credenciales → Crear credenciales → ID de cliente de OAuth 2.0**, tipo
   *Aplicación web*.
4. **URI de redireccionamiento autorizado**, exactamente:

   ```
   https://<tu-servicio>.up.railway.app/oauth/google/callback
   ```

   Google compara el string completo: una barra de más devuelve
   `redirect_uri_mismatch`.
5. Copiar el ID y el secreto al `.env` y a Railway.

> Con la app en modo *Testing*, **los refresh tokens de Google caducan a los 7
> días**. Alcanza para el cuatrimestre, pero hay que reconectar cada semana
> hasta publicar la app.

### Verificar que quedó bien

En Supabase, la fila de `oauth_tokens` tiene que existir con
`access_token_cifrado` empezando en `gA` y **`refresh_token_cifrado` distinto de
NULL**. Si el refresh es NULL, faltó `access_type=offline` en la autorización.

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

En **Variables**, y **todas juntas**: el validador corta en el primer grupo que falla, así que
de a una necesitarías cinco deploys para descubrir las cinco que faltan.

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
| `GOOGLE_API_KEY` | key de Gemini (PB-005) |
| `GEMINI_MODEL` | *(opcional)* `gemini-3.5-flash-lite` es el default; la cuota gratuita es **por modelo** |
| `SUPABASE_DB_URL` | *(opcional pero recomendada)* Session pooler de Supabase — la memoria del bot (ver [Memoria conversacional](#memoria-conversacional-persistida-pb-013)) |
| `LOG_LEVEL` | `INFO` |

Marcá como **Sealed** las seis sensibles: una vez selladas, Railway no vuelve a mostrar el valor
ni por la UI ni por la API.

> `GOOGLE_API_KEY` es obligatoria desde PB-005: **cargala antes de desplegar** o el servicio no
> levanta. Y no alcanza con tenerla: la API "Generative Language" tiene que estar habilitada en el
> proyecto de Google Cloud, o cada respuesta del agente falla con 403.

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
- **Un redeploy mata las tareas en vuelo.** Meta ya recibió el 200, así que ese mensaje puntual
  no se responde. Desde PB-013 la conversación y las confirmaciones pendientes **sí sobreviven**
  al redeploy: sólo se pierde la respuesta del mensaje que estaba en el aire.

---

## Desarrollo

```bash
uv run pytest                # tests (no necesitan red ni base de datos)
uv run pytest --cov=src      # tests con cobertura
uv run ruff check .          # linting
uv run ruff format --check . # formato (sin --check, formatea)
uv run mypy src              # tipado estricto del código
uv run mypy tests            # ... y de los tests (CI corre ambos)
```

`tests/integration/` prueba caminos completos sin red: por ejemplo, de la herramienta del agente
al adaptador de Google Calendar, asertando el cuerpo exacto que recibiría Google.

Hay además una suite de **evaluación del comportamiento del modelo** (RF-10) que llama a
Gemini de verdad y gasta cuota — por eso es opt-in y CI la saltea:

```bash
uv run pytest tests/eval/ -m gemini
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
| `GOOGLE_API_KEY` | — | Key de Gemini para el agente conversacional |
| `SUPABASE_DB_URL` | — | Session pooler de Supabase (PB-013). Sin ella, la memoria del bot vive en RAM y muere en cada redeploy; con ella, las conversaciones se guardan **cifradas** |
| `GOOGLE_CLIENT_ID` | — | ID de cliente OAuth2 (PB-009). Opcional: sin él sólo se deshabilita /conectar |
| `GOOGLE_CLIENT_SECRET` | — | Secreto del cliente OAuth2 |
| `GOOGLE_REDIRECT_URI` | — | Debe coincidir **exactamente** con la registrada en Google |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Modelo a usar. La cuota gratuita es **por modelo y por día** (lite: 500/día; flash: 20/día) |

Las tres de Supabase son opcionales fuera de producción (modo degradado) y **obligatorias** con
`ENVIRONMENT=production`: sin ellas el arranque falla, para no desplegar nunca sin cifrado. Lo
mismo vale para las de WhatsApp y para `GOOGLE_API_KEY`. Los secretos de Google OAuth están
documentados en `.env.example` y se activan en la tarea que los consume (PB-009).

### Logging

Un evento por línea, con `request_id` propagado automáticamente a todo el request:

```
2026-08-16T14:32:07Z [info] http.request  request_id=3f2a… method=GET path=/health status_code=200 duracion_ms=1.84
```

En producción la misma línea sale como JSON, lista para ingestar en cualquier colector.

---

## Estado del proyecto

**Sprint 1** (entrega 19/08/2026) – Infraestructura + WhatsApp + LangGraph base — **✅ completo**.

| Tarea | Descripción | Estado |
|-------|-------------|--------|
| PB-001 | Repositorio + estructura Clean Architecture / DDD | ✅ |
| PB-002 | Setup FastAPI + dependencias + config por entornos | ✅ |
| PB-003 | Supabase (PostgreSQL + storage cifrado de tokens) | ✅ |
| PB-004 | Integración WhatsApp Cloud API (webhook + envío/recepción) | ✅ |
| PB-005 | LangGraph + Gemini + tool-calling base | ✅ |
| PB-006 | Logging estructurado + errores + health checks | ✅ absorbido por PB-002/003/004 |
| PB-007 | Despliegue inicial (Railway) + variables seguras | ✅ |

**Sprint 2** (entrega 02/09/2026) – OAuth completo + calendario + memoria — **✅ completo,
verificado contra los servicios reales** (Google, Gemini, Supabase, Meta).

| Tarea | Descripción | Estado |
|-------|-------------|--------|
| PB-009 | Flujo OAuth2 completo con Google | ✅ |
| PB-010 | Refresco automático de tokens | ✅ absorbido por PB-009 (refresco perezoso) |
| PB-011 | Cuentas: `/estado` · `/conectar` · `/desconectar` con revocación real | ✅ |
| PB-012 | Confirmación explícita (RF-08) | ✅ absorbido por PB-016 (`interrupt` del grafo) |
| PB-013 | Conversación persistida y cifrada en Supabase (RF-09) | ✅ |
| PB-014 | Manejo de mensajes ambiguos (RF-10) | ✅ evaluado contra el modelo real |
| PB-015 | Lectura de Google Calendar | ✅ |
| PB-016 | Crear/eliminar eventos con confirmación obligatoria | ✅ |
| PB-017 | Modificar eventos | ✅ (adelantado de Sprint 3) |

**Sprint 3** (entrega 16/09/2026) – Tareas + cierre del núcleo. PB-018/022/023/024 se
adelantaron en el Sprint 2; PB-028 (Google Tasks: modelo + listar/crear/completar) ✅;
PB-019/020 cubiertos por `/ayuda`, `/estado` y RF-19. Pendiente del sprint: PB-029
(posponer/eliminar tareas) llega en el Sprint 4 según plan.

Planificación completa en [`docs/02-sprint-planning.md`](docs/02-sprint-planning.md).

<img width="303" height="626" alt="image" src="https://github.com/user-attachments/assets/9f26247e-41ec-4545-a528-68d3e2cb91c2" />

