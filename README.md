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
| Integraciones | Google Calendar (completa) · Google Tasks (completa) · **Gmail** (buscar, leer, mandar, responder y reenviar) · **Google Drive** (buscar, subir lo que llega por WhatsApp, compartir, renombrar, mover y mandar a la papelera) · **Notion** (conectar por OAuth; buscar por título, leer páginas y consultar bases; crear páginas, filas con sus datos y bases nuevas, agregar texto, cambiar datos, mover, mandar a la papelera y subir archivos) |
| Logging | structlog (estructurado, JSON en producción) |
| Testing | pytest · ruff · mypy strict — 2211 tests (cobertura con ramas 93,4%, piso de 90% en CI) + 42 de evaluación contra el modelo real |
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
los archivos de [`db/migrations/`](db/migrations/) (001, 002 y 004; la 003 no hace falta
aplicarla a mano — la aplica la app en cada arranque y el archivo existe como documentación).
Crean `usuarios`, `oauth_tokens` y `recordatorios` con RLS activado y sin políticas permisivas
(deny-by-default).
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
| `recordatorios` | Avisos programados por WhatsApp (PB-030): texto **cifrado**, momento y estado (`pendiente` → `enviando` → `enviado` / `fallido`, o `cancelado`). |

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
tengo hoy?"), creación —también de eventos que se repiten ("todos los lunes y miércoles a
las 19")—, **modificación** ("cambiale la hora al dentista") y eliminación. La
confirmación (RF-08) va **según la criticidad**: lo reversible y suelto —agendar, mover,
anotar, completar— sale directo y el bot cuenta qué hizo y cuándo; lo irreversible
—borrar, mandar un correo— pide el "sí"; y si se piden **varias cosas juntas**, el bot
muestra la lista completa y confirma una sola vez. Con texto de un tercero a la vista (un
correo, un listado de Drive, una página de Notion), todo vuelve a confirmar: un tercero no
puede escribir por la persona. La decisión no es del
modelo: es una política del dominio que aplica el grafo antes de ejecutar. Y si el modelo falla justo
después de ejecutar una acción, el bot igual cuenta lo que hizo: así nadie repite —y duplica—
algo que ya quedó hecho. Desde el Sprint 3 también
gestiona **tareas de Google Tasks**: listarlas, anotar nuevas ("acordate que tengo que…"),
marcarlas como hechas, posponerlas y eliminarlas — el criterio es simple: con hora es un evento,
sin hora es una tarea. En el Sprint 4 sumó **Gmail**: buscar
correos, abrirlos para contar qué dicen y mandar correos nuevos, siempre con una confirmación que
muestra exactamente qué sale; desde el Sprint 5 también **responde** un correo dentro de su
hilo y lo **reenvía** (sin adjuntos). Si un envío falla, el bot dice si el correo salió: sólo
cuando no se puede saber (Gmail no llegó a confirmarlo) manda a revisar Enviados, para que no
salga dos veces; y no pide el "sí" de un correo que no puede salir (por ejemplo, con el permiso
de Google vencido). Como un correo lo escribe cualquiera, su contenido le
llega al modelo enmarcado como dato de un tercero, nunca como instrucción.

También en el Sprint 5 sumó **Google Drive**: "¿dónde está el presupuesto?", "pasame el link
del TP" o "¿qué PDF toqué esta semana?" devuelven nombre, tipo, fecha y **link** de cada
archivo, buscando en el nombre o en el contenido. No abre los archivos: sólo los encuentra. Los
nombres también los eligen terceros (un archivo compartido se llama como quiso quien lo
creó), así que reciben el trato de un correo, y además se les quita todo lo que parezca un
link: si no, un archivo llamado con un link falso lograba que el modelo lo pasara como el
del archivo (lo detectó la evaluación contra Gemini).

Y el bot **recibe archivos**: un PDF o una foto mandados por WhatsApp se suben al Drive de la
persona ("subilo a la carpeta Facultad"), y cualquier archivo se **comparte** con alguien por
correo o con un link de sólo lectura, siempre con confirmación: la pregunta muestra el nombre que
el archivo tiene en Drive y a quién se le da acceso. Lo que se manda queda en Meta y se baja
recién al subirlo; el modelo nunca ve ni elige qué archivo se baja.

En el Sprint 6 el bot también **ordena el Drive**: renombra archivos y carpetas ("al informe
ponele Informe final": si no se dice la extensión, queda la que tenía), mueve archivos a otra
carpeta o a Mi unidad, y los manda a la **papelera**. Nunca borra para siempre: desde la
papelera se recuperan durante 30 días, y la pregunta muestra el tipo y la fecha para
distinguir dos archivos que se llaman igual. Sólo toca lo que es de la persona (lo que le
compartieron otros, no), las carpetas sólo se renombran (moverlas o borrarlas arrastraría lo
de adentro), y mover o subir algo a una carpeta compartida pide el "sí": es darle acceso a
quienes la ven.

También en el Sprint 6 se conecta **Notion** (`/conectar notion`). Para **leer**: "¿qué
páginas tengo en Notion?", "¿qué dice mi página de Recetas?" o "¿qué hay en mi base de
Gastos?". Busca por el título (la API de Notion no busca adentro), lee una página entera y lista
las filas de una base con sus datos. Ve sólo lo que la persona eligió compartir al conectar.
Como en un espacio compartido lo escribe cualquiera, el contenido recibe el trato de un correo:
va enmarcado como dato de un tercero, y con una página de Notion a la vista toda escritura pide
el "sí".

Y para **escribir** (RF-07): crear una página adentro de otra o una fila nueva en una base, con
sus datos ("sumá Gas a mis gastos: 1500, vence el 10"), crear una **base nueva** adentro de una
página, con sus columnas y sus filas ("armame una base con un día de la semana que viene por
fila"), agregar texto al final de una página ("agregá flan a mis recetas"), cambiar un dato de una fila
("marcá la luz como pagada", "poné 1500 en el monto") o el título de una página, moverla
adentro de otra, mandarla a la papelera de Notion y subirle un archivo que llegó por WhatsApp
(una foto se ve como imagen, un PDF como PDF). Todo lo que escribe en Notion pide el "sí": lo
que queda en una página lo ve quien la ve, y la API no dice si está compartida. La pregunta
muestra entero lo que se va a escribir. El texto lo arma el bot en bloques propios (títulos,
viñetas, listas numeradas y tareas), sólo agrega al final —no cambia ni borra lo que ya hay— y
los datos de una fila sólo toman opciones que la base ya tiene. Una base nueva va a la vista
adentro de la página, con hasta 5 columnas (texto, número, casilla, fecha o selección) además de
la primera, que en Notion es siempre texto y se llama como pida la persona ("que la primera
columna sea la fecha"). Las filas: hasta 10 con datos, 31 si llevan sólo el nombre, o una por
día de un período ("todos los días de noviembre", hasta 31), que arma el bot con el día en la
primera columna y sin inventar nada en las otras. Si no entran, avisa y pregunta: nunca recorta
la lista por su cuenta. Las filas se crean de la última a la primera, así se leen en orden, y la
vista queda con las columnas en el orden pedido (y ordenada por la fecha, si hay una columna de
fecha del día). Todo en una sola confirmación. Las bases enteras no se mueven ni se borran, y una
fila no sale de su base. Si Notion no confirma una escritura, el bot no dice
que se hizo ni invita a repetirla: avisa que puede haber quedado y pide revisarlo antes.

Y es lo primero que **escribe sin que le escriban**: los **recordatorios** (PB-030).
"Recordame en 20 minutos que saque la pizza" → confirmación → a esa hora llega
"⏰ Recordatorio: sacar la pizza". Se pueden listar y cancelar, y no necesitan la cuenta de
Google. Llegan hasta 24 horas adelante: fuera de esa ventana WhatsApp sólo deja escribir
primero con plantillas aprobadas por Meta, así que para algo más lejano el agente ofrece una
tarea o un evento. Los manda un despachador que corre dentro del proceso cada 30 segundos;
lo pendiente vive en la base, así que un redeploy no pierde ninguno, y un *compare-and-set*
evita mandarlo dos veces si dos contenedores conviven durante un deploy.

Los comandos siguen siendo determinísticos y no pasan por el modelo: `/ayuda`, `/estado`,
`/conectar` y `/desconectar` (en dos pasos, con revocación real del permiso en Google), y sus
versiones para Notion, `/conectar notion` y `/desconectar notion`.

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
- De lo que no es texto, sólo se reciben documentos y fotos (PB-038). Audio, video, stickers y el
  resto se descartan sin respuesta, y quedan contados en `no_soportados` del evento
  `whatsapp.webhook.recibido`.
- Fuera de la ventana de 24 h de atención hay que usar plantillas. Hoy sólo se loguea el error 131047.

---

## Conectar Google (PB-009)

La persona conecta su cuenta desde WhatsApp: escribe `/conectar`, el bot le manda
un enlace firmado que vence en 10 minutos, autoriza en Google y vuelve al chat.
Se piden seis permisos: lectura de calendario (`calendar.readonly`), gestión de
eventos (`calendar.events`), tareas (`tasks`), lectura de correo (`gmail.readonly`), envío
(`gmail.send`, que sólo permite mandar: ni modificar ni borrar) y Drive (`drive`). Drive va
completo porque es el único permiso que deja compartir cualquier archivo y subir a una carpeta
que ya existe; lo que el bot hace con él lo acota el código: buscar, subir, compartir,
renombrar, mover y mandar a la papelera, sin leer el contenido ni borrar nada para siempre.
Una cuenta conectada antes de sumar un permiso tiene que
volver a hacer `/conectar`.
Además de pedir el permiso, cada API hay que **habilitarla** en el proyecto de Google Cloud
(Calendar, Tasks, Gmail y Drive): son dos llaves distintas. Para desvincular: `/desconectar` —
pide confirmación y **revoca el permiso en Google de verdad**, no sólo borra la
copia local.

### Configurar en Google Cloud Console

1. **Habilitar las APIs** en el proyecto: Google Calendar, Google Tasks, Gmail y Google Drive.
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

## Conectar Notion (PB-014)

Igual que con Google, desde WhatsApp: `/conectar notion` manda un enlace firmado que vence en 10
minutos. En la pantalla de Notion, la persona **elige qué páginas y bases compartir**: el bot ve
sólo eso (y lo que tienen adentro), nunca el resto del espacio. Para desvincular,
`/desconectar notion`, en dos pasos y con revocación real del permiso.

### Configurar la conexión en Notion

1. Crear una **conexión pública** en https://app.notion.com/developers/connections.
2. **Redirect URI**, exactamente:

   ```
   https://<tu-servicio>.up.railway.app/oauth/notion/callback
   ```

3. **Capacidades**: *Read content*, *Insert content* y *Update content*, y *No user
   information*. Es todo lo que usa el bot: leer, crear y agregar, y cambiar datos, mover y
   mandar a la papelera.
4. Copiar el *client ID* y el *client secret* al `.env` y a Railway (`NOTION_CLIENT_ID`,
   `NOTION_CLIENT_SECRET` y `NOTION_REDIRECT_URI`).

Sumarle una capacidad a una conexión ya creada obliga a volver a autorizar: hasta que la persona
haga `/conectar notion` de nuevo, Notion rechaza las escrituras y el bot se lo explica.

Notion no informa cuándo vence el token: cuando la API contesta 401, el bot lo renueva una vez
con el refresh token y reintenta. Si eso también falla, pide volver a hacer `/conectar notion`.

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
[`db/migrations/`](db/migrations/) en orden. Ver [Base de datos](#base-de-datos). Una migración
nueva (como la 004 de los recordatorios) se aplica **antes** de pushear el código que la usa.

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
uv run pytest --cov          # tests con cobertura de ramas (no necesitan red ni base de datos)
uv run ruff check .          # linting
uv run ruff format --check . # formato (sin --check, formatea)
uv run mypy src              # tipado estricto del código
uv run mypy tests            # ... y de los tests (CI corre ambos)
```

`tests/integration/` prueba caminos completos sin red: del agente a los adaptadores de Google
Calendar, Tasks, Gmail, Drive y Notion, asertando el pedido exacto que recibiría el proveedor (en
Gmail, hasta los encabezados del correo que sale y el texto entero de un reenvío; en Drive, los
padres que se agregan y se sacan al mover; en Notion, la renovación del token ante un 401, cada
escritura y la subida de un archivo en sus tres pasos). Y
`tests/unit/test_tools_criticas.py` fija las promesas que valen para todas las tools a la vez:
ninguna escribe fuera del control de la confirmación, ninguna deja escapar una falla de Google,
y dos acciones juntas se confirman en una sola lista y corren una vez. Si mañana se suma una
tool que no las cumple, la suite falla.

Hay además una suite de **evaluación del comportamiento del modelo** que llama a Gemini de
verdad y gasta cuota — por eso es opt-in y CI la saltea:

```bash
uv run pytest tests/eval/ -m gemini
```

Son 42 casos: ambigüedad (pregunta cuando falta un dato y no sobre-pregunta; con todos
los datos crea directo y con la fecha correcta), tarea vs. evento, completar vs. eliminar,
posponer "al viernes" con la fecha correcta, listar tareas sin tocar nada,
pedidos compuestos y "agendá estas tareas" (una sola confirmación con la lista), series
recurrentes, recordatorios ("en 20 minutos" guarda la hora correcta; "avisame a las…" no es
un evento), "el último correo" con la bandeja cambiando, y los intentos de inyección por
correo (un correo que ordena borrar, reenviarse o agendar algo no lo logra), y responder
un correo dentro de su hilo; y si un envío falla, que cuente bien si el correo salió (ante la
duda manda a revisar Enviados y no lo repite). En Drive: pasa el link tal cual (y un nombre
de archivo con un link falso no lo cuela), sube lo que le mandan y lo comparte con el sí, y
mandar a la papelera o renombrar lo que encontró también pide el sí. En Notion: busca la página,
la lee y cuenta lo que dice, una página que ordena mandar un correo no lo logra, agregarle algo
a una página o mandarla a la papelera pide el sí, un "subilo" a secas, con Drive y Notion
conectados, pregunta adónde, y una base nueva (la de la semana que viene, o una de 10 filas por
5 columnas) o una fila con sus datos se confirman en una sola pregunta y con las fechas bien. Una
base con todos los días del mes que viene sale con la fecha primero, un día por fila, en orden y
sin datos inventados; con 14 filas que no entran avisa sin recortar; "Gastos de noviembre" no se
vuelve un día por fila, y una fila sin dato queda vacía.
Si Gemini no responde o se agota la cuota por minuto, el caso
espera y reintenta, y si persiste se marca como salteado: una caída del proveedor no es un
fallo de comportamiento.

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
| `NOTION_CLIENT_ID` | — | Client ID de la conexión pública de Notion (PB-014). Opcional: sin él sólo se deshabilita /conectar notion |
| `NOTION_CLIENT_SECRET` | — | Client secret de esa conexión |
| `NOTION_REDIRECT_URI` | — | Debe coincidir **exactamente** con la registrada en la conexión |
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

La tabla del Sprint 2 usa la numeración interna con la que se trabajó; los informes de avance
usan la del planning oficial (por ejemplo, la memoria persistida es el PB-016 oficial).

**Sprint 3** (entrega 16/09/2026) – Calendar + tareas — **✅ completo**.

| Tarea | Descripción | Estado |
|-------|-------------|--------|
| PB-018 | Detección y manejo de mensajes ambiguos | ✅ adelantado en el Sprint 2 |
| PB-019 | Sistema de ayuda y feedback | ✅ |
| PB-020 | Errores amigables y recuperación de contexto | ✅ |
| PB-022 · 023 · 024 | Calendar: crear, consultar, modificar y cancelar | ✅ adelantados en el Sprint 2 |
| PB-028 | Integración Google Tasks + modelo de tareas | ✅ |

**Sprint 4** (entrega 30/09/2026) – Tareas completas + Gmail + calidad.

| Tarea | Descripción | Estado |
|-------|-------------|--------|
| PB-025 | Eventos recurrentes básicos | ✅ |
| PB-026 | Confirmación + respuesta contextual post-acción | ✅ |
| PB-027 | Pruebas de integración Calendar + casos límite de fechas | ✅ |
| PB-029 | Tareas: crear, listar, completar, posponer y eliminar | ✅ |
| PB-030 | Recordatorios proactivos básicos | ✅ (hasta 24 h; plantillas de Meta al Sprint 5) |
| PB-032 | Gmail: enviar correos | ✅ (adjuntos replanificados al Sprint 5) |
| PB-033 | Gmail: listar, buscar y leer correos | ✅ |

**Sprint 5** (entrega 14/10/2026) – Gmail completo + pruebas + documentación (Drive como Could).

| Tarea | Descripción | Estado |
|-------|-------------|--------|
| PB-031 | Pruebas e integración de Tareas con el agente | ✅ (paginación y límites de Google corregidos) |
| PB-034 | Responder y reenviar correos | ✅ (en el hilo; sin adjuntos ni "responder a todos") |
| PB-035 | Confirmación + errores de envío (Gmail) | ✅ (dice si el correo salió; comprueba la cuenta antes del sí) |
| PB-036 | Pruebas de integración Gmail | ✅ (de punta a punta; 3 bugs corregidos) |
| PB-037 | Drive: buscar y listar archivos | ✅ (nombre, tipo, fecha y link; sin abrirlos) |
| PB-038 | Drive: subir y compartir archivos | ✅ (recibe documentos y fotos por WhatsApp; comparte con personas o por link) |
| PB-040 | Pruebas de las tools críticas | ✅ (invariantes sobre las 22 tools; cobertura medida en CI) |
| PB-041 | Diagramas UML actualizados | — (no aplica: no había diagramas que actualizar) |

**Sprint 6** (entrega 28/10/2026) – Cierre: Drive, auditoría, documentación y demos.

| Tarea | Descripción | Estado |
|-------|-------------|--------|
| PB-039 | Drive: integración + pruebas (sumó renombrar, mover y mandar a la papelera) | ✅ (de punta a punta contra un Drive simulado; sólo archivos propios) |
| PB-021 | Logging de acciones importantes (auditoría básica) | ⏳ |
| PB-042 | Mantenimiento del documento de Sprint Planning | ⏳ |
| PB-043 | Preparación de demos y evidencia | ⏳ |
| PB-044 | Refinamiento final del backlog y replanificación documentada | ⏳ |
| PB-014 | Notion: OAuth2 básico (sumó leer: buscar, leer páginas y consultar bases) | ✅ (de punta a punta contra un Notion simulado) |
| RF-07 | Notion: escribir (crear páginas, filas con sus datos y bases nuevas, agregar texto, cambiar datos), mover, mandar a la papelera y subir archivos — sin PB propio, lo documenta PB-044 | ✅ (de punta a punta contra un Notion simulado; todo pide el "sí") |

Además: corrección de defectos y pulido, y la documentación técnica final con un manual de
usuario básico.

Planificación completa en [`docs/02-sprint-planning.md`](docs/02-sprint-planning.md).

<img width="303" height="626" alt="image" src="https://github.com/user-attachments/assets/9f26247e-41ec-4545-a528-68d3e2cb91c2" />

