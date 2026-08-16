-- ============================================================================
-- PB-003 · Persistencia de LifeSync en Supabase (PostgreSQL)
-- RF-01 (autenticación OAuth2) · RF-18 (seguridad y privacidad)
--
-- Los tokens se guardan CIFRADOS por la aplicación con Fernet, en
-- src/infrastructure/persistence/encryption.py. PostgreSQL nunca ve el valor
-- en claro: por eso las columnas se llaman *_cifrado y NO llevan índice
-- (el cifrado no es determinístico, no se puede consultar por valor).
--
-- Cómo aplicarla: Supabase Studio → SQL Editor → pegar y ejecutar.
-- Es idempotente: se puede correr más de una vez sin romper nada.
-- ============================================================================

-- gen_random_uuid() viene con pgcrypto.
create extension if not exists pgcrypto with schema extensions;


-- ---------------------------------------------------------------------------
-- Trigger compartido: mantiene `actualizado_en` al día en cada UPDATE.
-- ---------------------------------------------------------------------------
create or replace function public.set_actualizado_en()
returns trigger
language plpgsql
security invoker      -- no escala privilegios
set search_path = ''  -- evita search_path hijacking (lo exige el linter de Supabase)
as $$
begin
  new.actualizado_en = now();
  return new;
end;
$$;


-- ---------------------------------------------------------------------------
-- usuarios
-- ---------------------------------------------------------------------------
create table if not exists public.usuarios (
    id               uuid        primary key default gen_random_uuid(),
    telefono_whatsapp text       not null unique,
    nombre           text,
    creado_en        timestamptz not null default now(),
    actualizado_en   timestamptz not null default now(),

    constraint usuarios_telefono_no_vacio
        check (length(btrim(telefono_whatsapp)) > 0),
    constraint usuarios_nombre_no_vacio
        check (nombre is null or length(btrim(nombre)) > 0)
);

comment on table public.usuarios is
    'Personas que usan LifeSync. Identidad natural: el número de WhatsApp.';
comment on column public.usuarios.telefono_whatsapp is
    'Número de WhatsApp del usuario. La validación E.164 llega con PB-004.';

drop trigger if exists usuarios_set_actualizado_en on public.usuarios;
create trigger usuarios_set_actualizado_en
    before update on public.usuarios
    for each row execute function public.set_actualizado_en();


-- ---------------------------------------------------------------------------
-- oauth_tokens
-- ---------------------------------------------------------------------------
create table if not exists public.oauth_tokens (
    id                    uuid        primary key default gen_random_uuid(),
    usuario_id            uuid        not null
                                      references public.usuarios (id) on delete cascade,
    proveedor             text        not null,
    access_token_cifrado  text        not null,
    refresh_token_cifrado text,
    expira_en             timestamptz,
    scopes                text[]      not null default '{}',
    creado_en             timestamptz not null default now(),
    actualizado_en        timestamptz not null default now(),

    -- text + CHECK en vez de un ENUM: sumar un proveedor es un ALTER simple.
    -- Debe coincidir con ProveedorOAuth en src/domain/value_objects/.
    constraint oauth_tokens_proveedor_valido
        check (proveedor in ('google', 'notion')),

    -- Un token por (usuario, proveedor). Es lo que habilita el UPSERT.
    constraint oauth_tokens_usuario_proveedor_unico
        unique (usuario_id, proveedor),

    -- Red de contención de RF-18: todo token Fernet empieza con 'gA' (versión
    -- 0x80 en base64url). Si alguien escribiera texto plano por un bug o por
    -- un INSERT manual, la base lo rechaza.
    constraint oauth_tokens_access_token_cifrado
        check (access_token_cifrado ~ '^gA[A-Za-z0-9_-]+={0,2}$'),
    constraint oauth_tokens_refresh_token_cifrado
        check (refresh_token_cifrado is null
               or refresh_token_cifrado ~ '^gA[A-Za-z0-9_-]+={0,2}$')
);

comment on table public.oauth_tokens is
    'Credenciales OAuth2 por usuario y proveedor. Cifradas por la app (RF-18).';
comment on column public.oauth_tokens.access_token_cifrado is
    'Token Fernet en base64url. NUNCA texto plano. No indexable.';

-- No hace falta índice sobre usuario_id: el UNIQUE (usuario_id, proveedor) ya
-- genera uno cuya primera columna es usuario_id y cubre esa búsqueda.
create index if not exists oauth_tokens_expira_en_idx
    on public.oauth_tokens (expira_en)
    where expira_en is not null;  -- para el refresco proactivo (PB-009)

drop trigger if exists oauth_tokens_set_actualizado_en on public.oauth_tokens;
create trigger oauth_tokens_set_actualizado_en
    before update on public.oauth_tokens
    for each row execute function public.set_actualizado_en();


-- ---------------------------------------------------------------------------
-- Row Level Security
--
-- El backend usa la SERVICE_ROLE key, que por diseño saltea RLS. Igual se
-- habilita RLS y NO se crea ninguna policy: deny-by-default. Si la anon key
-- se filtrara alguna vez, ni `anon` ni `authenticated` leen un solo token.
--
-- No hace falta —ni sirve— una policy `to service_role using (true)`:
-- service_role saltea RLS por completo. Es un error común.
-- ---------------------------------------------------------------------------
alter table public.usuarios     enable row level security;
alter table public.oauth_tokens enable row level security;

revoke all on public.usuarios     from anon, authenticated;
revoke all on public.oauth_tokens from anon, authenticated;


-- ---------------------------------------------------------------------------
-- Verificación (opcional, para pegar en el SQL Editor después de aplicar)
--
--   select tablename, rowsecurity from pg_tables  where schemaname = 'public';
--   select tablename, policyname  from pg_policies where schemaname = 'public';
--   -- No debería devolver ninguna policy: eso es lo correcto acá.
-- ---------------------------------------------------------------------------
