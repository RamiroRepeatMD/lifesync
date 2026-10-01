-- ============================================================================
-- PB-030 · Recordatorios proactivos básicos
-- RF-16 (proactividad) · RF-05 (tareas y recordatorios) · RF-18 (privacidad)
--
-- Un recordatorio es lo único que LifeSync manda sin que la persona haya
-- escrito antes: a la hora pedida, el despachador del proceso le escribe por
-- WhatsApp (src/interfaces/jobs/recordatorios.py).
--
-- El texto va CIFRADO por la aplicación (Fernet, el mismo cifrador de los
-- tokens): es contenido de la persona, y la conversación de la que sale ya
-- está cifrada en los checkpoints. Un dump de la base no lo expone.
--
-- Cómo aplicarla: Supabase Studio → SQL Editor, o el MCP de Supabase.
-- Es idempotente: se puede correr más de una vez sin romper nada.
-- ============================================================================

create table if not exists public.recordatorios (
    id             uuid        primary key default gen_random_uuid(),
    usuario_id     uuid        not null
                               references public.usuarios (id) on delete cascade,
    texto_cifrado  text        not null,
    momento        timestamptz not null,
    estado         text        not null default 'pendiente',
    enviado_en     timestamptz,
    creado_en      timestamptz not null default now(),
    actualizado_en timestamptz not null default now(),

    -- Debe coincidir con EstadoDeRecordatorio en src/domain/entities/.
    -- `enviando` es el reclamo: dos despachadores (el contenedor viejo y el
    -- nuevo durante un deploy) no pueden mandar el mismo recordatorio.
    constraint recordatorios_estado_valido
        check (estado in ('pendiente', 'enviando', 'enviado', 'cancelado', 'fallido')),

    -- Red de contención de RF-18, como en oauth_tokens: todo Fernet empieza
    -- con 'gA'. Un texto en claro por un bug o un INSERT manual se rechaza.
    constraint recordatorios_texto_cifrado
        check (texto_cifrado ~ '^gA[A-Za-z0-9_-]+={0,2}$')
);

comment on table public.recordatorios is
    'Avisos que LifeSync manda por WhatsApp a la hora pedida (PB-030). Texto cifrado (RF-18).';
comment on column public.recordatorios.texto_cifrado is
    'Token Fernet en base64url. NUNCA texto plano.';

-- La consulta del despachador, cada 30 s: pendientes cuyo momento ya llegó.
create index if not exists recordatorios_pendientes_por_momento_idx
    on public.recordatorios (momento)
    where estado = 'pendiente';

-- "¿Qué recordatorios tengo?" y cancelar: los pendientes de una persona.
create index if not exists recordatorios_pendientes_por_usuario_idx
    on public.recordatorios (usuario_id)
    where estado = 'pendiente';

drop trigger if exists recordatorios_set_actualizado_en on public.recordatorios;
create trigger recordatorios_set_actualizado_en
    before update on public.recordatorios
    for each row execute function public.set_actualizado_en();


-- ---------------------------------------------------------------------------
-- Row Level Security: deny-by-default, igual que en la migración 001. El
-- backend usa la service_role key, que saltea RLS; anon y authenticated no
-- leen un solo recordatorio.
-- ---------------------------------------------------------------------------
alter table public.recordatorios enable row level security;

revoke all on public.recordatorios from anon, authenticated;
