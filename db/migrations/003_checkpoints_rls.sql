-- ============================================================================
-- PB-013 · Hardening de las tablas del checkpointer de LangGraph
-- RF-09 (contexto conversacional) · RF-18 (seguridad y privacidad)
--
-- IMPORTANTE: esta migración NO hay que aplicarla a mano. La aplica la
-- aplicación en cada arranque, justo después de que el saver crea o migra sus
-- tablas (src/infrastructure/llm/checkpointer.py, SENTENCIAS_DE_HARDENING).
-- Esta copia existe como documentación del esquema y como red de contención
-- por si hiciera falta aplicarla manualmente. Si se edita una, hay que editar
-- la otra: hay un test que las compara.
--
-- Por qué existe: `AsyncPostgresSaver.setup()` crea sus tablas en `public`
-- SIN RLS, y Supabase por defecto les da GRANT a `anon` y `authenticated`.
-- Sin este cierre, cualquiera con la anon key podría leer los checkpoints por
-- la API REST — y los checkpoints son la conversación completa.
--
-- Defensa en profundidad: además de este cierre, el CONTENIDO va cifrado con
-- Fernet (clave derivada de TOKEN_ENCRYPTION_KEY), así que ni un dump de la
-- base expone una conversación.
--
-- La app entra por psycopg como rol dueño de las tablas: RLS no la afecta.
-- Es idempotente.
-- ============================================================================

ALTER TABLE IF EXISTS public.checkpoints ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.checkpoints FROM anon, authenticated;
ALTER TABLE IF EXISTS public.checkpoint_blobs ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.checkpoint_blobs FROM anon, authenticated;
ALTER TABLE IF EXISTS public.checkpoint_writes ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.checkpoint_writes FROM anon, authenticated;
ALTER TABLE IF EXISTS public.checkpoint_migrations ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.checkpoint_migrations FROM anon, authenticated;

-- ---------------------------------------------------------------------------
-- Verificación (opcional, para pegar después del primer arranque)
--
--   select relname, relrowsecurity from pg_class
--   where relname like 'checkpoint%' and relkind = 'r';
--   -- Las cuatro filas deben decir `t`.
-- ---------------------------------------------------------------------------
