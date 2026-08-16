-- ============================================================================
-- PB-004 · Formato canónico del teléfono de WhatsApp
-- RF-01 · RF-18
--
-- La migración 001 dejó escrito "La validación E.164 llega con PB-004". Esto
-- lo cumple.
--
-- Por qué importa: `unique` garantiza que no haya dos filas con el MISMO
-- string, pero no que un humano no pueda entrar dos veces con dos formas
-- distintas del mismo número (el caso clásico es el 9 de los móviles
-- argentinos: +5491141234567 vs +541141234567). El CHECK fija la forma
-- canónica en la base, así que un bug en la normalización falla ruidosamente
-- al insertar en vez de crear un usuario duplicado en silencio.
--
-- Es el mismo instinto defensivo que el CHECK del prefijo Fernet en
-- oauth_tokens: que la base rechace lo que el código no debería producir.
--
-- Cómo aplicarla: Supabase Studio → SQL Editor → pegar y ejecutar.
-- Es idempotente.
-- ============================================================================

-- El CHECK anterior sólo exigía "no vacío"; queda subsumido por el nuevo.
alter table public.usuarios
    drop constraint if exists usuarios_telefono_no_vacio;

alter table public.usuarios
    drop constraint if exists usuarios_telefono_e164;

-- E.164: "+", código de país que no empieza en 0, entre 8 y 15 dígitos.
-- Debe coincidir con _E164 en src/domain/value_objects/numero_whatsapp.py
alter table public.usuarios
    add constraint usuarios_telefono_e164
    check (telefono_whatsapp ~ '^\+[1-9][0-9]{7,14}$');

comment on column public.usuarios.telefono_whatsapp is
    'Número de WhatsApp en E.164 canónico, con "+". Identidad natural del usuario. '
    'Se construye desde el wa_id del webhook (ver NumeroWhatsApp.desde_wa_id).';


-- ---------------------------------------------------------------------------
-- Verificación (opcional, para pegar después de aplicar)
--
--   select conname, pg_get_constraintdef(oid)
--   from pg_constraint
--   where conrelid = 'public.usuarios'::regclass;
--
--   -- Debe fallar:
--   insert into public.usuarios (telefono_whatsapp) values ('5491141234567');
--   -- Debe funcionar:
--   insert into public.usuarios (telefono_whatsapp) values ('+5491141234567');
-- ---------------------------------------------------------------------------
