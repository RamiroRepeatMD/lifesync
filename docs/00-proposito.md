# Propósito del Sistema – LifeSync

**Proyecto:** LifeSync  
**Materia:** Seminario de Integración Profesional (SIP)  
**Alumno:** Ramiro Gracia  
**Carrera:** Ingeniería Informática – Universidad del Salvador  
**Año:** 2026  
**Docente:** Lic. Christian López Pasarón  
**Número de proyecto:** P18

---

## Visión

LifeSync es un **asistente personal digital conversacional**, accesible principalmente mediante **WhatsApp** (WhatsApp Cloud API), que actúa como una agenda personal inteligente.

Su propósito es **centralizar la gestión de tareas, eventos, recordatorios y notas** en un único espacio, evitando la dispersión entre múltiples aplicaciones.

El usuario se comunica en **lenguaje natural en español**. LifeSync interpreta los mensajes y los transforma en acciones concretas.

---

## Qué hace el sistema

- Agendar, consultar, modificar y cancelar eventos en el calendario (Google Calendar).
- Enviar, responder y gestionar correos electrónicos (Gmail).
- Crear, listar, modificar y completar tareas y recordatorios.
- Buscar, subir, compartir y gestionar archivos en Google Drive.
- Gestionar contenido en Notion (páginas, bases de datos, etc.).
- Recibir respuestas contextuales y proactivas basadas en su agenda.
- Mantener el contexto de la conversación.
- Solicitar **confirmación explícita** antes de ejecutar acciones críticas.

---

## Integraciones principales

- **Google Workspace** (Calendar, Gmail, Drive, Tasks) mediante OAuth2 seguro.
- **Notion** mediante OAuth2.
- Canal principal de interacción: **WhatsApp Cloud API**.

---

## Alcance y limitaciones

LifeSync se enfoca **exclusivamente en la organización personal**.

- No reemplaza aplicaciones especializadas.
- No realiza acciones complejas fuera de este dominio.
- **No toma decisiones por el usuario**. Actúa como un asistente que ejecuta y organiza la información según las indicaciones recibidas.

---

## Público objetivo

Cualquier persona que desee organizar sus responsabilidades de forma más simple e intuitiva, sin requerir conocimientos técnicos. Especialmente orientado a estudiantes y profesionales.

---

## Principios de diseño clave

1. **Usabilidad conversacional** – interacción natural en español.
2. **Privacidad y seguridad** – OAuth2, tokens cifrados, confirmaciones explícitas.
3. **Confiabilidad** – manejo robusto de errores y recuperación de contexto.
4. **Proactividad controlada** – recordatorios y resúmenes útiles, nunca invasivos.
5. **Arquitectura limpia** – Clean Architecture + principios DDD.
