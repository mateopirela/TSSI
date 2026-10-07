# Despliegue en Railway

El sitio TSSI corre como **un solo servicio FastAPI** (uvicorn) + un **Postgres con
pgvector**. El asistente usa **OpenAI** (embeddings `text-embedding-3-small` +
generación `gpt-4o-mini` con tool-calling). Las notificaciones de leads salen por
**Resend**.

```
┌───────────────────────┐      ┌──────────────────────────┐
│  Servicio: web        │      │  Servicio: Postgres      │
│  FastAPI (serve.py)   │─────▶│  pgvector/pgvector       │
│  uvicorn :$PORT       │ SQL  │  (vector, pg_trgm, uuid) │
└──────────┬────────────┘      └──────────────────────────┘
           │ HTTPS
           ▼
   OpenAI API     Resend API
 (RAG + embeddings) (emails de leads)
```

## 1. Crear el proyecto y la base de datos

1. En [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
   → elige `fckyeslol/TSSI`. Railway detecta Python por `requirements.txt` y usa el
   `startCommand` de `railway.json`.
2. Añade el Postgres con **pgvector** (el Postgres estándar de Railway no trae la
   extensión). Opción recomendada y determinista:
   - **New** → **Empty Service** → pestaña **Settings** → **Source** → **Docker Image**
     → `pgvector/pgvector:pg17`.
   - Variables del servicio: `POSTGRES_USER=sura`, `POSTGRES_PASSWORD=<algo-seguro>`,
     `POSTGRES_DB=sura`.
   - **Settings → Volumes** → monta un volumen en `/var/lib/postgresql/data` (para que
     los datos sobrevivan a los redeploys).
   - (Alternativa: usar la plantilla "pgvector" del marketplace de Railway.)

## 2. Variables de entorno del servicio web

En el servicio **web** → **Variables**:

| Variable          | Valor                                                        |
|-------------------|--------------------------------------------------------------|
| `DATABASE_URL`    | Referencia al Postgres: `${{Postgres.DATABASE_URL}}` (ajusta el nombre del servicio) |
| `OPENAI_API_KEY`  | tu clave de https://platform.openai.com/api-keys             |
| `RESEND_API_KEY`  | tu clave de Resend (**rota la anterior**, quedó expuesta)    |
| `LEAD_NOTIFY_TO`  | `mateopirela08@gmail.com` (opcional, ya es el default)       |
| `LEAD_FROM`       | `TSSI Leads <onboarding@resend.dev>` (opcional)              |

`PORT` lo inyecta Railway automáticamente; no lo definas tú.

## 3. Sembrar la base de datos (una sola vez)

El esquema + catálogo + embeddings se cargan con un único script. Hazlo desde tu
máquina apuntando a la **URL pública** del Postgres de Railway (Postgres → **Connect**
→ *Public Network*):

```bash
pip install -r requirements.txt

# PowerShell
$env:DATABASE_URL = "postgresql://sura:...@<host-publico>:<port>/sura"
$env:OPENAI_API_KEY = "sk-..."
python sura-db/seed/seed_all.py
```

Esto aplica `schema.sql`, carga `load.sql` (catálogo) y genera los 792 embeddings
(`text-embedding-3-small`, ~0,01 USD). Es idempotente: puedes re-ejecutarlo.

> Alternativa sin exponer el puerto: `railway run python sura-db/seed/seed_all.py`
> tras `railway login` y `railway link` (usa la red privada del proyecto).

## 4. Verificar

- Abre la URL pública del servicio web → debe cargar `index.html`.
- Prueba el asistente (`/api/chat`): debe responder citando fuentes SURA.
- Envía un lead de prueba desde el wizard → debe llegar el correo vía Resend.

## Despliegue en Vercel

Vercel importa una instancia de FastAPI llamada `app` desde un fichero con nombre
soportado en la raíz (`main.py`, `app.py`, `index.py`, `server.py`, `wsgi.py`,
`asgi.py`). Como la app vive en `tssi-site/serve.py` y `tssi-site` no es un nombre
de módulo válido en Python (lleva guion), la raíz incluye `main.py`, que solo
reexporta la app:

```python
sys.path.insert(0, os.path.join(HERE, "tssi-site"))
from serve import app
```

`vercel.json` mantiene el HTML, `assets/`, `data/` y `sura-db/app/` dentro del
bundle de la función (`includeFiles`), porque `serve.py` los lee en tiempo de
ejecución.

### Variables de entorno (Project Settings -> Environment Variables)

| Variable         | Nota                                                          |
|------------------|---------------------------------------------------------------|
| `DATABASE_URL`   | Postgres con pgvector **accesible desde internet**. Un host de red privada (p. ej. `*.railway.internal`) no resuelve desde Vercel. |
| `OPENAI_API_KEY` | Embeddings + generación.                                       |
| `RESEND_API_KEY` | Notificación de leads por correo.                              |
| `LEAD_NOTIFY_TO` | Opcional (default `mateopirela08@gmail.com`).                  |
| `LEAD_FROM`      | Opcional (default `TSSI Leads <onboarding@resend.dev>`).       |

La base tiene que estar sembrada (`sura-db/seed/seed_all.py`) contra esa misma
`DATABASE_URL`, sobre una base vacía: `schema.sql` usa `CREATE TYPE`/`CREATE TABLE`
sin `IF NOT EXISTS`, así que falla si se ejecuta dos veces.

### Verificar

`GET /api/health` devuelve qué claves están configuradas y si la base responde:

```json
{"openai_key":true,"resend_key":true,"database_url":true,
 "db":"ok","chunks":792,"embedding_dims":1536}
```

`embedding_dims` tiene que ser **1536** (`text-embedding-3-small`). Si sale 768,
la base quedó con los embeddings viejos de Ollama y la búsqueda vectorial falla
por dimensiones incompatibles.

## Notas

- **Leads:** se notifican por correo (Resend) y además se anexan a
  `tssi-site/leads.jsonl`. En Railway ese archivo es efímero (se borra en cada
  redeploy); el correo es la captura confiable. Si quieres histórico persistente,
  monta un volumen en el servicio web o guarda los leads en Postgres.
- **Costo OpenAI:** a este volumen, centavos al día. Ajusta `GEN_MODEL` si quieres
  otro modelo con tool-calling.
- **Local:** `docker compose -f sura-db/rag/docker-compose.yml up -d` levanta el
  Postgres; luego `uvicorn serve:app --app-dir tssi-site --port 5173`.
