"""
Entrypoint de la aplicación para Vercel.

Vercel busca una instancia de FastAPI llamada `app` en un fichero con nombre
soportado en la raíz del repo (app.py, index.py, server.py, main.py, wsgi.py,
asgi.py). La app real vive en tssi-site/serve.py, y "tssi-site" no es un nombre
de módulo válido en Python (lleva guion), así que este fichero añade ese
directorio al sys.path y reexporta `app`.

En local sigue funcionando igual:
  uvicorn main:app --port 5173                  # desde la raíz
  uvicorn serve:app --app-dir tssi-site --port 5173   # como hasta ahora
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tssi-site"))

from serve import app  # noqa: E402  -- Vercel importa `app` desde este módulo

__all__ = ["app"]
