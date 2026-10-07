"""
Backend de datos en memoria para el asistente SURA.

Replica las consultas SQL de rag_engine leyendo los JSON de sura-db/seed y los
embeddings precalculados (seed/chunk_embeddings.npy). Permite servir el
asistente sin Postgres, que es lo que hace falta en serverless.

- Vectorial: coseno contra la matriz de embeddings (ya normalizada al generarla).
- Léxico: BM25 sobre el contenido, en sustitución del tsvector 'spanish'.
- Fusión: RRF con K=60, igual que la versión SQL.

Se activa solo cuando no hay DATABASE_URL; con base de datos manda el SQL.
"""
import json
import math
import os
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.normpath(os.path.join(HERE, "..", "seed"))

RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "no", "una", "su", "al", "es", "lo", "como", "mas", "pero",
    "sus", "le", "ya", "o", "este", "si", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "ese", "eso", "ante", "ellos", "e", "esto", "mi", "antes", "algunos",
    "que", "cual", "cuales", "son", "ser", "the", "of",
}

_D = None


# ------------------------------------------------------------------ utilidades
def _norm(s):
    """Minúsculas sin acentos, para comparar y tokenizar."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _stem(t):
    """Quita el plural castellano. Aproxima el stemming de to_tsvector('spanish'),
    que es de donde salía el acierto léxico en la versión SQL: 'exclusiones' y
    'exclusion', o 'seguros' y 'seguro', tienen que caer en el mismo término."""
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _tokens(s):
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", _norm(s))
            if len(t) >= 2 and t not in STOPWORDS]


def _read(name):
    with open(os.path.join(SEED, name), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ carga
def _build():
    chunks = [c for c in _read("chunks.json") if (c.get("contenido") or "").strip()]
    emb = np.load(os.path.join(SEED, "chunk_embeddings.npy")).astype(np.float32)
    if emb.shape[0] != len(chunks):
        raise RuntimeError(
            f"chunk_embeddings.npy ({emb.shape[0]}) no cuadra con chunks.json "
            f"({len(chunks)}); regenera con seed/build_embeddings.py")

    productos = _read("productos.json")
    procesos = _read("procesos.json")

    # Índice invertido + estadísticas BM25 sobre el contenido de los chunks.
    postings, doc_len = {}, []
    for i, c in enumerate(chunks):
        tf = {}
        for t in _tokens(c["contenido"]):
            tf[t] = tf.get(t, 0) + 1
        doc_len.append(sum(tf.values()) or 1)
        for t, n in tf.items():
            postings.setdefault(t, []).append((i, n))
    n_docs = len(chunks)
    avgdl = sum(doc_len) / n_docs
    idf = {t: math.log(1 + (n_docs - len(p) + 0.5) / (len(p) + 0.5))
           for t, p in postings.items()}

    canales = {}
    for r in _read("canales.json"):
        canales.setdefault(r["producto_slug"], []).append(r["canal"])

    elegibilidad = {}
    for r in _read("elegibilidad.json"):
        elegibilidad.setdefault(r["producto_slug"], r)

    return {
        "chunks": chunks,
        "emb": emb,
        "productos": productos,
        "producto_por_slug": {p["slug"]: p for p in productos},
        "proceso_por_slug": {p["slug"]: p for p in procesos},
        "plan_cob": _read("plan_coberturas.json"),
        "cob_nombre": {c["slug"]: c["nombre"] for c in _read("coberturas.json")},
        "canales": canales,
        "elegibilidad": elegibilidad,
        "postings": postings,
        "idf": idf,
        "doc_len": doc_len,
        "avgdl": avgdl,
    }


def _data():
    global _D
    if _D is None:
        _D = _build()
    return _D


# ------------------------------------------------------------------ búsqueda
def _bm25_top(d, query, pool):
    scores = {}
    for t in set(_tokens(query)):
        posting = d["postings"].get(t)
        if not posting:
            continue
        w = d["idf"][t]
        for i, tf in posting:
            norm = 1 - BM25_B + BM25_B * d["doc_len"][i] / d["avgdl"]
            scores[i] = scores.get(i, 0.0) + w * tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm)
    return sorted(scores, key=scores.get, reverse=True)[:pool]


def _fuente(d, chunk):
    p = d["producto_por_slug"].get(chunk.get("producto_slug"))
    if p:
        return p["nombre"]
    q = d["proceso_por_slug"].get(chunk.get("proceso_slug"))
    return q["nombre"] if q else None


def hybrid_search(query_vec, query, k=6, pool=20):
    """RRF de coseno + BM25. Misma forma de salida que la versión SQL."""
    d = _data()
    q = np.asarray(query_vec, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n:
        q = q / n
    sims = d["emb"] @ q
    vec_ids = np.argsort(-sims)[:pool].tolist()
    lex_ids = _bm25_top(d, query, pool)

    scores = {}
    for rank, i in enumerate(vec_ids):
        scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank)
    for rank, i in enumerate(lex_ids):
        scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank)

    out = []
    for i in sorted(scores, key=scores.get, reverse=True)[:k]:
        c = d["chunks"][i]
        out.append({"seccion": c.get("seccion"), "contenido": c["contenido"],
                    "url": c.get("url"), "fuente": _fuente(d, c),
                    "score": round(scores[i], 4)})
    return out


# ------------------------------------------------------------------ tools
def _sim(a, b):
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def detalle_producto(nombre):
    """Equivalente a tool_detalle_producto sin SQL (similarity -> SequenceMatcher)."""
    d = _data()
    if not d["productos"]:
        return f"No encontré un producto para '{nombre}'.", []
    p = max(d["productos"],
            key=lambda x: max(_sim(x["nombre"], nombre), _sim(x["slug"], nombre)))

    cobs = [(d["cob_nombre"].get(r["cobertura_slug"], r["cobertura_slug"]),
             r["tipo"], r.get("limite"))
            for r in d["plan_cob"] if r["producto_slug"] == p["slug"]]
    cobs.sort(key=lambda r: (r[1] or "", r[0] or ""))

    def grupo(t):
        return [n + (f" (límite: {lim})" if lim else "") for n, tt, lim in cobs if tt == t]

    cot = p.get("cotizador_url")
    parts = [
        f"PRODUCTO: {p['nombre']}",
        f"Descripción: {p.get('descripcion')}",
        f"Tipo de persona: {p.get('tipo_persona')} | "
        f"Digital: {'sí' if p.get('es_digital') else 'no'}"
        + (f" | Cotizador: {cot}" if cot else ""),
    ]
    inc, opc, exc = grupo("incluida"), grupo("opcional"), grupo("excluida")
    if inc:
        parts.append("CUBRE (incluidas):\n- " + "\n- ".join(inc))
    if opc:
        parts.append("OPCIONALES:\n- " + "\n- ".join(opc))
    if exc:
        parts.append("NO CUBRE (exclusiones):\n- " + "\n- ".join(exc))
    eleg = d["elegibilidad"].get(p["slug"])
    if eleg and eleg.get("descripcion"):
        parts.append(f"A QUIÉN APLICA: {eleg['descripcion']}")
    canales = d["canales"].get(p["slug"])
    if canales:
        parts.append("CANALES: " + ", ".join(canales))
    parts.append(f"URL: {p.get('url')}")
    return "\n\n".join(parts), [{"nombre": p["nombre"], "url": p.get("url")}]


def comparar_productos(termino):
    """Equivalente a tool_comparar_productos sin SQL (ILIKE -> substring)."""
    d = _data()
    t = _norm(termino)
    prods = [p for p in d["productos"]
             if t in _norm(p.get("nombre")) or t in _norm(p.get("ramo"))
             or t in _norm(p.get("descripcion"))]
    prods.sort(key=lambda p: p["nombre"])
    prods = prods[:5]
    if not prods:
        return f"No encontré productos para comparar sobre '{termino}'.", []

    byprod = {}
    for p in prods:
        for r in d["plan_cob"]:
            if r["producto_slug"] != p["slug"] or r["tipo"] not in ("incluida", "opcional"):
                continue
            cn = d["cob_nombre"].get(r["cobertura_slug"], r["cobertura_slug"])
            byprod.setdefault(p["nombre"], []).append(
                cn + (" (opc)" if r["tipo"] == "opcional" else ""))

    parts = [f"COMPARACIÓN sobre '{termino}':"]
    for pn, items in byprod.items():
        parts.append(f"\n### {pn}\nCubre: " + ", ".join(items[:25]))
    return "\n".join(parts), [{"nombre": p["nombre"], "url": p.get("url")} for p in prods]
