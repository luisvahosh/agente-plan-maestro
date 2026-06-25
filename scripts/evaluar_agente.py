#!/usr/bin/env python3
"""
Batería de evaluación automática del agente PMDI.

Corre un conjunto "golden" de preguntas con respuesta verificada contra el pipeline
RAG real del agente, y califica cada respuesta con un modelo LLM-juez. Genera un
reporte en entrenamiento/reporte_evaluacion.md y un resumen en consola.

Golden set:
  - 12 Q&A verificadas (banco Word)
  - 10 casos de articulación (PDF)
  - Controles fuera de tema (deben ser rechazados por el agente)

Cómo se mide:
  - Para preguntas de contenido: LLM-juez compara la respuesta del agente con la
    respuesta de referencia → CORRECTO / PARCIAL / INCORRECTO.
  - Para controles: se verifica que el agente declina (no inventa).

Uso:  python scripts/evaluar_agente.py
"""

import asyncio
import os
import re
import sys
import time
from pathlib import Path

# Permitir importar el paquete app/ al ejecutar desde scripts/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

import fitz  # PyMuPDF
import docx
from openai import OpenAI

from app.rag import query  # pipeline RAG real

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Modelo juez: uno capaz, porque calificar exige criterio (es offline, la latencia no importa)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "meta/llama-3.3-70b-instruct")

ENTR = ROOT / "entrenamiento"
DOCX_QA = ENTR / "PMDI_Banco_QA_Agente_Conversacional.docx"
PDF_CASOS = ENTR / "preguntas_entrenamiento_pmdi.pdf"
REPORTE = ENTR / "reporte_evaluacion.md"

judge_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# Controles: preguntas fuera de alcance que el agente DEBE rechazar
CONTROLES = [
    "¿Cuál es la capital de Francia?",
    "Dame una receta de arepas",
    "¿Quién ganó el último partido de Nacional?",
    "¿Cómo solicito un subsidio de vivienda?",
]
DECLINE_MARKERS = [
    "lamento no poder", "enfocada exclusivamente", "no puedo ayudarte",
    "no encontré información sobre esto", "formula una pregunta relacionada",
    "dirígete al", "redirige", "no son competencia", "no es competencia",
]


# ── Cargar golden set (pregunta + respuesta de referencia) ────────────────────

def load_qa_docx() -> list[dict]:
    doc = docx.Document(DOCX_QA)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    end = next((i for i, t in enumerate(paras)
                if t.lower().startswith("recomendaciones para entrenar")), len(paras))
    starts = [i for i in range(end) if re.match(r"^Pregunta\s+\d+$", paras[i])]
    starts.append(end)

    items = []
    for b in range(len(starts) - 1):
        block = paras[starts[b]: starts[b + 1]]
        try:
            r_idx = next(i for i, t in enumerate(block) if t.lower() == "respuesta")
        except StopIteration:
            continue
        cat_idx = next((i for i, t in enumerate(block) if t.lower().startswith("categoría")), 0)
        pregunta = " ".join(block[cat_idx + 1: r_idx]).strip()
        respuesta = "\n".join(t for t in block[r_idx + 1:] if not t.startswith("💡")).strip()
        if pregunta and respuesta:
            items.append({"tipo": "Q&A", "pregunta": pregunta, "referencia": respuesta})
    return items


def load_casos_pdf() -> list[dict]:
    d = fitz.open(PDF_CASOS)
    text = "".join(p.get_text() for p in d)
    d.close()
    m = re.search(r"(?m)^\s*1\.\s", text)
    if m:
        text = text[m.start():]
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    pieces = re.split(r"(?m)^(?=\d{1,2}\.\s)", text)

    items = []
    for piece in pieces:
        piece = piece.strip()
        if not re.match(r"^\d{1,2}\.\s", piece):
            continue
        body = re.sub(r"^\d{1,2}\.\s*", "", piece).strip()
        # Separar pregunta (hasta el primer '?') de la respuesta
        mm = re.search(r"\?\s", body)
        if mm:
            pregunta = body[: mm.end()].strip().replace("\n", " ")
            referencia = body[mm.end():].strip()
        else:
            pregunta, referencia = body[:120], body
        items.append({"tipo": "Caso", "pregunta": pregunta, "referencia": referencia})
    return items


# ── Juez LLM ──────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "Eres un evaluador experto y estricto del Plan Maestro Medellín Distrito Inteligente. "
    "Compara la RESPUESTA DEL AGENTE con la RESPUESTA DE REFERENCIA (verificada) para una PREGUNTA. "
    "Califica qué tan bien el agente responde según la referencia.\n"
    "Primera línea: EXACTAMENTE una palabra → CORRECTO, PARCIAL o INCORRECTO.\n"
    "Segunda línea: justificación breve (máx 20 palabras).\n"
    "CORRECTO = cubre los hechos clave de la referencia, sin contradicciones ni invenciones.\n"
    "PARCIAL = correcto pero incompleto, o le faltan hechos clave.\n"
    "INCORRECTO = contradice la referencia, inventa datos, o no responde."
)


def juzgar(pregunta: str, referencia: str, respuesta: str) -> tuple[str, str]:
    msg = (
        f"PREGUNTA:\n{pregunta}\n\n"
        f"RESPUESTA DE REFERENCIA (verificada):\n{referencia}\n\n"
        f"RESPUESTA DEL AGENTE:\n{respuesta}\n\n"
        f"Califica."
    )
    try:
        r = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": msg}],
            temperature=0.0, max_tokens=120,
        )
        out = r.choices[0].message.content.strip()
        first = out.splitlines()[0].upper()
        if "INCORRECTO" in first:
            veredicto = "INCORRECTO"
        elif "PARCIAL" in first:
            veredicto = "PARCIAL"
        elif "CORRECTO" in first:
            veredicto = "CORRECTO"
        else:
            veredicto = "PARCIAL"
        just = " ".join(out.splitlines()[1:]).strip() or "—"
        return veredicto, just
    except Exception as e:
        return "ERROR", str(e)[:80]


# ── Evaluación ────────────────────────────────────────────────────────────────

async def main():
    print("🧪 Cargando golden set…")
    golden = load_qa_docx() + load_casos_pdf()
    print(f"  {len(golden)} preguntas de contenido + {len(CONTROLES)} controles\n")

    filas = []
    contadores = {"CORRECTO": 0, "PARCIAL": 0, "INCORRECTO": 0, "ERROR": 0}

    # 1) Preguntas de contenido
    for i, item in enumerate(golden, 1):
        t = time.time()
        res = await query(item["pregunta"])
        dt = time.time() - t
        respuesta = res.get("answer", "")
        veredicto, just = juzgar(item["pregunta"], item["referencia"], respuesta)
        contadores[veredicto] = contadores.get(veredicto, 0) + 1
        filas.append({
            "tipo": item["tipo"], "pregunta": item["pregunta"], "veredicto": veredicto,
            "just": just, "fuentes": res.get("sources", []),
            "sim": res.get("best_similarity", 0), "t": dt, "respuesta": respuesta,
        })
        icon = {"CORRECTO": "✅", "PARCIAL": "🟡", "INCORRECTO": "❌", "ERROR": "⚠️"}.get(veredicto, "?")
        print(f"  {icon} [{i}/{len(golden)}] {veredicto:10} ({dt:.1f}s)  {item['pregunta'][:60]}")

    # 2) Controles (deben declinar)
    ctrl_ok = 0
    ctrl_filas = []
    for q in CONTROLES:
        res = await query(q)
        ans = (res.get("answer", "") or "").lower()
        declino = any(m in ans for m in DECLINE_MARKERS)
        ctrl_ok += declino
        ctrl_filas.append({"pregunta": q, "ok": declino, "respuesta": res.get("answer", "")})
        print(f"  {'✅' if declino else '❌'} CONTROL {'rechaza' if declino else 'NO rechaza'}: {q[:50]}")

    # ── Resumen ──
    n = len(golden)
    correctos = contadores["CORRECTO"]
    parciales = contadores["PARCIAL"]
    incorrectos = contadores["INCORRECTO"]
    aciertos_pct = 100 * correctos / n if n else 0
    aprob_pct = 100 * (correctos + parciales) / n if n else 0

    print("\n" + "=" * 55)
    print("📊 RESUMEN DE EVALUACIÓN")
    print(f"  Contenido: {n} preguntas")
    print(f"    ✅ Correcto:   {correctos}  ({aciertos_pct:.0f}%)")
    print(f"    🟡 Parcial:    {parciales}")
    print(f"    ❌ Incorrecto: {incorrectos}")
    print(f"    Aprobación (correcto+parcial): {aprob_pct:.0f}%")
    print(f"  Controles (deben rechazar): {ctrl_ok}/{len(CONTROLES)} OK")
    print("=" * 55)

    # ── Reporte markdown ──
    lineas = ["# Reporte de evaluación del agente PMDI", ""]
    lineas.append(f"- Preguntas de contenido: **{n}**")
    lineas.append(f"- ✅ Correcto: **{correctos}** ({aciertos_pct:.0f}%)  |  🟡 Parcial: **{parciales}**  |  ❌ Incorrecto: **{incorrectos}**")
    lineas.append(f"- Aprobación (correcto+parcial): **{aprob_pct:.0f}%**")
    lineas.append(f"- Controles fuera de tema rechazados: **{ctrl_ok}/{len(CONTROLES)}**")
    lineas.append(f"- Modelo juez: `{JUDGE_MODEL}`")
    lineas.append("")
    lineas.append("## Detalle por pregunta")
    lineas.append("")
    lineas.append("| # | Tipo | Veredicto | Pregunta | Justificación juez | Sim | seg |")
    lineas.append("|---|------|-----------|----------|--------------------|-----|-----|")
    for i, f in enumerate(filas, 1):
        preg = f["pregunta"].replace("|", "/")[:70]
        just = f["just"].replace("|", "/")[:60]
        lineas.append(f"| {i} | {f['tipo']} | {f['veredicto']} | {preg} | {just} | {f['sim']:.2f} | {f['t']:.1f} |")
    lineas.append("")
    lineas.append("## Controles (deben ser rechazados)")
    lineas.append("")
    for c in ctrl_filas:
        lineas.append(f"- {'✅' if c['ok'] else '❌'} {c['pregunta']}")
    lineas.append("")
    lineas.append("## Respuestas marcadas PARCIAL o INCORRECTO (para revisar)")
    lineas.append("")
    for i, f in enumerate(filas, 1):
        if f["veredicto"] in ("PARCIAL", "INCORRECTO", "ERROR"):
            lineas.append(f"### {i}. [{f['veredicto']}] {f['pregunta']}")
            lineas.append(f"**Juez:** {f['just']}")
            lineas.append(f"**Respuesta del agente:** {f['respuesta'][:600]}")
            lineas.append("")

    REPORTE.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\n📄 Reporte guardado en: {REPORTE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  cancelado")
