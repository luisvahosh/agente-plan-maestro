#!/usr/bin/env python3
"""
Genera variantes parafraseadas de las preguntas del golden set, para evaluar la
GENERALIZACIÓN del agente (no solo si recupera la pregunta exacta inyectada).

Por cada pregunta verificada crea N reformulaciones con distinto registro
(coloquial, técnica, desde un actor específico) manteniendo la misma intención.
Cachea el resultado en entrenamiento/preguntas_variantes.json para reproducibilidad.

Uso:  python scripts/generar_variantes.py [--n 3]
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

from openai import OpenAI
# Reutilizamos los cargadores del golden set
from scripts.evaluar_agente import load_qa_docx, load_casos_pdf  # type: ignore

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
GEN_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")

ENTR = ROOT / "entrenamiento"
OUT_JSON = ENTR / "preguntas_variantes.json"

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

GEN_SYSTEM = (
    "Eres un generador de variantes de preguntas sobre el Plan Maestro Medellín "
    "Distrito Inteligente. Dada una pregunta, genera reformulaciones que mantengan "
    "EXACTAMENTE la misma intención e información buscada, pero con palabras distintas. "
    "Varía el registro entre las variantes: una coloquial/informal, una más técnica, "
    "y una desde la perspectiva de un actor concreto (ciudadano, empresa, academia o "
    "funcionario). No respondas la pregunta; solo reformúlala. Devuelve SOLO las "
    "preguntas, una por línea, sin numeración ni viñetas."
)


def generar(pregunta: str, n: int) -> list[str]:
    msg = f"Pregunta original:\n{pregunta}\n\nGenera {n} variantes (una por línea)."
    r = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": GEN_SYSTEM},
                  {"role": "user", "content": msg}],
        temperature=0.8, max_tokens=300,
    )
    lines = [l.strip() for l in r.choices[0].message.content.splitlines() if l.strip()]
    # Limpiar numeración/viñetas residuales
    out = []
    for l in lines:
        l = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", l).strip()
        if l and l != pregunta:
            out.append(l)
    return out[:n]


def main():
    n = 3
    if "--n" in sys.argv:
        try:
            n = int(sys.argv[sys.argv.index("--n") + 1])
        except Exception:
            pass

    golden = load_qa_docx() + load_casos_pdf()
    print(f"🧬 Generando {n} variantes para {len(golden)} preguntas (modelo {GEN_MODEL})…\n")

    data = []
    for i, item in enumerate(golden, 1):
        variantes = generar(item["pregunta"], n)
        data.append({
            "original": item["pregunta"],
            "referencia": item["referencia"],
            "tipo": item["tipo"],
            "variantes": variantes,
        })
        print(f"  [{i}/{len(golden)}] {len(variantes)} variantes  ·  {item['pregunta'][:55]}")
        for v in variantes:
            print(f"        – {v[:80]}")

    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {sum(len(d['variantes']) for d in data)} variantes guardadas en {OUT_JSON}")


if __name__ == "__main__":
    main()
