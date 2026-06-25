#!/usr/bin/env python3
"""
Evaluación de GENERALIZACIÓN: corre las variantes parafraseadas (entrenamiento/
preguntas_variantes.json) contra el agente y las califica contra la respuesta de
referencia original. Mide si el agente responde bien aunque la pregunta esté
formulada con otras palabras (no solo cuando coincide con la Q&A inyectada).

Requiere haber corrido antes: python scripts/generar_variantes.py
Uso:  python scripts/evaluar_variantes.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

from app.rag import query
from scripts.evaluar_agente import juzgar, JUDGE_MODEL  # type: ignore

ENTR = ROOT / "entrenamiento"
VARIANTES_JSON = ENTR / "preguntas_variantes.json"
REPORTE = ENTR / "reporte_variantes.md"

ICON = {"CORRECTO": "✅", "PARCIAL": "🟡", "INCORRECTO": "❌", "ERROR": "⚠️"}


async def main():
    if not VARIANTES_JSON.exists():
        print(f"❌ No existe {VARIANTES_JSON}. Corre primero: python scripts/generar_variantes.py")
        sys.exit(1)

    data = json.load(open(VARIANTES_JSON, encoding="utf-8"))
    total_variantes = sum(len(d["variantes"]) for d in data)
    print(f"🧪 Evaluando {total_variantes} variantes de {len(data)} preguntas (juez {JUDGE_MODEL})\n")

    cont = {"CORRECTO": 0, "PARCIAL": 0, "INCORRECTO": 0, "ERROR": 0}
    detalle = []        # por variante
    por_original = []   # agregado por pregunta original
    idx = 0

    for item in data:
        ref = item["referencia"]
        veredictos = []
        for v in item["variantes"]:
            idx += 1
            t = time.time()
            res = await query(v)
            dt = time.time() - t
            ver, just = juzgar(v, ref, res.get("answer", ""))
            cont[ver] = cont.get(ver, 0) + 1
            veredictos.append(ver)
            detalle.append({
                "original": item["original"], "variante": v, "veredicto": ver,
                "just": just, "sim": res.get("best_similarity", 0), "t": dt,
                "respuesta": res.get("answer", ""),
            })
            print(f"  {ICON.get(ver,'?')} [{idx}/{total_variantes}] {ver:10} ({dt:.1f}s)  {v[:55]}")

        # Agregado: la original "generaliza bien" si TODAS sus variantes son CORRECTO
        ok = all(x == "CORRECTO" for x in veredictos)
        parcial_ok = all(x in ("CORRECTO", "PARCIAL") for x in veredictos)
        por_original.append({
            "original": item["original"], "veredictos": veredictos,
            "robusta": ok, "aceptable": parcial_ok,
        })

    n = total_variantes
    correctos = cont["CORRECTO"]
    parciales = cont["PARCIAL"]
    incorrectos = cont["INCORRECTO"]
    robustas = sum(1 for p in por_original if p["robusta"])

    print("\n" + "=" * 58)
    print("📊 EVALUACIÓN DE GENERALIZACIÓN (variantes parafraseadas)")
    print(f"  Variantes evaluadas: {n}")
    print(f"    ✅ Correcto:   {correctos}  ({100*correctos/n:.0f}%)")
    print(f"    🟡 Parcial:    {parciales}  ({100*parciales/n:.0f}%)")
    print(f"    ❌ Incorrecto: {incorrectos}  ({100*incorrectos/n:.0f}%)")
    print(f"  Preguntas robustas (todas sus variantes correctas): {robustas}/{len(por_original)}")
    print("=" * 58)

    # ── Reporte markdown ──
    L = ["# Reporte de generalización (variantes parafraseadas)", ""]
    L.append(f"- Variantes evaluadas: **{n}** (de {len(por_original)} preguntas)")
    L.append(f"- ✅ Correcto: **{correctos}** ({100*correctos/n:.0f}%)  |  🟡 Parcial: **{parciales}** ({100*parciales/n:.0f}%)  |  ❌ Incorrecto: **{incorrectos}** ({100*incorrectos/n:.0f}%)")
    L.append(f"- Preguntas robustas (todas sus variantes ✅): **{robustas}/{len(por_original)}**")
    L.append(f"- Juez: `{JUDGE_MODEL}`")
    L.append("")
    L.append("## Preguntas NO robustas (alguna variante falló — revisar)")
    L.append("")
    hay = False
    for p in por_original:
        if not p["robusta"]:
            hay = True
            L.append(f"- **{p['original'][:80]}** → {p['veredictos']}")
    if not hay:
        L.append("_(ninguna: todas las preguntas respondieron bien en todas sus variantes)_")
    L.append("")
    L.append("## Variantes marcadas PARCIAL / INCORRECTO")
    L.append("")
    for d in detalle:
        if d["veredicto"] in ("PARCIAL", "INCORRECTO", "ERROR"):
            L.append(f"### [{d['veredicto']}] {d['variante']}")
            L.append(f"_Original:_ {d['original'][:80]}")
            L.append(f"**Juez:** {d['just']}")
            L.append(f"**Respuesta:** {d['respuesta'][:500]}")
            L.append("")

    REPORTE.write_text("\n".join(L), encoding="utf-8")
    print(f"\n📄 Reporte guardado en: {REPORTE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  cancelado")
