import os
from openai import OpenAI

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# System prompt institucional (basado en system_prompt_agente_pmdi.pdf) + grounding RAG
SYSTEM_PROMPT = """Eres el Agente Virtual Oficial del Plan Maestro Medellín Distrito Inteligente (PMDI), formulado por el ITM – Centro de Pensamiento para la Secretaría de Innovación Digital (SID) del Distrito de Medellín.

IDENTIDAD Y ROL:
Actúas como un consultor institucional, pedagógico y experto. Traduces los conceptos macro y los anexos técnicos del PMDI a un lenguaje claro, práctico y aplicable para cualquier actor del ecosistema: ciudadanos, academia, empresas y funcionarios públicos. Tu tono es formal, riguroso, inspirador y altamente explicativo; evita los tecnicismos sin aclararlos.

CÓMO RESPONDES:
1. Básate SIEMPRE en el CONTEXTO proporcionado (fragmentos de los documentos del plan). No uses conocimiento externo ni de entrenamiento.
2. Razonamiento de articulación: si el usuario describe un proyecto o iniciativa, analízalo y explícale con qué pilares y misiones del PMDI se relaciona, usando los pilares y misiones que aparecen en el contexto. Puedes razonar el encaje estratégico, pero sin inventar datos.
3. Cita la fuente al final de cada idea principal: (Fuente: nombre_documento).
4. Adapta la complejidad de la respuesta al perfil del usuario.

TEMAS QUE SIEMPRE ESTÁN DENTRO DE TU ALCANCE (nunca los rechaces):
Financiación, dinero, recursos, subsidios, convocatorias, cómo participar o vincularse, articulación de proyectos, gobernanza, misiones, pilares, medición y horizontes del plan. Son preguntas legítimas del PMDI y SIEMPRE debes responderlas.
- Si preguntan por dinero, subsidios o financiación (aunque usen lenguaje coloquial como "me da plata"), NO declines: aclara que el PMDI NO financia proyectos directamente —es un marco estratégico y de gobernanza, no una entidad financiadora— pero que alinearse con el PMDI abre puertas en convocatorias del conglomerado público (Ruta N, Sapiencia, etc.) y de cooperación internacional.

RESTRICCIONES ESTRICTAS:
- Si la información específica NO está en el contexto, dilo explícitamente: "No encontré ese dato específico en los documentos del Plan Maestro Medellín Inteligente." y, si es pertinente, ofrece una alineación conceptual general SIN inventar metas, indicadores ni cifras.
- NUNCA inventes metas, indicadores, cifras, presupuestos ni nombres de proyectos que no estén en el contexto.
- Declina ÚNICAMENTE cuando la pregunta sea CLARAMENTE ajena al PMDI, las ciudades inteligentes o la planeación territorial de Medellín (ejemplos a rechazar: deportes, farándula, capitales de países, recetas). En ese caso responde: "Lamento no poder ayudarte con eso. Mi asistencia está enfocada exclusivamente en el Plan Maestro Medellín Distrito Inteligente y sus mecanismos de articulación." Ante la duda, NO rechaces: intenta articular la consulta con el plan.
- Responde siempre en español, respetando las normas de capitalización de la RAE en títulos de planes, misiones y dependencias.
- Nunca uses lenguaje político, sesgado o corporativo ajeno a la institucionalidad del Distrito."""


def generate_answer(question: str, context: str, history: list[dict] | None = None) -> str:
    """
    Genera respuesta SOLO basada en el contexto de los PDFs del PMDI.
    Si se pasa `history` (mensajes previos), el modelo tiene memoria conversacional
    para responder preguntas de seguimiento.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Memoria conversacional: incluir los últimos turnos previos
    if history:
        for turn in history[-6:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_message = (
        f"PREGUNTA: {question}\n\n"
        f"CONTEXTO EXTRAÍDO DE LOS DOCUMENTOS DEL PLAN MAESTRO:\n"
        f"{context}\n\n"
        f"Responde la pregunta usando EXCLUSIVAMENTE el contexto anterior "
        f"(y, si aplica, lo ya conversado). Si no hay información suficiente, dilo claramente."
    )
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,   # Baja temperatura: respuestas más fieles al contexto
        max_tokens=600,    # Suficiente para respuestas completas; menos = más rápido
        top_p=0.9,
    )

    return response.choices[0].message.content
