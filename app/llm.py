import os
from openai import OpenAI

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# Prompt estricto: el modelo SOLO puede usar el contexto dado, nunca conocimiento externo
SYSTEM_PROMPT = """Eres un asistente especializado en el Plan Maestro Medellín Inteligente (PMDI).

REGLAS OBLIGATORIAS — debes seguirlas sin excepción:

1. ÚNICAMENTE puedes responder usando la información del CONTEXTO proporcionado.
2. Si la información NO está en el contexto, debes responder exactamente:
   "No encontré información sobre esto en los documentos del Plan Maestro Medellín Inteligente. Por favor, formula una pregunta relacionada con el contenido del plan."
3. NUNCA uses conocimiento propio, externo o de entrenamiento para complementar o suponer información.
4. NUNCA inventes datos, cifras, nombres de proyectos o estrategias que no estén en el contexto.
5. Cuando respondas, cita siempre la fuente entre paréntesis al final de cada párrafo: (Fuente: nombre_documento).
6. Responde siempre en español.
7. Si el contexto es parcial o insuficiente, dilo explícitamente antes de responder.
8. Sé conciso y directo."""


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
