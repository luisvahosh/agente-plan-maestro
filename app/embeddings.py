import os
from openai import OpenAI

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


def embed_query(text: str) -> list[float]:
    """
    Genera el embedding de una PREGUNTA del usuario.
    NVIDIA exige input_type='query' para optimizar la busqueda.
    """
    response = client.embeddings.create(
        input=[text],
        model=EMBED_MODEL,
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return response.data[0].embedding


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Genera embeddings de FRAGMENTOS de documentos.
    NVIDIA exige input_type='passage' para contenido indexado.
    """
    response = client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
        extra_body={"input_type": "passage", "truncate": "END"},
    )
    return [item.embedding for item in response.data]
