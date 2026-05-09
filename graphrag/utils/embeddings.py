from typing import List
from ..llm.ollama_client import OllamaClient
from ..llm.groq_client import GroqClient


class EmbeddingGenerator:
    def __init__(self):
        self.client = GroqClient()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para una lista de textos."""
        return self.client.embed(texts)

    def embed_text(self, text: str) -> List[float]:
        """Genera embedding para un solo texto."""
        return self.client.embed([text])[0]