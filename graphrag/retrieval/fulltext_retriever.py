from typing import List, Dict, Any

from graphrag.config import get_settings
from graphrag.graph.neo4j_manager import Neo4jManager
from graphrag.llm.ollama_client import OllamaClient
from graphrag.llm.groq_client import GroqClient


class FullTextRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.settings = get_settings()
        self._create_fulltext_index()
        self.client = GroqClient()

    def _create_fulltext_index(self):
        """Crea un índice de texto completo."""
        query = """
        CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
        FOR (c:Chunk)
        ON EACH [c.text]
        """
        try:
            self.neo4j.execute_query(query)
        except Exception as e:
            print(f"Índice fulltext ya existe o error: {e}")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recupera chunks usando búsqueda vectorial pura de texto completo.
        """
        top_k = top_k or self.settings.top_k_candidates

        # Construir el prompt para el LLM
        system_prompt = """You are a precise keyword extraction assistant for an Animal Knowledge Graph. 
Your ONLY task is to extract the core search terms from a natural language query to be used in a Full-Text search.

RULES:
1. Remove all stop words (e.g., 'what', 'are', 'the', 'of', 'in', 'is', 'how', 'tell', 'me', 'about', etc.).
2. Keep essential nouns, adjectives, and proper names (e.g., species names, habitats, diets, geographic locations).
3. Return ONLY a single line of space-separated keywords.
4. Do NOT add any conversational text, explanations, or quotes.

EXAMPLES:
User: "What are the natural predators of the African Elephant?"
Output: natural predators African Elephant

User: "Tell me about the habitat and diet of carnivorous reptiles."
Output: habitat diet carnivorous reptiles

User: "Why do ducks turn their heads backward?"
Output: ducks heads backward"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Mantenemos temperature a 0.0 para que sea determinista y preciso
        keywords = self.client.chat(messages, temperature=0.0)
        
        # Limpieza de seguridad post-generación
        keywords = keywords.strip().replace('"', '').replace("'", "")

        cypher_query = """
        // Fulltext search
        CALL db.index.fulltext.queryNodes('chunk_fulltext', $keywords, {limit: $top_k})
        YIELD node, score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN node.text AS text,
               node.id AS chunk_id,
               score
        """

        results = self.neo4j.execute_query(cypher_query, {
            "keywords": keywords,
            "top_k": top_k
        })

        return results