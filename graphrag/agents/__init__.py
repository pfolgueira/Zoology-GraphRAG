from typing import List, Dict, Any
from ..graph.neo4j_manager import Neo4jManager
from ..llm.ollama_client import OllamaClient
from .retriever_tools import RetrieverTools
from .retriever_router import RetrieverRouter
from .answer_critic import AnswerCritic
from ..llm.groq_client import GroqClient
from ..llm.gemini_client import GeminiClient

class AgenticRAG:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.client = GeminiClient()
        self.tools = RetrieverTools(neo4j_manager)
        self.router = RetrieverRouter(self.tools)
        self.critic = AnswerCritic()
        self.conversation_history = []
        self.add_text2cypher_examples()
        self.add_terminology_maps()

    def answer(self, question: str, max_iterations: int = 2) -> Dict[str, Any]:
        """
        Responde una pregunta usando el sistema agéntico.

        Args:
            question: Pregunta del usuario
            max_iterations: Número máximo de iteraciones de refinamiento

        Returns:
            Dict con la respuesta y metadatos
        """
        iterations = []
        current_question = question

        for iteration in range(max_iterations):
            # 1. Recuperar contexto
            retrieval_result = self.router.retrieve(
                current_question,
                self.conversation_history
            )

            # 2a. Greeting / out-of-scope / skills: return the fixed response immediately
            if "direct_response" in retrieval_result:
                answer = retrieval_result["direct_response"]
                iterations.append({
                    "iteration": iteration + 1,
                    "question": current_question,
                    "retrieval": retrieval_result,
                    "answer": answer,
                    "critique": {"is_complete": True, "is_faithful": True, "missing_info": []},
                })
                break

            context = retrieval_result["context"]

            # 2b. No context retrieved → don't hallucinate
            if not context:
                answer = "This information is not in the knowledge base."
                iterations.append({
                    "iteration": iteration + 1,
                    "question": current_question,
                    "retrieval": retrieval_result,
                    "answer": answer,
                    "critique": {"is_complete": False, "is_faithful": True, "missing_info": []},
                })
                break

            # 2c. Generate answer from retrieved context
            answer = self._generate_answer(current_question, context)

            # 3. Criticar respuesta
            critique = self.critic.critique(current_question, context, answer)

            iterations.append({
                "iteration": iteration + 1,
                "question": current_question,
                "tools": retrieval_result.get("tools", [retrieval_result.get("tool")]),
                "retrieval": retrieval_result,
                "answer": answer,
                "critique": critique
            })

            # Si la respuesta es completa y fiel, terminar
            if critique["is_complete"] and critique["is_faithful"]:
                break

            # Si hay información faltante y no es la última iteración, refinar
            if critique["missing_info"] and iteration < max_iterations - 1:
                current_question = " ".join([
                    current_question,
                    "Additional questions:",
                    " ".join(critique["missing_info"])
                ])

        # Actualizar historial de conversación
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": iterations[-1]["answer"]
        })

        return {
            "question": question,
            "answer": iterations[-1]["answer"],
            "iterations": iterations,
            "final_critique": iterations[-1]["critique"]
        }

    def _generate_answer(self, question: str, context: List[str]) -> str:
        """Genera una respuesta usando el LLM."""
        context_str = "\n\n".join([f"[{i + 1}] {c}" for i, c in enumerate(context)])

        system_prompt = """You are an expert question-answering assistant specializing in zoology and animal biology. 
Your task is to answer the user's question using ONLY the provided context.

STRICT RULES — follow all of them:
1. NO PRIOR KNOWLEDGE: Answer exclusively from the provided context. Do not use outside information.
2. DIRECT START: Begin your reply with the answer itself. Never use filler phrases like "I", "Let me", "Based on the context", or "The text states".
3. NO EXPLANATIONS: Do not explain your reasoning or thinking process. Only state facts.
4. CONCISENESS: Keep the answer as concise as possible, ideally not exceeding one paragraph.
5. CITATIONS: Add an inline citation immediately after each distinct fact using brackets. Example: "The blue whale's heart weighs about 400 pounds [1]."
6. FALLBACK: If the provided context does not contain the answer to the question, reply EXACTLY with: "This information is not in the knowledge base."
"""

        user_message = f"Context:\n{context_str}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        return self.client.chat(messages)

    def reset_conversation(self):
        """Reinicia el historial de conversación."""
        self.conversation_history = []

    def add_text2cypher_examples(self):
        """Agrega ejemplos al prompt de text2cypher."""
        text2cypher_examples = [
            ("How many Mammal species are there?", "MATCH (s:Species)-[:BELONGS_TO_CLASS]->(c:AnimalClass {type: 'Mammal'}) RETURN count(s) AS totalMammals"),
            ("What is the top speed and maximum weight of a Lion?", "MATCH (s:Species {name: 'Lion'}) RETURN s.top_speed_kmh, s.weight_max_kg"),
            ("What type of diet do tigers have and what do they prey on?", "MATCH (s:Species {name: 'Tiger'})-[:HAS_DIET_TYPE]->(d:DietType), (s)-[:PREYS_ON]->(prey:Species) RETURN d.type, prey.name"),
            ("Which animals have a lifespan greater than 50 years?", "MATCH (s:Species) WHERE s.lifespan_years > 50 RETURN s.name, s.lifespan_years"),
            ("Where do whales migrate to in the winter?", "MATCH (s:Species {name: 'Whale'})-[m:MIGRATES_TO {season: 'Winter'}]->(l:Location) RETURN l.type"),
            ("What does a chimpanzee eat?", "MATCH (s:Species {name: 'Chimpanzee'}) OPTIONAL MATCH (s)-[:PREYS_ON]->(prey:Species) OPTIONAL MATCH (s)-[:FEEDS_ON]->(food:FoodSource) RETURN s.name AS species, collect(DISTINCT prey.name) AS preys_on, collect(DISTINCT food.type) AS feeds_on"),
            ("In which countries or regions are kangaroos found?", "MATCH (s:Species {name: 'Kangaroo'})-[:FOUND_IN]->(l:Location) RETURN l.type"),
            ("What kind of ecosystem or biome does the polar bear inhabit?", "MATCH (s:Species {name: 'Polar Bear'})-[:INHABITS]->(h:Habitat) RETURN h.type")
        ]
        for question, cypher in text2cypher_examples:
            self.router.tools.text2cypher.add_few_shot_example(question, cypher)

    def add_terminology_maps(self, term: str, description: str):
        """Agrega términos al mapa de terminología."""
        terminology_maps = [
            ("animal, creature, species", "Refers to the node with label (:Species)"),
            ("what does X eat, what do X eat, diet of X, food of X. When asking what an animal eats, check BOTH relationships: ", "[:PREYS_ON]->(:Species) for animal prey AND [:FEEDS_ON]->(:FoodSource) for non-animal food sources. Use OPTIONAL MATCH for both and return all results."),
            ("country, continent, geographic region, area, located in", "Use the relationship [:FOUND_IN]->(:Location) to refer to specific geographical and political places."),
            ("biome, ecosystem, type of environment, terrain, inhabits", "Use the relationship [:INHABITS]->(:Habitat) to refer to the natural biome or habitat type.")    
        ]
        for terms, explanation in terminology_maps:
            self.router.tools.text2cypher.add_terminology_map(term, explanation)