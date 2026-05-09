from typing import List, Dict, Any, Tuple
from ..graph.neo4j_manager import Neo4jManager
from ..llm.ollama_client import OllamaClient
from ..llm.groq_client import GroqClient


class Text2CypherRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.client = GroqClient()
        self.few_shot_examples = []
        self.terminology_maps = {} # Nuevo: Diccionario para almacenar mapas terminológicos

    def add_few_shot_example(self, question: str, cypher: str):
        """Añade un ejemplo few-shot."""
        self.few_shot_examples.append({
            "question": question,
            "cypher": cypher
        })

    def add_terminology_map(self, term: str, graph_equivalent: str):
        """
        Añade una regla al mapa terminológico.
        """
        self.terminology_maps[term] = graph_equivalent

    def generate_cypher(self, question: str) -> str:
        """
        Genera una query Cypher a partir de una pregunta en lenguaje natural.
        """
        schema = self.neo4j.get_schema()
        schema_str = self.neo4j.format_schema(schema)

        # 1. Construir mapas terminológicos
        terminology_str = "No specific terminology mapped."
        if self.terminology_maps:
            terminology_str = "\n".join([
                f"- '{term}' means/refers to: {mapping}"
                for term, mapping in self.terminology_maps.items()
            ])

        # 2. Construir ejemplos few-shot
        examples_str = "No examples provided."
        if self.few_shot_examples:
            examples_str = "\n".join([
                f"Question: {ex['question']}\nCypher: {ex['cypher']}\n"
                for ex in self.few_shot_examples
            ])

        enums_str = """When filtering by the 'type' property (or 'name' for Species) on the following nodes, you should use one of these Title Case values. Do not use lowercase or alter the spelling:

- (:AnimalClass {type: ...}): 'Mammal', 'Reptile', 'Bird', 'Fish', 'Amphibian'
- (:SkeletalStructure {type: ...}): 'Vertebrate', 'Invertebrate'
- (:ReproductionMethod {type: ...}): 'Oviparous', 'Viviparous', 'Ovoviviparous'
- (:EnvironmentType {type: ...}): 'Aquatic', 'Terrestrial', 'Aerial'
- (:ActivityCycle {type: ...}): 'Nocturnal', 'Diurnal', 'Crepuscular'
- (:SocialStructure {type: ...}): 'Solitary', 'Pair-living', 'Family group', 'Herd', 'Pack', 'Colony', 'Eusocial'
- (:DietType {type: ...}): 'Carnivore', 'Herbivore', 'Omnivore'
- (:FoodSource {type: ...}): 'Grass', 'Leaves', 'Fruits', 'Seeds', 'Bark', 'Nectar', 'Aquatic Plants'
- (:ConservationStatus {type: ...}): 'EX (Extinct)', 'EW (Extinct in the Wild)', 'CR (Critically Endangered)', 'EN (Endangered)', 'VU (Vulnerable)', 'NT (Near Threatened)', 'LC (Least Concern)', 'DD (Data Deficient)', 'NE (Not Evaluated)'"""

        system_prompt = f"""You are an Information Retrieval Agent operating a Neo4j Knowledge Graph. Your task is to translate natural language questions into exact Cypher queries to extract the required data.

--- GRAPH SCHEMA ---
{schema_str}

--- TERMINOLOGY MAP ---
Use the following domain-specific terminology mappings to understand the user's intent:
{terminology_str}

--- ALLOWED ENUM VALUES (STRICT EXACT MATCH) ---
{enums_str}

--- FEW-SHOT EXAMPLES ---
Use these examples as a reference for the expected Cypher structure:
{examples_str}

--- FORMATTING INSTRUCTIONS ---
1. Return ONLY the raw Cypher query.
2. Do NOT wrap the query in markdown code blocks (e.g., no ```cypher or ```).
3. Do NOT provide any explanations, apologies, or conversational text before or after the query.
4. Ensure the query is syntactically correct and optimized for Neo4j.
5. Use EXACTLY the node labels, relationship types, and properties provided in the schema and terminology map. Do not hallucinate properties.
6. GENERAL TITLE CASE RULE: For ALL OTHER nodes and string properties not listed in the enums (such as Species 'name', Location 'type', etc.), you MUST format the search values in Title Case (e.g., use 'African Elephant' instead of 'african elephant', or 'South Africa' instead of 'south africa').
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate a Cypher query to answer the following question:" + question}
        ]

        # Mantenemos temperature a 0.0 para que sea determinista y preciso
        cypher = self.client.chat(messages, temperature=0.0)

        # Limpiar la respuesta (doble comprobación por si el LLM ignora las instrucciones de formato)
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()

        return cypher

    def retrieve(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Genera una query Cypher y la ejecuta.

        Returns:
            Tupla con (cypher_query, results)
        """
        cypher = self.generate_cypher(question)

        try:
            results = self.neo4j.execute_query(cypher)
            return cypher, results
        except Exception as e:
            print(f"Error ejecutando Cypher: {e}")
            print(f"Query generada: {cypher}")
            return cypher, []