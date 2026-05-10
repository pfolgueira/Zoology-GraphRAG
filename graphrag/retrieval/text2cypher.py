from typing import List, Dict, Any, Tuple
from rapidfuzz import process, fuzz
from ..graph.neo4j_manager import Neo4jManager
from ..llm.groq_client import GroqClient
from ..llm.gemini_client import GeminiClient


class Text2CypherRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.client = GeminiClient()
        self.few_shot_examples = []
        self.terminology_maps = {}
        self._known_species: List[str] = self._load_known_species()

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

    def _load_known_species(self) -> List[str]:
        """Carga y cachea todos los nombres de especies de la BD."""
        try:
            result = self.neo4j.execute_query("MATCH (s:Species) RETURN s.name AS name")
            return [r["name"] for r in result]
        except Exception as e:
            print(f"Warning: could not load species list: {e}")
            return []

    def normalize_species_name(self, raw_name: str) -> str:
        """
        Normaliza el nombre de una especie al formato de la BD:
        1. Title case exacto
        2. Nombre genérico (una sola palabra del input)
        3. Levenshtein como fallback para typos
        """
        normalized = raw_name.strip().title()

        # 1. Coincidencia exacta
        if normalized in self._known_species:
            return normalized

        # 2. La BD usa nombres genéricos: probar cada palabra por separado
        for word in normalized.split():
            if word in self._known_species:
                return word

        # 3. Levenshtein como fallback
        if self._known_species:
            match, score, _ = process.extractOne(
                normalized, self._known_species, scorer=fuzz.WRatio
            )
            if score >= 80:
                return match

        return normalized

    def _extract_and_normalize_species(self, question: str) -> str:
        """
        Detecta y normaliza TODOS los nombres de especies en la pregunta,
        manteniendo el resto de la pregunta intacta.
        """
        print(f"Original question: {question}")
        sorted_species = sorted(self._known_species, key=len, reverse=True)

        matched_positions = set()
        replacements = {}  # {(start, end): normalized_species}

        lower_q = question.lower()

        # 1. Coincidencias directas, marcando posiciones para no solapar
        for species in sorted_species:
            lower_s = species.lower()
            idx = lower_q.find(lower_s)
            if idx != -1 and idx not in matched_positions:
                replacements[(idx, idx + len(lower_s))] = species
                matched_positions.update(range(idx, idx + len(lower_s)))

        print(f"Replacements found: {replacements}")
        # 2. Levenshtein para zonas no matcheadas (bigramas y trigramas)
        words = question.split()
        word_positions = []
        pos = 0
        for word in words:
            idx = question.find(word, pos)
            word_positions.append((idx, idx + len(word), word))
            pos = idx + len(word)

        for i in range(len(words)):
            for n in (3, 2):
                if i + n <= len(words):
                    chunk_words = words[i:i+n]
                    start = word_positions[i][0]
                    end = word_positions[i+n-1][1]
                    if any(p in matched_positions for p in range(start, end)):
                        continue
                    chunk = " ".join(chunk_words).strip("?.,!")
                    normalized = self.normalize_species_name(chunk)
                    if normalized != chunk.title():
                        replacements[(start, end)] = normalized
                        matched_positions.update(range(start, end))
                        break

        # 3. Reconstruir la pregunta reemplazando solo las posiciones matcheadas
        if not replacements:
            return question

        result = []
        prev = 0
        for start, end in sorted(replacements.keys()):
            result.append(question[prev:start])      # texto original intacto
            result.append(replacements[(start, end)]) # especie normalizada
            prev = end
        result.append(question[prev:])               # resto de la pregunta

        result = "".join(result)
        print(f"Normalized question: {result}")
        return "".join(result)

    def generate_cypher(self, question: str) -> str:
        """Genera una query Cypher a partir de una pregunta en lenguaje natural."""

        # Normalizar especies antes de pasarle la pregunta al LLM
        question = self._extract_and_normalize_species(question)

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