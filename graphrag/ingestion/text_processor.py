import uuid
import time
from typing import List, Dict, Any
from tqdm import tqdm
from ..graph.neo4j_manager import Neo4jManager
from ..utils.chunking import chunk_text
from ..utils.embeddings import EmbeddingGenerator
from .entity_extractor import EntityExtractor
from .graph_cleaner import GraphCleaner
from .hypothetical_question_generator import HypotheticalQuestionGenerator
from typing import Literal
from pydantic import BaseModel, Field
from ..llm.ollama_client import OllamaClient
from ..llm.gemini_client import GeminiClient

class SpeciesResolution(BaseModel):
        status: Literal["MATCH", "NEW", "DISCARD"] = Field(..., description="MUST BE 'MATCH' if the extracted name is an exact match, a synonym, OR a specific sub-type/breed that can be generalized to a broader animal present in the canonical list. Set to 'NEW' ONLY if it represents a completely unrepresented animal lineage. Set to 'DISCARD' if the entity is not an animal.")
        resolved_name: str = Field(..., description="If status is 'MATCH', this MUST be the exact name from the canonical list (use the broader parent name if generalizing, e.g., 'Penguin' for 'Emperor Penguin'). If status is 'NEW', provide a standard, high-level generic English name for the new animal. If status is 'DISCARD' you must return an empty string")


class TextProcessor:
    def __init__(
            self,
            neo4j_manager: Neo4jManager,
            species_names: List[str],
            chunk_size: int = 500,
            chunk_overlap: int = 50
    ):
        self.neo4j = neo4j_manager
        try:
            self.neo4j.create_vector_index(
                index_name="species_name_embeddings",
                label="Species",
                property_name="name_embedding"
            )
        except Exception as e:
            print(f"Error al crear el índice vectorial: {e}")

        self.embedding_gen = EmbeddingGenerator()
        self.entity_extractor = EntityExtractor()
        self.graph_cleaner = GraphCleaner(self.neo4j)
        self.hypothetical_question_generator = HypotheticalQuestionGenerator()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.gemini_client = GeminiClient()
        self.ollama_client = OllamaClient()
        self.species_names = set(species_names)
        self.species_names_lower = {animal.lower() for animal in species_names}


    def _load_species_names(self):
        try:
            with open('../scraping/species.txt', 'r', encoding='utf-8') as fichero:
                lista = [linea.strip() for linea in fichero.readlines() if linea.strip()]
            return lista
        except FileNotFoundError:
            print(f"El fichero no existe.")
            return []


    def process_document(
            self,
            text: str,
            document_id: str = None,
            metadata: Dict[str, Any] = None
    ):
        """
        Procesa un documento: lo divide en chunks, extrae entidades y relaciones,
        y lo almacena en Neo4j.
        """
        document_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}

        # 1. Crear nodo de documento
        self._create_document_node(document_id, metadata)

        # 2. Dividir en chunks
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        print(f"Documento dividido en {len(chunks)} chunks")

        # 3. Procesar cada chunk
        for i, chunk in enumerate(tqdm(chunks, desc="Procesando chunks")):
            chunk_id = f"{document_id}_chunk_{i}"
            self._process_chunk(chunk_id, chunk.page_content, chunk.metadata, document_id, i)
            time.sleep(2)
            
        # Limpiar el grafo para eliminar nodos o relaciones inconsistentes
        self.graph_cleaner.clean_graph()

        print(f"Documento {document_id} procesado exitosamente")

    def _create_document_node(self, document_id: str, metadata: Dict[str, Any]):
        """Crea un nodo de documento en Neo4j."""
        query = """
        MERGE (d:Document {id: $document_id})
        SET d += $metadata
        """
        self.neo4j.execute_query(query, {
            "document_id": document_id,
            "metadata": metadata
        })

    def _process_chunk(
            self,
            chunk_id: str,
            text: str,
            metadata: Dict[str, Any],
            document_id: str,
            index: int
    ):
        """Procesa un chunk individual."""
        # Generar embedding
        embedding = self.embedding_gen.embed_text(text)

        # Crear nodo de chunk
        query = """
        MATCH (d:Document {id: $document_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.embedding = $embedding,
            c.index = $index
        SET c += $metadata
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "index": index,
            "document_id": document_id
        })

        # Extraer entidades y relaciones
        tmp_entities, tmp_relationships = self.entity_extractor.extract_entities_and_relationships(text)

        entities, relationships = self.entity_extractor.review_extraction(text, tmp_entities, tmp_relationships)

        # Almacenar entidades
        for label, entities_list in entities.items():
            for entity_data in entities_list:
                self._store_entity(label, entity_data, chunk_id)

        # Almacenar relaciones
        for rel_type, rels_list in relationships.items():
            for rel_data in rels_list:                
                self._store_relationship(rel_type, rel_data, chunk_id)

        # Generar preguntas hipotéticas para RAG
        hypothetical_questions = self.hypothetical_question_generator.generate_hypothetical_questions(text)
        for question in hypothetical_questions:
            question_embedding = self.embedding_gen.embed_text(question)
            self._store_hypothetical_question(question, question_embedding, chunk_id)

    def _store_entity(self, label: str, entity_data: Dict[str, Any], chunk_id: str):
        """Almacena una entidad y la conecta al chunk, 
           aplicando Entity Resolution a los nombres de las especies."""
        # Detecta si la clave principal es 'name' (para Species) o 'type' (para el resto)
        primary_value = entity_data.get("name") or entity_data.get("type")
        
        if not primary_value:
            return
        
        # Entitie resolution para las Species
        if label == "Species":
            primary_value = self._resolve_species_name(primary_value)
            if primary_value is None:
                return

        # Extrae el resto de propiedades ignorando las claves principales
        properties = {
            k: v for k, v in entity_data.items() 
            if k not in ["name", "type"] and v is not None
        }

        if label == "Species":
            properties["name_embedding"] = self.embedding_gen.embed_text(primary_value)

        primary_key = "name" if label == "Species" else "type"

        query = f"""
        MATCH (c:Chunk {{id: $chunk_id}})
        MERGE (e:{label} {{{primary_key}: $primary_value}})
        SET e += $properties
        MERGE (c)-[:HAS_ENTITY]->(e)
        """
        
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "primary_value": primary_value,
            "properties": properties
        })

    def _store_relationship(self, rel_type: str, rel_data: Dict[str, Any], chunk_id: str):
        """
        Almacena una relación semántica asegurando la compatibilidad con el esquema
        y aplicando Entity Resolution a las especies implicadas.
        """
        
        # Entidades implicadas en la relación
        source_val = rel_data.get("source")
        target_val = rel_data.get("target")

        if hasattr(source_val, 'value'): source_val = source_val.value
        if hasattr(target_val, 'value'): target_val = target_val.value

        if not source_val or not target_val:
            print(f"Relación [{rel_type}] omitida: Faltan origen o destino en el chunk {chunk_id}")
            return

        source_val = self._resolve_species_name(source_val)
        if source_val is None:
            return
        source_embedding = self.embedding_gen.embed_text(source_val)

        # Resolución y embedding del destino (si aplica)
        target_embedding = None
        if rel_type == "PREYS_ON":
            target_val = self._resolve_species_name(target_val)
            if target_val is None:
                return
            target_embedding = self.embedding_gen.embed_text(target_val)

        # 3. Mapeo de etiquetas y propiedades basado estrictamente en tu diseño
        source_label = "Species"
        source_prop = "name"

        # Mapeamos el tipo de relación con la etiqueta del nodo de destino y su clave
        target_mapping = {
            "MEMBER_OF_FAMILY": ("Family", "type"),
            "BELONGS_TO_CLASS": ("AnimalClass", "type"),
            "HAS_SKELETAL_STRUCTURE": ("SkeletalStructure", "type"),
            "REPRODUCES_VIA": ("ReproductionMethod", "type"),
            "LIVES_IN_ENVIRONMENT": ("EnvironmentType", "type"),
            "INHABITS": ("Habitat", "type"),
            "FOUND_IN": ("Location", "type"),
            "MIGRATES_TO": ("Location", "type"),
            "HAS_ACTIVITY_CYCLE": ("ActivityCycle", "type"),
            "ORGANIZED_IN": ("SocialStructure", "type"),
            "HAS_DIET_TYPE": ("DietType", "type"),
            "PREYS_ON": ("Species", "name"),
            "FEEDS_ON": ("FoodSource", "type"),
            "HAS_CONSERVATION_STATUS": ("ConservationStatus", "type")
        }

        target_label, target_prop = target_mapping.get(rel_type, ("Unknown", "type"))
        
        if target_label == "Unknown":
            print(f"Relación [{rel_type}] no reconocida en el esquema. Omitiendo.")
            return

        # Extraer propiedades adicionales de la relación
        properties = {
            k: (v.value if hasattr(v, 'value') else v) 
            for k, v in rel_data.items() 
            if k not in ["source", "target", "description"] and v is not None
        }

        query = f"""
        MATCH (c:Chunk {{id: $chunk_id}})
        MERGE (source:{source_label} {{{source_prop}: $source_val}})
        SET source.name_embedding = $source_embedding
        
        MERGE (target:{target_label} {{{target_prop}: $target_val}})
        """
        
        # Inyectamos el vector en el destino solo si es una especie (PREYS_ON)
        if target_embedding is not None:
            query += "SET target.name_embedding = $target_embedding\n"
            
        query += f"""
        MERGE (c)-[:HAS_ENTITY]->(source)
        MERGE (c)-[:HAS_ENTITY]->(target)
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        """
    
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "source_val": source_val,
            "target_val": target_val,
            "properties": properties,
            "source_embedding": source_embedding,
            "target_embedding": target_embedding
        })

    def _resolve_species_name(self, extracted_name: str) -> str:
        """
        Utiliza el LLM para evaluar si el nombre extraído es un sinónimo, 
        variación o el mismo animal que alguno de la lista species_names. Si es nuevo, obtiene su nombre común.
        """
        extracted_name_lower = extracted_name.lower()
        # Si los nombres de la especie coinciden exactamente no se realiza la llamada al LLM
        if extracted_name_lower in self.species_names_lower:
            return extracted_name.title()     
        # Si el nombre del animal está en plural, comprobamos si coincide con alguan especie de la lista eliminando la 's' final.
        elif extracted_name_lower.endswith('s'):
            singular_name = extracted_name_lower[:-1]
            if singular_name in self.species_names_lower:
                return extracted_name[:-1].title()
        
        extracted_name = extracted_name.title()

        # Similitud vectorial para reducir el espacio de búsqueda a las especies más cercanas semánticamente
        entity_embedding = self.embedding_gen.embed_text(extracted_name)
    
        # Recuperamos el top 10 candidatos
        candidate_species = self._get_top_k_candidate_species(entity_embedding, top_k=10)

        if len(candidate_species) == 0:
            self.species_names.add(extracted_name)
            self.species_names_lower.add(extracted_name_lower)
            print(f"Nueva especie añadida: {extracted_name}")
            
            return extracted_name

        system_prompt = """You are an expert Knowledge Graph engineer and taxonomist working with a simplified, high-level animal ontology. 
        Your task is Entity Resolution. You will be given an extracted animal name and a canonical list of base animal entities.
                
        Rules:
        1. SYNONYMS & EXACT MATCHES: If the extracted name is a plural form (e.g., 'Snakes' -> 'Snake', 'Wolves' -> 'Wolf'), a known synonym, regional name, or refers to the same animal as one in the canonical list (e.g., 'Cougar' -> 'Mountain Lion', 'Orca' -> 'Killer Whale'), set status to 'MATCH' and output the exact matching name from the canonical list.

        2. GENERALIZATION (SUB-SPECIES TO PARENT): If the extracted name is a specific type, breed, or sub-species of a broader generic animal that is already present in the canonical list, you MUST generalize it to the broader entity. Set status to 'MATCH'. 
        Examples: 
        - 'Laysan Albatross' -> 'Albatross'
        - 'Emperor Penguin' -> 'Penguin'
        - 'Grizzly Bear' -> 'Bear'

        3. NEW ENTITIES: If the extracted name is clearly an animal but represents a completely different animal lineage not covered by ANY broad category or synonym in the list, set status to 'NEW'. Provide the most standard, common English name for this new species in 'resolved_name'. Keep new names at a high, generic level if possible (e.g., output 'Eagle' instead of 'Bald Eagle').

        4. BROAD CLASSIFICATIONS & TRAITS (DISCARD): If the extracted name is a dietary classification (e.g., 'Carnivore', 'Herbivore', 'Predator'), a broad taxonomic class (e.g., 'Mammal', 'Reptile', 'Bird', 'Amphibian'), or a generic biological family term rather than a specific animal (e.g., 'Feline', 'Canine', 'Big Cat'), you MUST discard it. Set status to 'DISCARD' and set 'resolved_name' to an empty string.

        5. NON-ANIMALS (DISCARD): If the extracted name is NOT an animal (e.g., a plant, geographic location, inanimate object, person, or abstract concept), you MUST discard it. Set status to 'DISCARD' and set 'resolved_name' to an empty string."""

        user_message = f"Extracted Name: {extracted_name}\n\nCanonical List of Species: {candidate_species}"

        resolution: SpeciesResolution = self.gemini_client.structured_output(
            prompt=user_message,
            schema=SpeciesResolution,
            system_prompt=system_prompt
        )

        if resolution.status == "DISCARD":
            print(f"Especie {extracted_name} descartada")
            return None

        final_name = resolution.resolved_name.title()

        final_name_lower = final_name.lower()

        if resolution.status == "NEW" and final_name_lower not in self.species_names_lower:
            self.species_names.add(final_name)
            self.species_names_lower.add(final_name_lower)
            print(f"Nueva especie añadida: {final_name}")
            
        return final_name
    
    def _get_top_k_candidate_species(self, entity_embedding: list[float], top_k: int = 10) -> list[str]:
        """
        Busca las Top K especies más parecidas semánticamente en Neo4j.
        """
        cypher_query = """
        CALL db.index.vector.queryNodes('species_name_embeddings', $top_k, $entity_embedding)
        YIELD node, score
        WHERE score > 0.65 // Umbral mínimo de similitud
        RETURN node.name AS species_name
        """
        try:
            # Ajusta "self.execute_query" o el nombre de tu método de ejecución
            results = self.execute_query(cypher_query, {
                "entity_embedding": entity_embedding,
                "top_k": top_k
            })
            return [record["species_name"] for record in results]
        except Exception as e:
            print(f"Error buscando candidatos vectoriales: {e}")
            return []
    
    def _store_hypothetical_question(self, question: str, question_embedding: List[float], chunk_id: str):
        self.neo4j.execute_query("""
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (q:Question {text: $question})
        SET q.question_embedding = $question_embedding
        MERGE (c)-[:HAS_QUESTION]->(q)
        """, {
            "chunk_id": chunk_id,
            "question": question,
            "question_embedding": question_embedding
        })
