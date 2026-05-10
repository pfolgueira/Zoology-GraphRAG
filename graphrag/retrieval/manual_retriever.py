from typing import List, Dict, Any, Tuple
from rapidfuzz import process, fuzz

from graphrag.config import get_settings
from graphrag.graph.neo4j_manager import Neo4jManager


class ManualRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.settings = get_settings()
        self._known_species: List[str] = self._load_known_species()

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
        1. Coincidencia exacta en title case
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

    def retrieve(self, query_category: str, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Recupera y ejecuta una consulta Cypher predefinida basada en la categoría.
        """

        # Normalizar species_name si viene en kwargs
        if "species_name" in kwargs:
            kwargs["species_name"] = self.normalize_species_name(kwargs["species_name"])

        query_templates = {
            "species_full_profile": """
                MATCH (s:Species {name: $species_name})
                OPTIONAL MATCH (s)-[:BELONGS_TO_CLASS]->(c:AnimalClass)
                OPTIONAL MATCH (s)-[:MEMBER_OF_FAMILY]->(f:Family)
                OPTIONAL MATCH (s)-[:HAS_CONSERVATION_STATUS]->(cs:ConservationStatus)
                OPTIONAL MATCH (s)-[:HAS_SKELETAL_STRUCTURE]->(ss:SkeletalStructure)
                OPTIONAL MATCH (s)-[:REPRODUCES_VIA]->(rm:ReproductionMethod)
                OPTIONAL MATCH (s)-[:HAS_ACTIVITY_CYCLE]->(ac:ActivityCycle)
                OPTIONAL MATCH (s)-[:HAS_DIET_TYPE]->(d:DietType)
                OPTIONAL MATCH (s)-[:FEEDS_ON]->(fs:FoodSource)
                OPTIONAL MATCH (s)-[:PREYS_ON]->(prey:Species)
                OPTIONAL MATCH (s)-[:ORGANIZED_IN]->(soc:SocialStructure)
                OPTIONAL MATCH (s)-[:LIVES_IN_ENVIRONMENT]->(env:EnvironmentType)
                OPTIONAL MATCH (s)-[:INHABITS]->(hab:Habitat)
                OPTIONAL MATCH (s)-[:FOUND_IN]->(loc:Location)
                OPTIONAL MATCH (s)-[m:MIGRATES_TO]->(mig_loc:Location)
                RETURN s.name AS Species, 
                    collect(DISTINCT c.type) AS Class, 
                    collect(DISTINCT f.type) AS Family,
                    collect(DISTINCT cs.type) AS ConservationStatuses, 
                    collect(DISTINCT ss.type) AS SkeletalStructures,
                    collect(DISTINCT rm.type) AS ReproductionMethods,
                    collect(DISTINCT ac.type) AS ActivityCycles,
                    collect(DISTINCT d.type) AS Diets, 
                    collect(DISTINCT fs.type) AS FoodSources,
                    collect(DISTINCT prey.name) AS PreysOn,
                    collect(DISTINCT soc.type) AS SocialStructures,
                    collect(DISTINCT env.type) AS Environments,
                    collect(DISTINCT hab.type) AS Habitats,
                    collect(DISTINCT loc.type) AS Locations, 
                    collect(DISTINCT mig_loc.type) AS MigrationLocations,
                    collect(DISTINCT m.season) AS MigrationSeasons,
                    s.weight_max_kg AS MaxWeight, 
                    s.top_speed_kmh AS TopSpeed,
                    s.lifespan_years AS Lifespan
            """,
            "endangered_by_environment": """
                MATCH (s:Species)-[:LIVES_IN_ENVIRONMENT]->(e:EnvironmentType {type: $environment_name})
                MATCH (s)-[:HAS_CONSERVATION_STATUS]->(c:ConservationStatus)
                WHERE c.type IN ['CR (Critically Endangered)', 'EN (Endangered)']
                RETURN s.name, c.type, e.type
                ORDER BY c.type
            """,
            "predator_prey_chain": """
                MATCH (s:Species {name: $species_name})
                OPTIONAL MATCH (predator:Species)-[:PREYS_ON]->(s)
                OPTIONAL MATCH (s)-[:PREYS_ON]->(prey:Species)
                OPTIONAL MATCH (s)-[:FEEDS_ON]->(food:FoodSource)
                RETURN 
                    s.name AS species,
                    collect(DISTINCT predator.name) AS hunted_by,
                    collect(DISTINCT prey.name) AS hunts,
                    collect(DISTINCT food.type) AS feeds_on
            """,
            "social_structure_by_class": """
                MATCH (s:Species)-[:BELONGS_TO_CLASS]->(c:AnimalClass {type: $class_name})
                MATCH (s)-[:ORGANIZED_IN]->(ss:SocialStructure)
                RETURN ss.type AS social_structure, count(s) AS species_count
                ORDER BY species_count DESC
            """,
        }

        cypher = query_templates.get(query_category)
        if not cypher:
            error_msg = f"Error: Categoría de consulta '{query_category}' no encontrada."
            print(error_msg)
            return error_msg, []

        try:
            results = self.neo4j.execute_query(cypher, parameters=kwargs)
            return cypher, results
        except Exception as e:
            print(f"Error ejecutando Cypher predefinido ({query_category}): {e}")
            print(f"Parámetros recibidos: {kwargs}")
            return cypher, []