from typing import List, Dict, Any, Tuple

from graphrag.config import get_settings
from graphrag.graph.neo4j_manager import Neo4jManager


class ManualRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.settings = get_settings()

    def retrieve(self, query_category: str, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Recupera y ejecuta una consulta Cypher predefinida basada en la categoría.

        Args:
            query_category (str): El identificador de la consulta manual (ej. 'species_full_profile').
            **kwargs: Parámetros variables necesarios para la consulta (ej. species_name="Lion").

        Returns:
            Tupla con (cypher_query, results)
        """
        
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
            # "apex_predators_by_location": """
            #     MATCH (predator:Species)-[:FOUND_IN]->(l:Location {type: $location_name})
            #     MATCH (predator)-[:PREYS_ON]->(prey:Species)
            #     WHERE NOT EXISTS { ()-[:PREYS_ON]->(predator) }
            #     RETURN predator.name AS ApexPredator, 
            #         predator.weight_max_kg AS Weight,
            #         collect(DISTINCT prey.name) AS Preys
            #     ORDER BY predator.weight_max_kg DESC
            # """,
            # "migration_diet_analysis": """
            #     MATCH (s:Species)-[m:MIGRATES_TO]->(l:Location {type: $location_name})
            #     WHERE toLower(m.season) = toLower($season_name)
            #     OPTIONAL MATCH (s)-[:FEEDS_ON]->(fs:FoodSource)
            #     OPTIONAL MATCH (s)-[:HAS_CONSERVATION_STATUS]->(cs:ConservationStatus)
            #     RETURN s.name AS Species, m.season AS Season, cs.type AS Status,
            #         collect(DISTINCT fs.type) AS FoodSources
            # """,
            # "endangered_by_environment": """
            #     MATCH (s:Species)-[:LIVES_IN_ENVIRONMENT]->(e:EnvironmentType {type: $environment_name})
            #     MATCH (s)-[:HAS_CONSERVATION_STATUS]->(cs:ConservationStatus)
            #     WHERE cs.type IN ['Endangered', 'Critically Endangered', 'Vulnerable']
            #     OPTIONAL MATCH (s)-[:REPRODUCES_VIA]->(rm:ReproductionMethod)
            #     RETURN s.name AS Species, cs.type AS Status, s.lifespan_years AS Lifespan,
            #         collect(DISTINCT rm.type) AS ReproductionMethods
            #     ORDER BY s.lifespan_years ASC
            # """,
            # "family_extremes_comparison": """
            #     MATCH (s:Species)-[:MEMBER_OF_FAMILY]->(f:Family {type: $family_name})
            #     RETURN s.name AS Species, s.weight_max_kg AS MaxWeightKG, 
            #         s.top_speed_kmh AS TopSpeedKMH, s.length_max_m AS MaxLengthM
            #     ORDER BY s.weight_max_kg DESC
            #     LIMIT 5
            # """
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