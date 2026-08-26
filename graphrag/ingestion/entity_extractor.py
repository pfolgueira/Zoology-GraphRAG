from email.mime import text
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, Field
import json
from ..llm.gemini_client import GeminiClient

from pydantic import ValidationError


# ==========================================
#   ENUMS
# ==========================================

class AnimalClassEnum(str, Enum):
    MAMMAL = "Mammal"
    REPTILE = "Reptile"
    BIRD = "Bird"
    FISH = "Fish"
    AMPHIBIAN = "Amphibian"

class SkeletalStructureEnum(str, Enum):
    VERTEBRATE = "Vertebrate"
    INVERTEBRATE = "Invertebrate"

class ReproductionMethodEnum(str, Enum):
    OVIPAROUS = "Oviparous"
    VIVIPAROUS = "Viviparous"
    OVOVIVIPAROUS = "Ovoviviparous"

class EnvironmentTypeEnum(str, Enum):
    AQUATIC = "Aquatic"
    TERRESTRIAL = "Terrestrial"
    AERIAL = "Aerial"

class ActivityCycleEnum(str, Enum):
    NOCTURNAL = "Nocturnal"
    DIURNAL = "Diurnal"
    CREPUSCULAR = "Crepuscular"

class SocialStructureEnum(str, Enum):
    SOLITARY = "Solitary"
    PAIR_LIVING = "Pair-living"
    FAMILY_GROUP = "Family group"
    HERD = "Herd"
    PACK = "Pack"
    COLONY = "Colony"
    EUSOCIAL = "Eusocial"

class DietTypeEnum(str, Enum):
    CARNIVORE = "Carnivore"
    HERBIVORE = "Herbivore"
    OMNIVORE = "Omnivore"

class FoodSourceEnum(str, Enum):
    GRASS = "Grass"
    LEAVES = "Leaves"
    FRUITS = "Fruits"
    SEEDS = "Seeds"
    BARK = "Bark"
    NECTAR = "Nectar"
    AQUATIC_PLANTS = "Aquatic Plants"

class ConservationStatusEnum(str, Enum):
    EX = "EX (Extinct)"
    EW = "EW (Extinct in the Wild)"
    CR = "CR (Critically Endangered)"
    EN = "EN (Endangered)"
    VU = "VU (Vulnerable)"
    NT = "NT (Near Threatened)"
    LC = "LC (Least Concern)"
    DD = "DD (Data Deficient)"
    NE = "NE (Not Evaluated)"


# ==========================================
#   ENTITIES
# ==========================================

class SpeciesNode(BaseModel):
    name: str = Field(..., description="Common name of the species in colloquial language.")
    weight_min_kg: Optional[float] = Field(None, description="Minimum recorded weight of an adult specimen in kilograms.")
    weight_max_kg: Optional[float] = Field(None, description="Maximum recorded weight of an adult specimen in kilograms.")
    length_min_m: Optional[float] = Field(None, description="Minimum recorded body length in meters.")
    length_max_m: Optional[float] = Field(None, description="Maximum recorded body length in meters.")
    height_min_m: Optional[float] = Field(None, description="Minimum recorded height to the highest point of the animal in meters.")
    height_max_m: Optional[float] = Field(None, description="Maximum recorded height to the highest point of the animal in meters.")
    top_speed_kmh: Optional[float] = Field(None, description="Maximum speed the species can reach.")
    lifespan_years: Optional[float] = Field(None, description="Lifespan of the species.")
    gestation_period_days: Optional[float] = Field(None, description="Average duration of embryo development until birth in days.")
    maturity_age_years: Optional[float] = Field(None, description="Estimated age in years at which the species reaches sexual maturity or independence.")
    
class FamilyNode(BaseModel):
    type: str = Field(..., description="Taxonomic classification category: Hominidae, Felidae, Canidae, Equidae, etc.")

class HabitatNode(BaseModel):
    type: str = Field(..., description="Type of natural environment or ecosystem: Forest, Desert, Grassland, Ocean, etc."
                        "STRICT RULE: DO NOT output geographical locations, countries, continents, or specific map regions here. "
                        "For example, NEVER output 'Africa', 'Amazon Rainforest', or 'Spain'. If the text mentions a geographical location, use the LocationNode instead.")

class LocationNode(BaseModel):
    type: str = Field(..., description="The geographical location of the animal. "
            "STRICT RULE: You MUST generalize the location to a Country, Continent, Sea, or Ocean. "
            "Do NOT output specific regions, natural parks, or habitats. "
            "Examples: If the text says 'Amazon Rainforest', output 'Brazil' or 'South America'. "
            "If it says 'Serengeti', output 'Tanzania'. If it says 'Great Barrier Reef', output 'Pacific Ocean'.")
    
class AnimalClassNode(BaseModel):
    type: AnimalClassEnum = Field(..., description="Main biological category.")

class SkeletalStructureNode(BaseModel):
    type: SkeletalStructureEnum = Field(..., description="Anatomical classification.")

class ReproductionMethodNode(BaseModel):
    type: ReproductionMethodEnum = Field(..., description="Embryonic development mechanism.")

class EnvironmentTypeNode(BaseModel):
    type: EnvironmentTypeEnum = Field(..., description="Main physical environment where the species develops its life cycle.")

class ActivityCycleNode(BaseModel):
    type: ActivityCycleEnum = Field(..., description="Temporal pattern of biological activity.")

class SocialStructureNode(BaseModel):
    type: SocialStructureEnum = Field(..., description="Social organization pattern of a species.")

class DietTypeNode(BaseModel):
    type: DietTypeEnum = Field(..., description="Classification according to dietary diet.")

class FoodSourceNode(BaseModel):
    type: FoodSourceEnum = Field(..., description="Type of plant resource on which a herbivorous or omnivorous species feeds.")

class ConservationStatusNode(BaseModel):
    type: ConservationStatusEnum = Field(..., description="Extinction risk level according to IUCN.")


# ==========================================
#   RELATIONSHIPS
# ==========================================

class MemberOfFamilyRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Family")
    description: str = Field("MEMBER_OF_FAMILY", description="Indicates the taxonomic family to which the species belongs within the biological classification.")

class BelongsToClassRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Animal Class")
    description: str = Field("BELONGS_TO_CLASS", description="Specifies the biological class of the species.")

class HasSkeletalStructureRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Skeletal Structure")
    description: str = Field("HAS_SKELETAL_STRUCTURE", description="Defines the type of skeletal structure of the species.")

class ReproducesViaRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Reproduction Method")
    description: str = Field("REPRODUCES_VIA", description="Indicates the reproduction method of the species.")

class LivesInEnvironmentRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Environment Type")
    description: str = Field("LIVES_IN_ENVIRONMENT", description="Describes the physical environment where the species lives.")

class InhabitsRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Habitat")
    description: str = Field("INHABITS", description="Relates the species to the type of habitat or biome where it is usually found.")

class FoundInRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Location")
    description: str = Field("FOUND_IN", description="Indicates the geographical regions where the species is naturally present.")

class MigratesToRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Location")
    season: Optional[str] = Field(None, description="Indicates the season of the year when the migration takes place.")
    description: str = Field("MIGRATES_TO", description="Indicates the regions to which the species moves during migratory processes.")

class HasActivityCycleRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Activity Cycle")
    description: str = Field("HAS_ACTIVITY_CYCLE", description="Describes the period of the day when the species is mainly active.")

class OrganizedInRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Social Structure")
    description: str = Field("ORGANIZED_IN", description="Represents the form of social organization of the species.")

class HasDietTypeRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Diet Type")
    description: str = Field("HAS_DIET_TYPE", description="Defines the type of dietary diet of the species.")

class PreysOnRel(BaseModel):
    source: str = Field(..., description="Name of the predator Species")
    target: str = Field(..., description="Name of the prey Species")
    description: str = Field("PREYS_ON", description="Indicates the consumption of ANY ANIMAL MATTER. This STRICTLY includes large preys, Insect, Fish, Crustacean, Squid, etc. Use PREYS_ON ONLY if the food is a living creature/animal.")

class FeedsOnRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Food Source")
    description: str = Field("FEEDS_ON", description="Represents the consumption of NON-ANIMAL food sources. This includes Grass, Leaves, Fruits, Seeds, Bark, Nectar and Aquatic Plants.")

class HasConservationStatusRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Conservation Status")
    description: str = Field("HAS_CONSERVATION_STATUS", description="Indicates the conservation status of the species according to its extinction risk.")


# ==========================================
#   GLOBAL EXTRACTION CONTAINER
# ==========================================

class GraphExtraction(BaseModel):
    # Entities
    species: List[SpeciesNode] = Field(default_factory=list)
    families: List[FamilyNode] = Field(default_factory=list)
    animal_classes: List[AnimalClassNode] = Field(default_factory=list)
    skeletal_structures: List[SkeletalStructureNode] = Field(default_factory=list)
    reproduction_methods: List[ReproductionMethodNode] = Field(default_factory=list)
    environment_types: List[EnvironmentTypeNode] = Field(default_factory=list)
    habitats: List[HabitatNode] = Field(default_factory=list)
    locations: List[LocationNode] = Field(default_factory=list)
    activity_cycles: List[ActivityCycleNode] = Field(default_factory=list)
    social_structures: List[SocialStructureNode] = Field(default_factory=list)
    diet_types: List[DietTypeNode] = Field(default_factory=list)
    food_sources: List[FoodSourceNode] = Field(default_factory=list)
    conservation_statuses: List[ConservationStatusNode] = Field(default_factory=list)

    # Relationships
    member_of_family_rels: List[MemberOfFamilyRel] = Field(default_factory=list)
    belongs_to_class_rels: List[BelongsToClassRel] = Field(default_factory=list)
    has_skeletal_structure_rels: List[HasSkeletalStructureRel] = Field(default_factory=list)
    reproduces_via_rels: List[ReproducesViaRel] = Field(default_factory=list)
    lives_in_environment_rels: List[LivesInEnvironmentRel] = Field(default_factory=list)
    inhabits_rels: List[InhabitsRel] = Field(default_factory=list)
    found_in_rels: List[FoundInRel] = Field(default_factory=list)
    migrates_to_rels: List[MigratesToRel] = Field(default_factory=list)
    has_activity_cycle_rels: List[HasActivityCycleRel] = Field(default_factory=list)
    organized_in_rels: List[OrganizedInRel] = Field(default_factory=list)
    has_diet_type_rels: List[HasDietTypeRel] = Field(default_factory=list)
    preys_on_rels: List[PreysOnRel] = Field(default_factory=list)
    feeds_on_rels: List[FeedsOnRel] = Field(default_factory=list)
    has_conservation_status_rels: List[HasConservationStatusRel] = Field(default_factory=list)


class EntityPointer(BaseModel):
    category: str = Field(..., description="The category of the entity to delete (e.g., 'Species', 'Location', 'Habitat'...).")
    name_or_type: str = Field(..., description="The specific 'name' or 'type' value of the entity to delete.")

class RelationshipPointer(BaseModel):
    source: str = Field(..., description="The exact 'source' string from the initial extraction.")
    target: str = Field(..., description="The exact 'target' string from the initial extraction.")
    relationship_type: str = Field(..., description="The name of the relationship to delete (e.g., 'INHABITS', 'FEEDS_ON', 'PREYS_ON'...).")

# Agrupamos las adiciones para mantener el esquema limpio para el LLM
class MissingEntities(BaseModel):
    species: List[SpeciesNode] = Field(default=[], description="Missed species")
    families: List[FamilyNode] = Field(default=[], description="Missed families")
    habitats: List[HabitatNode] = Field(default=[], description="Missed habitats. DO NOT output geographical locations, countries, continents, or specific map regions here. Type of natural environment or ecosystem: Forest, Desert, Grassland, Ocean, etc.")
    locations: List[LocationNode] = Field(default=[], description="Missed locations")
    animal_classes: List[AnimalClassNode] = Field(default=[], description="Missed animal classes")
    skeletal_structures: List[SkeletalStructureNode] = Field(default=[], description="Missed skeletal structures")
    reproduction_methods: List[ReproductionMethodNode] = Field(default=[], description="Missed reproduction methods")
    environment_types: List[EnvironmentTypeNode] = Field(default=[], description="Missed environment types")
    activity_cycles: List[ActivityCycleNode] = Field(default=[], description="Missed activity cycles")
    social_structures: List[SocialStructureNode] = Field(default=[], description="Missed social structures")
    diet_types: List[DietTypeNode] = Field(default=[], description="Missed diet types")
    food_sources: List[FoodSourceNode] = Field(default=[], description="Missed food sources")
    conservation_statuses: List[ConservationStatusNode] = Field(default=[], description="Missed conservation statuses")

class MissingRelationships(BaseModel):
    member_of_family_rels: List[MemberOfFamilyRel] = Field(default=[], description="Missed member-of-family relationships")
    belongs_to_class_rels: List[BelongsToClassRel] = Field(default=[], description="Missed belongs-to-class relationships")
    has_skeletal_structure_rels: List[HasSkeletalStructureRel] = Field(default=[], description="Missed has-skeletal-structure relationships")
    reproduces_via_rels: List[ReproducesViaRel] = Field(default=[], description="Missed reproduces-via relationships")
    lives_in_environment_rels: List[LivesInEnvironmentRel] = Field(default=[], description="Missed lives-in-environment relationships")
    inhabits_rels: List[InhabitsRel] = Field(default=[], description="Missed inhabits relationships")
    found_in_rels: List[FoundInRel] = Field(default=[], description="Missed found-in relationships")
    migrates_to_rels: List[MigratesToRel] = Field(default=[], description="Missed migrates-to relationships")
    has_activity_cycle_rels: List[HasActivityCycleRel] = Field(default=[], description="Missed has-activity-cycle relationships")
    organized_in_rels: List[OrganizedInRel] = Field(default=[], description="Missed organized-in relationships")
    has_diet_type_rels: List[HasDietTypeRel] = Field(default=[], description="Missed has-diet-type relationships")
    preys_on_rels: List[PreysOnRel] = Field(default=[], description="Missed preys-on relationships")
    feeds_on_rels: List[FeedsOnRel] = Field(default=[], description="Missed feeds-on relationships")
    has_conservation_status_rels: List[HasConservationStatusRel] = Field(default=[], description="Missed has-conservation-status relationships")

class GraphPatch(BaseModel):
    thought_process: str = Field(
        ...,
        description=(
            "You MUST structure your analysis in two parts: "
            "1. [HALLUCINATION CHECK]: Compare the INITIAL EXTRACTION to the TEXT. Identify any entity or relationship that is NOT explicitly written in the text or contradicts it. "
            "2. [OMISSION CHECK]: Scan the TEXT for missing facts. CRITICAL: You must ONLY look for missing info that fits EXACTLY into our allowed schema categories (e.g., Habitat, DietType, Family). IGNORE general trivia (like size, popularity, physical descriptions, or fun facts)."
        )
    )
    entities_to_delete: List[EntityPointer] = Field(
        default=[], 
        description="Entities from the initial extraction that MUST be deleted because they are NOT explicitly in the text."
    )
    relationships_to_delete: List[RelationshipPointer] = Field(
        default=[], 
        description="Relationships from the initial extraction that MUST be deleted because they are NOT explicitly in the text."
    )
    missing_entities_to_add: MissingEntities = Field(
        default_factory=MissingEntities, 
        description="NEW entities explicitly found in the text but missing from the initial extraction."
    )
    missing_relationships_to_add: MissingRelationships = Field(
        default_factory=MissingRelationships, 
        description="NEW relationships explicitly found in the text but missing from the initial extraction."
    )

# ==========================================
#   ENTITY EXTRACTOR CLASS
# ==========================================

class EntityExtractor:
    def __init__(self):
        self.client = GeminiClient()

    def extract_entities_and_relationships(
            self,
            text: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts entities and relationships using structured_output to prevent KeyErrors.
        """
        system_prompt = """You are an expert zoology Knowledge Graph extractor.
        Your goal is to extract animal species, their biometrics, their classifications, and specific relationships from the text provided.
        CRITICAL RULES (ZERO TOLERANCE FOR HALLUCINATIONS)
        1. STRICT GROUNDING: You must base your extraction on the source text. DO NOT use pre-trained knowledge, common sense, or assumptions.
        2. SCHEMA & ENUM COMPLIANCE: Strictly adhere to the provided schema descriptions and ENUM constraints.
        3. ENTITY CONSISTENCY: Ensure the source and target of each relationship exactly match the extracted entities.
        """
        prompt = f"Extract zoological entities and their relationships from the following text:\n\n{text}"

        extraction: GraphExtraction = self.client.structured_output(
            prompt=prompt,
            schema=GraphExtraction,
            system_prompt=system_prompt
        )

        nodes_dict = {
            "Species": [x.model_dump(mode='json', exclude_none=True) for x in extraction.species],
            "Family": [x.model_dump(mode='json') for x in extraction.families],
            "AnimalClass": [x.model_dump(mode='json') for x in extraction.animal_classes],
            "SkeletalStructure": [x.model_dump(mode='json') for x in extraction.skeletal_structures],
            "ReproductionMethod": [x.model_dump(mode='json') for x in extraction.reproduction_methods],
            "EnvironmentType": [x.model_dump(mode='json') for x in extraction.environment_types],
            "Habitat": [x.model_dump(mode='json') for x in extraction.habitats],
            "Location": [x.model_dump(mode='json') for x in extraction.locations],
            "ActivityCycle": [x.model_dump(mode='json') for x in extraction.activity_cycles],
            "SocialStructure": [x.model_dump(mode='json') for x in extraction.social_structures],
            "DietType": [x.model_dump(mode='json') for x in extraction.diet_types],
            "FoodSource": [x.model_dump(mode='json') for x in extraction.food_sources],
            "ConservationStatus": [x.model_dump(mode='json') for x in extraction.conservation_statuses],
        }
        
        rels_dict = {
            "MEMBER_OF_FAMILY": [x.model_dump(mode='json', exclude_none=True) for x in extraction.member_of_family_rels],
            "BELONGS_TO_CLASS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.belongs_to_class_rels],
            "HAS_SKELETAL_STRUCTURE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_skeletal_structure_rels],
            "REPRODUCES_VIA": [x.model_dump(mode='json', exclude_none=True) for x in extraction.reproduces_via_rels],
            "LIVES_IN_ENVIRONMENT": [x.model_dump(mode='json', exclude_none=True) for x in extraction.lives_in_environment_rels],
            "INHABITS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.inhabits_rels],
            "FOUND_IN": [x.model_dump(mode='json', exclude_none=True) for x in extraction.found_in_rels],
            "MIGRATES_TO": [x.model_dump(mode='json', exclude_none=True) for x in extraction.migrates_to_rels],
            "HAS_ACTIVITY_CYCLE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_activity_cycle_rels],
            "ORGANIZED_IN": [x.model_dump(mode='json', exclude_none=True) for x in extraction.organized_in_rels],
            "HAS_DIET_TYPE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_diet_type_rels],
            "PREYS_ON": [x.model_dump(mode='json', exclude_none=True) for x in extraction.preys_on_rels],
            "FEEDS_ON": [x.model_dump(mode='json', exclude_none=True) for x in extraction.feeds_on_rels],
            "HAS_CONSERVATION_STATUS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_conservation_status_rels],
        }

        num_entities = sum(len(nodes_dict[n]) for n in nodes_dict)
        num_relationships = sum(len(rels_dict[r]) for r in rels_dict)
        print(f"Initial extraction: {num_entities} entities and {num_relationships} relationships")

        return nodes_dict, rels_dict
    
    def _flatten_entities(self, entities: dict) -> str:
        """Convierte el diccionario de entidades a una lista plana de Markdown."""
        lines = []
        for category, items in entities.items():
            if not items:
                continue
            for item in items:
                # Extraemos el valor ya sea 'name' o 'type'
                val = item.get('name') or item.get('type')
                lines.append(f"- {category}: {val}")
        return "\n".join(lines) if lines else "No entities extracted."

    def _flatten_relationships(self, relationships: dict) -> str:
        """Convierte el diccionario de relaciones a una lista plana de Markdown."""
        lines = []
        for rel_type, items in relationships.items():
            if not items:
                continue
            for item in items:
                src = item.get('source')
                tgt = item.get('target')
                lines.append(f"- {src} [{rel_type}] {tgt}")
        return "\n".join(lines) if lines else "No relationships extracted."
        
    def review_extraction(
            self,
            text: str,
            entities: Dict[str, List[Dict[str, Any]]],
            relationships: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Review step to consolidate the entities and relationships extracted for a chunk.
        """

        initial_entities_json = self._flatten_entities(entities)
        initial_relations_json = self._flatten_relationships(relationships)

        system_prompt = """You are a ruthless Zoology Data Auditor. Your ONLY missions are to purge hallucinations and recover forgotten data from an initial extraction.

CRITICAL RULES:
1. REASON FIRST: You MUST use the 'thought_process' field to analyze the text before filling any lists. Structure your thoughts using [HALLUCINATION CHECK] and [OMISSION CHECK].
2. THE HALLUCINATION PURGE (DELETE): Every single entity and relationship in the INITIAL EXTRACTION must be explicitly backed by the text. If it relies on biological world knowledge or assumptions, YOU MUST DELETE IT.
3. BEWARE OF NEGATIONS & CONTRASTS: Pay extremely close attention to words like 'not', 'never', 'unlike', or 'while'. If the text states an animal DOES NOT do something, and the extraction says it does, DELETE IT.
4. SCHEMA-BOUND OMISSION (ADD): If a fact is clearly stated in the text but missing from the extraction, YOU CAN ONLY ADD IT IF IT FITS AN EXACT SCHEMA CATEGORY (e.g., 'Species', 'Habitat', 'Location'). 
5. ZERO DUPLICATES: Never add an omission if it (or a direct synonym) is already present in the INITIAL EXTRACTION. Check the list twice.
6. SCHEMA STRICTNESS: You must use the exact categories and relationship types provided in the schema."""

        brief_reasoning_system_prompt = """You are a ruthless Zoology Data Auditor. Your ONLY missions are to purge hallucinations and recover forgotten data from an initial extraction.

CRITICAL RULES:
1. REASON FIRST: You MUST use the 'thought_process' field to analyze the text before filling any lists. Structure your thoughts using [HALLUCINATION CHECK] and [OMISSION CHECK].
2. REASONING BREVITY: Limit your 'thought_process' to a MAXIMUM of 3 sentences for hallucinations and 3 sentences for omissions. Be direct, no fluff.
3. THE HALLUCINATION PURGE (DELETE): Every single entity and relationship in the INITIAL EXTRACTION must be explicitly backed by the text. If it relies on biological world knowledge or assumptions, YOU MUST DELETE IT.
4. BEWARE OF NEGATIONS & CONTRASTS: Pay extremely close attention to words like 'not', 'never', 'unlike', or 'while'. If the text states an animal DOES NOT do something, and the extraction says it does, DELETE IT.
5. SCHEMA-BOUND OMISSION (ADD): If a fact is clearly stated in the text but missing from the extraction, YOU CAN ONLY ADD IT IF IT FITS AN EXACT SCHEMA CATEGORY (e.g., 'Species', 'Habitat', 'Location'). 
6. ZERO DUPLICATES: Never add an omission if it (or a direct synonym) is already present in the INITIAL EXTRACTION. Check the list twice.
7. SCHEMA STRICTNESS: You must use the exact categories and relationship types provided in the schema."""


        prompt = f"""Execute your audit on the data below.

STEP 1: Perform the [HALLUCINATION CHECK]. Read the INITIAL EXTRACTION. If a data point is not explicitly in the TEXT, or contradicts it, mark it for deletion in your thought process.
STEP 2: Perform the [OMISSION CHECK]. Read the TEXT. Find facts missing from the extraction and mark them for addition in your thought process.
STEP 3: Generate the JSON patch using your thought process as a guide.

--- TEXT ---
{text}

--- INITIAL EXTRACTION ---
{initial_entities_json}
{initial_relations_json}"""

        try:
            extraction: GraphPatch = self.client.structured_output(
                prompt=prompt,
                schema=GraphPatch,
                system_prompt=system_prompt
            )
            
            if extraction is None:
                return entities, relationships

        except Exception as e:
            print(e)
            print("Reintentando revisión con un razonamiento más breve")
            try:
                extraction: GraphPatch = self.client.structured_output(
                    prompt=prompt,
                    schema=GraphPatch,
                    system_prompt=brief_reasoning_system_prompt
                )
                
                if extraction is None:
                    return entities, relationships
            except Exception as e:
                print(e)
                print("Error en la segunda revisión. Se añadirán las entidades y relaciones extraídas inicialmente.")
                return entities, relationships
        
        # ==========================================
        # 1. PROCESAR ELIMINACIONES (ALUCINACIONES)
        # ==========================================
        
        # Eliminar entidades
        for entity_ptr in extraction.entities_to_delete:
            category = entity_ptr.category  # Ej: "Species", "Habitat"
            identifier = entity_ptr.name_or_type
            
            if category in entities:
                entities[category] = [
                    e for e in entities[category] 
                    if e.get('name') != identifier and e.get('type') != identifier
                ]

        # Eliminar relaciones
        for rel_ptr in extraction.relationships_to_delete:
            rel_type = rel_ptr.relationship_type  # Ej: "INHABITS"
            src = rel_ptr.source
            tgt = rel_ptr.target
            
            if rel_type in relationships:
                relationships[rel_type] = [
                    r for r in relationships[rel_type] 
                    if not (r.get('source') == src and r.get('target') == tgt)
                ]

        # ==========================================
        # 2. PROCESAR ADICIONES (RECALL)
        # ==========================================
        
        new_ents = extraction.missing_entities_to_add
        new_rels = extraction.missing_relationships_to_add

        added_ents = 0
        added_rels = 0

        # Función auxiliar para añadir entidades sin duplicar
        def add_unique_entities(category: str, items: list, is_species: bool = False):
            id_key = 'name' if is_species else 'type'
            existing_ids = {e.get(id_key) for e in entities.get(category, [])}
            ents = 0
            
            for item in items:
                item_dict = item.model_dump(mode='json', exclude_none=True)
                item_id = item_dict.get(id_key)
                if item_id and item_id not in existing_ids:
                    entities[category].append(item_dict)
                    existing_ids.add(item_id)
                    ents += 1

            return ents

        added_ents = add_unique_entities("Species", new_ents.species, is_species=True)
        added_ents += add_unique_entities("Family", new_ents.families)
        added_ents += add_unique_entities("AnimalClass", new_ents.animal_classes)
        added_ents += add_unique_entities("SkeletalStructure", new_ents.skeletal_structures)
        added_ents += add_unique_entities("ReproductionMethod", new_ents.reproduction_methods)
        added_ents += add_unique_entities("EnvironmentType", new_ents.environment_types)
        added_ents += add_unique_entities("Habitat", new_ents.habitats)
        added_ents += add_unique_entities("Location", new_ents.locations)
        added_ents += add_unique_entities("ActivityCycle", new_ents.activity_cycles)
        added_ents += add_unique_entities("SocialStructure", new_ents.social_structures)
        added_ents += add_unique_entities("DietType", new_ents.diet_types)
        added_ents += add_unique_entities("FoodSource", new_ents.food_sources)
        added_ents += add_unique_entities("ConservationStatus", new_ents.conservation_statuses)

        # Función auxiliar para añadir relaciones sin duplicar
        def add_unique_relationships(category: str, items: list):
            existing_pairs = {(r.get('source'), r.get('target')) for r in relationships.get(category, [])}
            rels = 0
            for item in items:
                item_dict = item.model_dump(mode='json', exclude_none=True)
                pair = (item_dict.get('source'), item_dict.get('target'))
                if pair not in existing_pairs:
                    relationships[category].append(item_dict)
                    existing_pairs.add(pair)
                    rels += 1
            return rels

        added_rels += add_unique_relationships("MEMBER_OF_FAMILY", new_rels.member_of_family_rels)
        added_rels += add_unique_relationships("BELONGS_TO_CLASS", new_rels.belongs_to_class_rels)
        added_rels += add_unique_relationships("HAS_SKELETAL_STRUCTURE", new_rels.has_skeletal_structure_rels)
        added_rels += add_unique_relationships("REPRODUCES_VIA", new_rels.reproduces_via_rels)
        added_rels += add_unique_relationships("LIVES_IN_ENVIRONMENT", new_rels.lives_in_environment_rels)
        added_rels += add_unique_relationships("INHABITS", new_rels.inhabits_rels)
        added_rels += add_unique_relationships("FOUND_IN", new_rels.found_in_rels)
        added_rels += add_unique_relationships("MIGRATES_TO", new_rels.migrates_to_rels)
        added_rels += add_unique_relationships("HAS_ACTIVITY_CYCLE", new_rels.has_activity_cycle_rels)
        added_rels += add_unique_relationships("ORGANIZED_IN", new_rels.organized_in_rels)
        added_rels += add_unique_relationships("HAS_DIET_TYPE", new_rels.has_diet_type_rels)
        added_rels += add_unique_relationships("PREYS_ON", new_rels.preys_on_rels)
        added_rels += add_unique_relationships("FEEDS_ON", new_rels.feeds_on_rels)
        added_rels += add_unique_relationships("HAS_CONSERVATION_STATUS", new_rels.has_conservation_status_rels)

        # ==========================================
        # 3. CONTEO Y LOGS
        # ==========================================
        
        final_num_entities = sum(len(entities[n]) for n in entities)
        final_num_relationships = sum(len(relationships[r]) for r in relationships)
        
        deleted_ents = len(extraction.entities_to_delete)
        deleted_rels = len(extraction.relationships_to_delete)
        
        print(f"Audit completed:")
        print(f" - Entities: Deleted {deleted_ents} | Added {added_ents} -> Final: {final_num_entities}")
        print(f" - Relationships: Deleted {deleted_rels} | Added {added_rels} -> Final: {final_num_relationships}")
        
        return entities, relationships


    def summarize_entity(self, entity_name: str, descriptions: List[str]) -> str:
        """Consolidate descriptions (Entity Resolution step)."""
        system_prompt = "You are a specialist in zoological knowledge synthesis. Create a single third-person summary combining these biological descriptions."

        description_list = "\n- ".join(descriptions)
        user_message = f"Entity: {entity_name}\n\nDescriptions to merge:\n- {description_list}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.client.chat(messages)

    def summarize_relationship(self, source: str, target: str, descriptions: List[str]) -> str:
        """Consolidate relationships."""
        prompt = f"Summarize the ecological or biological relationship between {source} and {target} based on: {descriptions}"
        return self.client.chat([{"role": "user", "content": prompt}])