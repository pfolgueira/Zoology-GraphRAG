from typing import List, Dict, Any
import concurrent.futures

from ..llm.groq_client import GroqClient
from ..llm.gemini_client import GeminiClient
from ..llm.ollama_client import OllamaClient
from typing import Literal
from pydantic import BaseModel, Field
from graphrag.retrieval.vector_retriever import VectorRetriever
from ..retrieval.text2cypher import Text2CypherRetriever
from graphrag.graph.neo4j_manager import Neo4jManager

class PlanStep(BaseModel):
    step_id: str = Field(
        ..., 
        description="Step identifier, e.g., 'Step 1', 'Step 2', etc. This is for reference and should be sequential."
    )
    tool: Literal["text2cypher", "semantic_search"] = Field(
        ..., 
        description="The exact tool to use in this step."
    )
    query: str = Field(
        ..., 
        description="The natural language query optimized for the selected tool."
    )

class ExecutionPlan(BaseModel):
    steps: List[PlanStep] = Field(
        ..., 
        description="Sequential list of steps to execute."
    )

class MultiHopPlanner:
    def __init__(self, neo4j_manager: Neo4jManager, text2cypher_retriever: Text2CypherRetriever):
        self.neo4j = neo4j_manager
        self.text2cypher = text2cypher_retriever
        self.client = GroqClient()
        self.vector_retriever = VectorRetriever(neo4j_manager)
        self.species_with_chunks = self._load_species_with_chunks()
        self.species_ranking = self._load_species_ranking()

    def _load_species_with_chunks(self) -> set:
        """Loads and caches all species names from the DB that have an associated chunk."""
        try:
            query = """
            MATCH (c:Chunk)-[:HAS_ENTITY]->(s:Species)
            RETURN DISTINCT s.name AS species_name
            """
            result = self.neo4j.execute_query(query)
            # Use set comprehension to create and return a set directly
            return {r["species_name"] for r in result}
        except Exception as e:
            print(f"Warning: could not load species list: {e}")
            # Return an empty set if the query fails
            return set()
        
    def _load_species_ranking(self) -> dict:
        """Loads and caches the topological degree (number of relationships) of each species."""
        try:
            # Contamos todas las relaciones -[r]- conectadas a cada especie
            query = """
            MATCH (s:Species)-[r]-()
            WITH s.name AS species_name, count(r) AS degree
            RETURN species_name, degree
            """
            result = self.neo4j.execute_query(query)
            
            # Dictionary comprehension para acceso ultra-rápido O(1)
            return {r["species_name"]: r["degree"] for r in result}
        except Exception as e:
            print(f"Warning: could not load species ranking: {e}")
            return {}

    def _get_tool_descriptions(self) -> str:

        tools = [
            {
                "name": "semantic_search",
                "description": "Use this tool to search unstructured text documents. "
                               "It performs a pure semantic vector search to find relevant context based on meaning. "
                               "PRIORITIZE THIS TOOL when the user asks for general information, broad descriptions, explanations, or curiosities about animals (e.g., 'describe the habitat of...', "
                               "'explain how X hunts'). DO NOT use this tool for counting, aggregations, or exact property filtering."
                               ,
                "parameters": {
                    "query": "The search query in natural language, optimized for document retrieval."
                }
            },
            {
                "name": "text2cypher",
                "description": "Use this tool to query the structured knowledge graph directly. "
                            "PRIORITIZE THIS TOOL when the query requires precise data points, structured relationship traversals, exact property matching, or aggregations "
                            "(e.g., 'how many animals...', 'list all species in the family X', 'what is the exact diet of Y'). "
                            "DO NOT use this tool for requesting long-form text, general explanations, or descriptive paragraphs. "
                            ,
                "parameters": {
                    "query": "The user's EXACT question in natural language."
                }
            }
        ]

        def format_tool(tool: dict) -> str:
            lines = [f"- {tool['name']}: {tool['description']}"]
            params = tool.get("parameters", {})
            if params:
                lines.append("  Parameters:")
                for param_name, param_desc in params.items():
                    lines.append(f"    - {param_name}: {param_desc}")
            return "\n".join(lines)

        # Filtramos y formateamos solo las que NO están en el set de exclusión
        tools_str = "\n".join(
            format_tool(tool) 
            for tool in tools
        )

        return tools_str

    def plan_and_execute(self, question: str) -> Dict[str, Any]:
        """
        Planifica las herramientas a utilizar para el razonamiento multi paso.
        """

        tools_str = self._get_tool_descriptions()


        planner_system_prompt = f"""You are an Expert Query Planner for an advanced Zoology and Animal Biology Graph RAG system.
Your job is to break down complex, multi-hop user questions into a strict, sequential list of tool executions.

You do NOT execute the plan. You only create the logical steps. Assume that the execution engine will automatically pass the results of one step as the context or filter for the next step.

Available tools and their descriptions:
{tools_str}

PLANNING RULES & HEURISTICS:
- ALWAYS narrow down the search space first. Use "text2cypher" to get the specific list of entities before asking for their descriptive traits.
- Keep plans concise. Most complex queries can be solved in 2 sequential steps.
- The order is crucial: the execution engine will run Step 1, take its output, and use it to filter Step 2, and so on.

EXAMPLES OF LOGICAL FLOWS:
- User: "Of the species that live in the Savannah, what are their different survival strategies against drought?"
  Flow: 
  Step 1 (text2cypher): "List all animal species that inhabit the Savannah."
  Step 2 (semantic_search): "survival strategies against drought"

- User: "Compare the diets of felines that live in Africa vs those in Asia."
  Flow:
  Step 1 (text2cypher): "List all feline species that can be found in Africa."
  Step 2 (text2cypher): "List all feline species that can be found in Asia."
  Step 2 (semantic_search): "diet and eating habits"
"""

        messages = [
            {"role": "system", "content": planner_system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]

        plan = self.client.structured_output_with_chat(messages, schema=ExecutionPlan, temperature=0.0)

        accumulated_entities = set()
        combined_context = []
        combined_cypher = []
        tools_used = []

        for step in plan.steps:
            tools_used.append(step.tool)

            if step.tool == "text2cypher":
                # 1. Ejecutar consulta en el grafo (¡Desempaquetando la tupla directamente!)
                # Revisa si tu método espera 'query=' o 'question=' como parámetro
                cypher_query, graph_data = self.text2cypher.retrieve(question=step.query)
                
                # Por seguridad, si graph_data es None, lo convertimos a lista vacía
                graph_data = graph_data or []

                # 2. Guardar contexto estructurado para el LLM sintetizador
                combined_context.append({
                    "source": f"Graph query: {step.query}",
                    "data": graph_data
                })

                # Guardar el cypher si se generó
                if cypher_query:
                    combined_cypher.append(cypher_query)

                # 3. Extraer entidades para futuro filtrado
                for row in graph_data:
                    for key, value in row.items():
                        if isinstance(value, (str, int)):
                            accumulated_entities.add(str(value))
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, (str, int)):
                                    accumulated_entities.add(str(item))

            elif step.tool == "semantic_search":
                # Fast filtering: Intersection of accumulated entities and species with chunks
                valid_entities = list(accumulated_entities.intersection(self.species_with_chunks))

                if not valid_entities:
                    continue # Skip this step if no valid entities exist

                # Deterministic topological truncation (Top 10 most connected species)
                entities_to_search = sorted(
                    valid_entities, 
                    key=lambda entity: self.species_ranking.get(entity, 0), 
                    reverse=True
                )[:10]

                query_embedding = self.vector_retriever.embedding_gen.embed_text(step.query)
                
                try:
                    search_results = self.vector_retriever.retrieve_filtered_by_entities(
                        query_embedding=query_embedding,
                        entities_names=entities_to_search,
                        top_k=1
                    )
                    
                    if search_results:
                        combined_context.extend(search_results)
                        
                except Exception as e:
                    print(f"Error executing batch hybrid search for entities: {e}")
                    raise e

        # Return the unified payload 
        return {
            "tools": tools_used,
            "results": [], 
            "context": combined_context,
            "cypher": combined_cypher or None,
            "routing_decision": "complex_query_execution"
        }