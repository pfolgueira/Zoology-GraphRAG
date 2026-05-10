from typing import List, Dict, Any, Callable, TypedDict

from ..retrieval.manual_retriever import ManualRetriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.text2cypher import Text2CypherRetriever
from ..graph.neo4j_manager import Neo4jManager

import random

# Definimos la estructura de la herramienta
class ToolData(TypedDict):
    function: Callable
    description: str

class RetrieverTools:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.hybrid_retriever = HybridRetriever(neo4j_manager)
        self.text2cypher = Text2CypherRetriever(neo4j_manager)
        self.manual_retriever = ManualRetriever(neo4j_manager)
        self.custom_tools: Dict[str, ToolData] = {}

    def register_custom_tool(self, name: str, function: Callable, description: str):
        """Registra una herramienta personalizada."""
        self.custom_tools[name] = {
            "function": function,
            "description": description
        }

    def _format_graph_context(self, tool_name: str, query: str, results: List[Dict]) -> List[str]:
        header = f"[{tool_name} - {query}]"
        rows = "\n".join(str(r) for r in results)
        return [f"{header}\n{rows}"]

    _GREETING_RESPONSES = [
        # 🦉 El Búho Sabiondo
        ("Hoot hoot! 🦉 I am a highly intellectual knowledge assistant perched at the top of the animal kingdom. "
        "You can ask me about our taxonomy, physical or behavioral traits, diet (preferably mice 🐁), habitats, and "
        "conservation status."),
        
        # 🐕 El Perro Hiperactivo
        ("WOOF! Hi! Hello! 🐕 I am a VERY GOOD knowledge assistant specialized in the animal kingdom! 🎾 "
        "You can throw questions at me about our taxonomy, behavioral traits (like fetching!), diet (SQUIRREL?! 🐿️), habitats, and "
        "conservation status! 🐾"),
        
        # 🐈‍⬛ El Gato Arrogante
        ("Meow. 🐈‍⬛ I am your supreme feline overlord, but for now, I'll act as a knowledge assistant for the animal kingdom. "
        "You may grovel and ask me about taxonomy, physical traits, my preferred premium diet 🐟, habitats (mainly cardboard boxes 📦), and "
        "conservation status."),
        
        # 🦥 El Perezoso Relajado
        ("Yawn... Helloooo... 🦥 I am a knowledge assistant... eventually... specialized in the animal kingdom. "
        "Take your time asking me about our taxonomy, traits, diet (mostly leaves 🍃), habitats, and "
        "conservation status... no rush... 💤"),
        
        # 🦦 El Capibara "Chill"
        ("Sup. 🦦 I am the chillest knowledge assistant in the animal kingdom. "
        "Grab a spot in the hot spring ♨️ and ask me about our taxonomy, physical or behavioral traits, diet, habitats, and "
        "conservation status. We're all friends here. ✌️"),
        
        # 🐧 El Pingüino Formal
        ("Greetings! 🐧 Please excuse my tuxedo, I am a very formal knowledge assistant specialized in the animal kingdom. 🧊 "
        "You may inquire about our taxonomy, physical or behavioral traits, diet (strictly seafood 🦑), freezing habitats, and "
        "conservation status."),
        
        # 🐬 El Delfín Entusiasta
        ("Eee-eee! *Splash* 🐬 I am a super-smart, echolocating knowledge assistant riding the waves of the animal kingdom! 🌊 "
        "You can click and squeak at me about our taxonomy, behavioral traits, diet, marine habitats, and "
        "conservation status!"),
        
        # 🦎 El Camaleón Camuflado
        ("Hello... 🦎 Now you see me, now you don't! I am a highly adaptable knowledge assistant blending into the animal kingdom. 🌿 "
        "You can ask me to reveal facts about taxonomy, physical traits (like my fabulous colors ✨), diet (mostly crunchy bugs 🦗), habitats, and "
        "conservation status."),
        
        # 🐝 La Abeja Adicta al Trabajo
        ("Bzzzz! Welcome to the hive! 🐝 I am a very busy worker-bee knowledge assistant pollinating the animal kingdom. 🌸 "
        "Don't sting me with hard questions, but you can ask about taxonomy, behavioral traits, diet (sweet, sweet nectar 🍯), habitats, and "
        "conservation status!")
    ]

    _OUT_OF_SCOPE_RESPONSE = (
        "Umm... moo? 🐄 Sorry, this question is way outside my pasture. "
        "I am designed to chew the cud exclusively about zoology and animal biology. "
        "If you have a question about the animal world, I am ready to help! Otherwise, I'm just going back to eating grass. 🌱"
    )
    
    _SKILLS_RESPONSE = (
        "With my eight arms and three brains, I can juggle detailed questions about specific animals! 🐙 "
        "You can ask me to dive deep into their taxonomic classification (family, genus, species), their physical characteristics, "
        "what they eat, where they live, or their conservation status. Just don't ask me to untangle my tentacles! 🌊"
    )

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Obtiene las descripciones de todas las herramientas disponibles."""
        tools = [
            {
                "name": "greeting",
                "description": (
                    "Handle conversational messages that need no knowledge lookup: "
                    "greetings, farewells, or basic thanks. Do not use for questions about capabilities."
                ),
                "parameters": {}
            },
            {
                "name": "out_of_scope",
                "description": (
                    "Handle questions clearly unrelated to zoology, specific animal species, taxonomy, "
                    "or animal traits (e.g., politics, coding, sports, weather, etc.). "
                    "Questions about specific countries or regions are IN-SCOPE only if asking about "
                    "the animals that live there."
                ),
                "parameters": {}
            },
            {
                "name": "skills",
                "description": (
                    "Describe the system's capabilities. Use this when the user asks what you can do, "
                    "or what kind of questions you can answer."
                ),
                "parameters": {}
            },
            {
                "name": "hybrid_search",
                "description": "Use this tool to search unstructured text documents. "
                            "It performs a combined semantic and keyword search with reranking. "
                            "PRIORITIZE THIS TOOL when the user asks for general information, broad descriptions, explanations, or curiosities about animals "
                            "(e.g., 'describe the habitat of...', 'explain how X hunts'). "
                            "DO NOT use this tool for counting, aggregations, or exact property filtering.",
                "parameters": {
                    "query": "The search query in natural language, optimized for document retrieval."
                }
            }, 
            {
                "name": "text2cypher",
                "description": "Use this tool to query the structured knowledge graph directly. "
                            "PRIORITIZE THIS TOOL when the query requires precise data points, structured relationship traversals, exact property matching, or aggregations "
                            "(e.g., 'how many animals...', 'list all species in the family X', 'what is the exact diet of Y'). "
                            "DO NOT use this tool for requesting long-form text, general explanations, or descriptive paragraphs.",
                "parameters": {
                    "query": "The user's EXACT question in natural language."
                }
            },
            {
                "name": "predefined_species_full_profile",
                "description": (
                    "Retrieves a complete structured profile of a specific animal species from the knowledge graph. "
                    "Use this tool when the user asks for a full summary or complete overview of a species "
                    "(e.g., 'tell me everything about the lion', 'give me a full profile of the tiger', "
                    "'complete summary of the African elephant'). "
                    "DO NOT use this tool for partial queries, comparisons, aggregations, or general habitat/diet questions."
                ),
                "parameters": {
                    "query": "The user's EXACT question in natural language.",
                    "species_name": "The name of the animal species to retrieve the full profile for (e.g., 'Lion', 'Tiger')."
                }
            },
            {
                "name": "predefined_endangered_by_environment",
                "description": (
                    "Retrieves endangered or critically endangered species filtered by their environment type. "
                    "Use this tool when the user asks about threatened or endangered animals in a specific environment or habitat type. "
                    "Examples: 'Which endangered species are flying animals?', 'What marine animals are critically endangered?', "
                    "'List threatened species in terrestrial environments', 'Are there endangered flying animals?'. "
                    "DO NOT use for general habitat questions or non-conservation queries."
                ),
                "parameters": {
                    "query": "The user's EXACT question in natural language.",
                    "environment_name": (
                        "The environment type normalized to one of these exact values: "
                        "'Aquatic', "
                        "'Terrestrial', "
                        "'Aerial'"
                        "You MUST return one of these three values exactly, never a synonym."
                    )
                }
            },
            {
                "name": "predefined_predator_prey_chain",
                "description": (
                    "Retrieves the complete feeding relationships of a specific species: "
                    "what predators hunt it, what animal prey it hunts, and what non-animal food sources it feeds on. "
                    "Use this tool when the user asks about the full predator-prey chain or feeding relationships of a species. "
                    "Examples: 'What hunts the lion and what does it hunt?', 'Show me the predator-prey chain of the wolf', "
                    "'What are the feeding relationships of the zebra?'. "
                    "DO NOT use if the user only asks one direction (use text2cypher instead)."
                ),
                "parameters": {
                    "query": "The user's EXACT question in natural language.",
                    "species_name": "The name of the animal species to retrieve the predator-prey chain for (e.g., 'Lion', 'Zebra')."
                }
            },
            {
                "name": "predefined_social_structure_by_class",
                "description": (
                    "Retrieves the distribution of social structures among species of a specific animal class. "
                    "Use this tool when the user asks about how animals of a given class are socially organized. "
                    "Examples: 'How are mammals organized socially?', 'What social structures do reptiles have?', "
                    "'Show me the social organization of birds'. "
                    "DO NOT use for questions about a specific species or general social behavior descriptions."
                ),
                "parameters": {
                    "query": "The user's EXACT question in natural language.",
                    "class_name": (
                        "The animal class normalized to one of these exact values: "
                        "'Mammal', 'Reptile', 'Bird', 'Fish', 'Amphibian'. "
                        "You MUST return one of these five values exactly, never a synonym or plural form."
                    )
                }
            }       
        ]

        # Añadir herramientas personalizadas
        for name, tool_info in self.custom_tools.items():
            tools.append({
                "name": name,
                "description": tool_info["description"],
                "parameters": {}
            })

        return tools

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Ejecuta una herramienta específica."""
        # ====================================================================
        # HERRAMIENTAS QUE NO REQUIEREN BÚSQUEDA DE CONOCIMIENTO EN EL SISTEMA
        # ====================================================================
        if tool_name == "greeting":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": random.choice(self._GREETING_RESPONSES),
            }

        if tool_name == "out_of_scope":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": self._OUT_OF_SCOPE_RESPONSE,
            }

        if tool_name == "skills":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": self._SKILLS_RESPONSE,
            }
        if tool_name == "predefined_species_full_profile":
            cypher, results = self.manual_retriever.retrieve(
                query_category="species_full_profile",  # fijo, lo determina el tool_name
                species_name=kwargs.get("species_name", "").title(),
            )
            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": self._format_graph_context(tool_name, kwargs.get("query", ""), results)
            }
        
        if tool_name == "predefined_endangered_by_environment":
            cypher, results = self.manual_retriever.retrieve(
                query_category="endangered_by_environment",  # fijo, lo determina el tool_name
                environment_name=kwargs.get("environment_name", "").title(),
            )
            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": self._format_graph_context(tool_name, kwargs.get("query", ""), results)
            }
        
        if tool_name == "predefined_predator_prey_chain":
            cypher, results = self.manual_retriever.retrieve(
                query_category="predator_prey_chain",
                species_name=kwargs.get("species_name", "").title(),
            )
            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": self._format_graph_context(tool_name, kwargs.get("query", ""), results)
            }
                
        if tool_name == "predefined_social_structure_by_class":
            cypher, results = self.manual_retriever.retrieve(
                query_category="social_structure_by_class",  # fijo, lo determina el tool_name
                class_name=kwargs.get("class_name", "").title(),
            )
            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": self._format_graph_context(tool_name, kwargs.get("query", ""), results)
            }

        elif tool_name == "hybrid_search":
            results = self.hybrid_retriever.retrieve(kwargs.get("query", ""))
            return {
                "tool": tool_name,
                "results": results,
                "context": [r["text"] for r in results]
            }

        elif tool_name == "text2cypher":
            cypher, results = self.text2cypher.retrieve(kwargs.get("query", ""))

            # Fallback a hybrid_search si text2cypher no devuelve resultados
            if not results:
                print(f"text2cypher returned no results, falling back to hybrid_search.")
                fallback_results = self.hybrid_retriever.retrieve(kwargs.get("query", ""))
                return {
                    "tool": f"{tool_name}+hybrid_search_fallback",
                    "cypher": cypher,
                    "results": fallback_results,
                    "context": [r["text"] for r in fallback_results]
                }

            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": self._format_graph_context(tool_name, kwargs.get("query", ""), results)
            }

        elif tool_name in self.custom_tools:
            function = self.custom_tools[tool_name]["function"]
            results = function(**kwargs)
            return {
                "tool": tool_name,
                "results": results,
                "context": results if isinstance(results, list) else [str(results)]
            }

        else:
            raise ValueError(f"Unknown tool: {tool_name}")