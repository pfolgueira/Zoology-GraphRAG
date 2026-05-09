from typing import List, Dict, Any, Optional

from ..llm.groq_client import GroqClient
from ..llm.ollama_client import OllamaClient
from .retriever_tools import RetrieverTools
from typing import Literal
from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    """Schema for the LLM's routing decision."""

    tool: Literal[
        "predefined_species_full_profile",
        "hybrid_search",
        "text2cypher",
        "greeting",
        "out_of_scope",
        "skills",
    ] = Field(
        ...,
        description="The name of the tool selected to handle the query."
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this tool was chosen based on the question."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool-specific parameters as key-value pairs. "
            "For hybrid_search and text2cypher: {'query': '...'}. "
            "For predefined_species_full_profile: {'species_name': '...'}. "
            "For greeting, out_of_scope, skills: leave empty {}."
        )
    )

class RetrieverRouter:
    def __init__(self, retriever_tools: RetrieverTools):
        self.tools = retriever_tools
        self.client = GroqClient()

    def route(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> RouterDecision:
        """
        Selecciona la mejor herramienta para responder la pregunta.
        """
        conversation_history = conversation_history or []
        
        
        def format_tool(tool: dict) -> str:
            lines = [f"- {tool['name']}: {tool['description']}"]
            params = tool.get("parameters", {})
            if params:
                lines.append("  Parameters:")
                for param_name, param_desc in params.items():
                    lines.append(f"    - {param_name}: {param_desc}")
            return "\n".join(lines)

        tool_descriptions = self.tools.get_tool_descriptions()
        tools_str = "\n".join(format_tool(tool) for tool in tool_descriptions)

        system_prompt = f"""You are an expert routing assistant for a zoology and animal biology knowledge base.
Your only job is to analyze the user's query and select the single most appropriate tool to handle it.

Available tools and their descriptions:
{tools_str}

ROUTING RULES — Apply strictly in the following order:

1. greeting — Use when the message is conversational and needs NO knowledge lookup:
- Examples: "Hello", "Hi", "Good morning", "Thanks", "Goodbye", "Who are you?".
- Do NOT use this if the user asks about what the system can do.

2. skills
- Use ONLY when the user explicitly asks about the system's features, topics, or how to use it.
- Examples: "What can you do?", "What kind of animal questions can I ask?", "How do you work?".

3. out_of_scope
- Use when the question is clearly unrelated to zoology, animal biology, or the natural world.
- Examples: "What is the capital of France?", "Who won the World Cup?", "Give me a recipe for cake."

4. predefined cypher queries
- Use when the query is a common, direct question about animal relationships that matches one of the predefined Cypher queries.
- Examples: "Give me all the info about lions"

5. text2cypher
- Use for complex analytical questions requiring aggregations, counts, or multi-hop relationship traversals in a graph database.
- Good for: "How many", "Which species share...", "List all..."
- Examples: "How many species of mammals live in the Amazon?", "List all predators of the African elephant that also live in savannas.", "Which species are both predators and prey?"

6. hybrid_search
- Use for qualitative questions, conceptual explanations, or broad semantic searches over unstructured text documents.
- Good for: "How", "Why", "Describe", or general biological concepts.
- Examples: "How does a chameleon change its color?", "Describe the mating dance of the albatross", "Why do birds migrate?"

OUTPUT FORMAT:
Respond ONLY with a JSON object matching the schema, without any additional text or explanation.
"""

        messages = [
                       {"role": "system", "content": system_prompt}
                   ] + conversation_history + [
                       {"role": "user", "content": f"Question: {question}"}
                   ]

        return self.client.structured_output_with_chat(messages, schema=RouterDecision)

    def retrieve(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Selecciona la herramienta apropiada y ejecuta la búsqueda.
        """
        decision = self.route(question, conversation_history)

        tool_name = decision.tool
        params = decision.parameters

        result = self.tools.execute_tool(tool_name, **params)
        result["routing_decision"] = decision

        return result
