from typing import List, Dict, Any, Optional

from ..llm.groq_client import GroqClient
from ..llm.ollama_client import OllamaClient
from .retriever_tools import RetrieverTools
from typing import Literal
from pydantic import BaseModel, Field, model_validator

TOOL_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "hybrid_search":                  ["query"],
    "text2cypher":                    ["query"],
    "predefined_species_full_profile": ["species_name"],
    "predefined_endangered_by_environment": ["environment_name"],
    "predefined_predator_prey_chain": ["species_name"],
    "predefined_social_structure_by_class": ["class_name"],
    "greeting":                       [],
    "out_of_scope":                   [],
    "skills":                         [],
}

class ToolCall(BaseModel):
    """Representa una llamada a una herramienta con sus parámetros."""
    tool: Literal[
        "predefined_species_full_profile",
        "predefined_endangered_by_environment",
        "predefined_predator_prey_chain",
        "predefined_social_structure_by_class",
        "hybrid_search",
        "text2cypher",
        "greeting",
        "out_of_scope",
        "skills",
    ]
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters for this specific tool. "
            "For hybrid_search and text2cypher: {'query': '...'}. "
            "For predefined queries: { parameter name: '...'}. "
            "For greeting, out_of_scope, skills: leave empty {}."
        )
    )

    @model_validator(mode="after")
    def validate_parameters(self) -> "ToolCall":
        required = TOOL_REQUIRED_PARAMS.get(self.tool, [])
        missing = [p for p in required if not self.parameters.get(p)]
        if missing:
            raise ValueError(f"Tool '{self.tool}' is missing required parameters: {missing}")
        return self

class RouterDecision(BaseModel):
    """Schema for the LLM's routing decision."""

    tool_calls: List[ToolCall] = Field(
        ...,
        description=(
            "List of tool calls to execute, each with their own parameters. "
            "Use a single tool for most queries. "
            "Combine retrievers when the query requires both structured data and descriptive information. "
            "NEVER combine greeting, out_of_scope, or skills with other tools."
        )
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why these tools were chosen."
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
Your job is to analyze the user's query and select the most appropriate tool or combination of tools to handle it.

Available tools and their descriptions:
{tools_str}

ROUTING RULES — Apply strictly in the following order:

1. greeting
— Use when the message is conversational and needs NO knowledge lookup:
- Examples: "Hello", "Hi", "Good morning", "Thanks", "Goodbye", "Who are you?".
- Do NOT use this if the user asks about what the system can do.
- NEVER combine with other tools.

2. skills
- Use ONLY when the user explicitly asks about the system's features, topics, or how to use it.
- Examples: "What can you do?", "What kind of animal questions can I ask?", "How do you work?".
- NEVER combine with other tools.

3. out_of_scope
- Use when the question is clearly unrelated to zoology, animal biology, or the natural world.
- Examples: "What is the capital of France?", "Who won the World Cup?", "Give me a recipe for cake."
- NEVER combine with other tools.

4. predefined queries
- Use these tools when the query matches one of the available predefined query scenarios.
- ALWAYS prioritize these over text2cypher or hybrid_search when the intent clearly matches.
- Each predefined tool has its own description and required parameters detailed in the tools list above.

5. text2cypher
- Use for extracting specific factual data, structured relationships, and complex analytical questions.
- Good for: "What do X eat/hunt?", "Where do X live?", "What is the speed/weight/lifespan of X?", "How many...", "Which animals...", "List all...".
- Examples: "What animals do wolves hunt?", "How many species of mammals live in the Amazon?".
- Can be combined with hybrid_search when the query needs both structured data AND descriptive explanation.

6. hybrid_search
- Use for qualitative questions, conceptual explanations, or broad semantic searches over unstructured text.
- Good for: "How", "Why", "Describe", or general biological concepts.
- Examples: "How does a chameleon change its color?", "Why do birds migrate?".
- Can be combined with text2cypher or a predefined cypher tool when the query needs both descriptive and structured information.

COMBINATION RULES:
- Use a SINGLE tool for most queries.
- Combine hybrid_search + text2cypher when the query needs both a descriptive explanation AND specific structured data (e.g., "Describe how lions hunt and list all their prey").
- Combine hybrid_search + predefined cypher query when the query matches a predefined scenario but also asks for a descriptive explanation.
NEVER combine greeting, out_of_scope, or skills with any other tool, even if the query mixes topics:
    * "Describe how lions hunt and how are you?" → greeting only
    * "Describe how lions hunt and who won the last World Cup?" → out_of_scope only
    * "What can you do and what do lions eat?" → skills only
  In these cases, always prioritize the non-retrieval tool and discard the rest of the query.

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
        Selecciona las herramientas apropiadas y ejecuta las búsquedas.
        """
        decision = self.route(question, conversation_history)

        # Herramientas directas: no requieren búsqueda de conocimiento
        direct_tools = {"greeting", "out_of_scope", "skills"}
        if decision.tool_calls[0].tool in direct_tools:
            tc = decision.tool_calls[0]
            result = self.tools.execute_tool(tc.tool, **tc.parameters)
            result["routing_decision"] = decision
            return result

        # Ejecutar cada herramienta y fusionar resultados
        combined_results = []
        combined_context = []
        combined_cypher = []

        for tc in decision.tool_calls:
            result = self.tools.execute_tool(tc.tool, **tc.parameters)
            combined_results.extend(result.get("results", []))
            combined_context.extend([
                f"[{tc.tool}] {c}" for c in result.get("context", [])
            ])
            if "cypher" in result:
                combined_cypher.append(result["cypher"])
        return {
            "tools": [tc.tool for tc in decision.tool_calls],
            "results": combined_results,
            "context": combined_context,
            "cypher": combined_cypher or None,
            "routing_decision": decision,
        }
