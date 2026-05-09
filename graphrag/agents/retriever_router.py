from typing import List, Dict, Any, Optional
from ..llm.ollama_client import OllamaClient
from .retriever_tools import RetrieverTools
from typing import Literal
from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    """Schema for the LLM's routing decision."""

    tool: Literal[
        "predefined_cypher", "hybrid_search", "text2cypher",
        "greeting", "out_of_scope", "skills"
    ] = Field(
        ...,
        description="The name of the tool selected to handle the query."
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this tool was chosen based on the question."
    )
    query: str = Field(
        ...,
        description="The reformulated or original query to be passed to the tool."
    )


class RetrieverRouter:
    def __init__(self, retriever_tools: RetrieverTools):
        self.tools = retriever_tools
        self.client = OllamaClient()

    def route(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> RouterDecision:
        """
        Selecciona la mejor herramienta para responder la pregunta.
        """
        conversation_history = conversation_history or []

        # Obtener descripciones de herramientas
        tool_descriptions = self.tools.get_tool_descriptions()
        tools_str = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tool_descriptions
        ])

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

4. predefined_cypher
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
        query = decision.query or question

        result = self.tools.execute_tool(tool_name, query=query)
        result["routing_decision"] = decision

        return result
