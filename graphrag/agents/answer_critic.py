from typing import List, Dict, Any
from pydantic import BaseModel, Field
from ..llm.gemini_client import GeminiClient
from ..llm.ollama_client import OllamaClient
from ..llm.groq_client import GroqClient

class AnswerCritique(BaseModel):
    """Pydantic schema representing the evaluation of an answer."""
    is_complete: bool = Field(
        description="True if the answer fully addresses all aspects of the user's question. False otherwise."
    )
    is_faithful: bool = Field(
        description="True if all claims in the answer are strictly supported by the provided context (no hallucinations). False otherwise."
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="If incomplete, a list of specific search queries or questions needed to find the missing information. Empty if complete."
    )
    feedback: str = Field(
        description="A concise explanation justifying the evaluation, pointing out specific flaws or missing details."
    )

class AnswerCritic:
    def __init__(self):
        self.client = GeminiClient()

    def critique(self, question: str, context: List[str], answer: str) -> Dict[str, Any]:
        """
        Evalúa si la respuesta es completa y correcta.

        Returns:
            Dict con:
            - is_complete: bool
            - is_faithful: bool
            - missing_info: List[str] (preguntas adicionales si es necesario)
            - feedback: str
        """
        context_str = "\n\n".join([f"[{i + 1}] {c}" for i, c in enumerate(context)])

        system_prompt = """You are an expert at evaluating answers to questions based on provided context.

Your task is to determine:
1. Is the answer complete? (Does it fully address all parts of the question?)
2. Is the answer faithful? (Is it supported by the provided context?)
3. What information is missing, if any?

Respond ONLY with valid JSON in this format:
{
    "is_complete": true/false,
    "is_faithful": true/false,
    "missing_info": ["additional question 1", "additional question 2"],
    "feedback": "brief explanation"
}

If the answer is complete and faithful, missing_info should be an empty list."""

        system_prompt = f"""You are an expert at evaluating answers to questions based on provided context. Your job is to evaluate a generated answer against a user's question and the provided context.

### EVALUATION CRITERIA:
1. FAITHFULNESS (is_faithful): 
   - Check every single claim made in the answer.
   - If the answer contains ANY information, facts, or numbers not explicitly stated in the context, it is UNFAITHFUL (False).
   - Deductions or logical leaps outside the context text are strictly forbidden.

2. COMPLETENESS (is_complete):
   - Does the answer address every part of the user's question?
   - If the question has multiple parts and the answer misses one, it is INCOMPLETE (False).

3. MISSING INFO (missing_info):
   - If the answer is incomplete, what exactly is missing? 
   - Formulate these as clear search queries or follow-up questions that a retrieval system could use to find the missing data.

### OUTPUT FORMAT:
Respond ONLY with the required JSON schema.
"""

        user_message = f"""Question: {question}

Context:
{context_str}

Answer: {answer}

Evaluate this answer and output the JSON."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self.client.structured_output_with_chat(messages, schema=AnswerCritique)

        return response.model_dump()
    


