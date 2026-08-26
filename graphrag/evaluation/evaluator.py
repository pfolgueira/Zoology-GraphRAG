from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pydantic import BaseModel

from ..llm.gemini_client import GeminiClient
from ..agents import AgenticRAG
from ..graph.neo4j_manager import Neo4jManager

import time
import json

_MAX_CONTEXT_CHARS = 4000  # per-call limit to avoid truncated JSON responses

import re as _re

def _vprint(verbose: bool, *args: Any, **kwargs: Any) -> None:
    if verbose:
        print(*args, **kwargs)


def _truncate_context(chunks: List[str], max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Join context chunks and hard-truncate to avoid LLM token-limit JSON truncation."""
    processed_chunks = []
    
    for chunk in chunks:
        if isinstance(chunk, dict):
            # If the chunk is a dictionary, convert it to a string (JSON format)
            processed_chunks.append(json.dumps(chunk, ensure_ascii=False))
        else:
            # Ensure any other type is also cast to a string
            processed_chunks.append(str(chunk))
            
    joined = "\n".join(processed_chunks)
    
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n[context truncated]"
        
    return joined


def _is_no_retrieval_needed(text: str) -> bool:
    """Returns True for meta-responses that do not require knowledge retrieval.

    Covers: KB-abstention, scope refusals, and conversational/greeting replies.
    For all of these, empty retrieved context is the *correct* system behaviour.
    """
    t = text.lower().strip()
    patterns = [
        r"not in the knowledge base",
        r"not in the database",
        r"not available",
        r"not provided",
        r"cannot be determined",
        r"can't be determined",
        r"no information",
        r"insufficient information",
        r"not enough information",
        r"no relevant information",
        r"information is unavailable",
        r"outside my scope",
        r"outside the scope",
        r"i only answer",
        r"cannot answer",
        r"i am a knowledge assistant",
        r"i can answer questions about",
        r"you are welcome",
        r"goodbye",
        r"hello",
        r"hoot hoot!",
        r"umm... moo",
        r"with my eight arms and three brains"
    ]
    return any(_re.search(p, t) for p in patterns)


def _is_abstention_answer(text: str) -> bool:
    """Returns True when the answer is a 'not in KB' meta-statement.

    These are not factual claims and should always be considered faithful.
    """
    t = text.lower().strip()
    return bool(_re.search(r"not in the knowledge base|this information is not", t))


# ── Pydantic schemas for schema-constrained LLM outputs ─────────────────────
# Using structured_output_with_chat (JSON schema mode) is more reliable than
# plain format="json" for small models that occasionally drop commas or quotes.

class _AttributionResult(BaseModel):
    sentences: List[str]
    attributions: List[int]
    reasoning: str

class _Statements(BaseModel):
    statements: List[str]

class _Verification(BaseModel):
    verdicts: List[int]
    reasoning: List[str]

class _ClassificationItem(BaseModel):
    statement: str
    category: str
    reason: str

class _Classification(BaseModel):
    classifications: List[_ClassificationItem]
    tp_count: int
    fp_count: int
    fn_count: int


class RAGEvaluator:
    """
    Evaluates a GraphRAG pipeline following the RAGAS methodology:

    1. run_benchmark()    – executes Cypher ground truths, calls the agent,
                           records answers, contexts and latency
    2. evaluate_results() – scores each row with context_recall, faithfulness
                           and answer_correctness
    3. print_summary()    – aggregated metrics table

    Each RAGAS metric maps to the exact prompt goals described in class:
      • context_recall     : binary attribution per sentence (Yes/No)
      • faithfulness       : two-step statement decomposition + verification
      • answer_correctness : TP/FP/FN classification against ground truth
    """

    def __init__(self, rag: AgenticRAG, neo4j_manager: Neo4jManager):
        self.rag = rag
        self.neo4j = neo4j_manager
        self.client = GeminiClient()
        self.decomposed_answer = None

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load benchmark CSV (semicolon-delimited, columns: question, cypher)."""
        return pd.read_csv(csv_path, delimiter=";")

    def get_answer(self, question: str) -> Tuple[Optional[str], List[str]]:
        """Run the agent and return (answer, retrieved_contexts).

        Returns (None, []) on failure so the benchmark loop never crashes.
        """
        for attempt in range(3):
            try:
                result = self.rag.answer(question)
                answer = result["answer"]
                context = result["iterations"][-1]["retrieval"]["context"]
                tools = result["iterations"][-1].get('tools', [result['iterations'][-1]['retrieval'].get('tool')])
                return answer, context, tools
            except Exception as e:
                if attempt < 2:
                    print(f"Error on attempt {attempt + 1}: {e}. Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print(f"Failed after 3 attempts: {e}")
                    return None, []
        return None, []

    # ------------------------------------------------------------------
    # RAGAS metrics
    # ------------------------------------------------------------------

    def evaluate_context_recall(
        self,
        question: str,
        ground_truth: str,
        retrieved_context: List[str],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Robust Context Recall Evaluator

        Measures whether the retrieved context contains the information necessary
        to support the ground-truth answer.

        Handles important edge cases:
        ---------------------------------------------------------
        1. Empty retrieval + abstention ground truth
        Example:
        GT = "This information is not in the knowledge base."

        If no context was retrieved, this is considered CORRECT retrieval behavior
        and recall = 1.0

        2. Empty retrieval + factual ground truth
        Recall = 0.0

        3. Semantic matching (not exact wording)

        Returns:
            {
                "sentences": [...],
                "attributions": [0/1...],
                "reasoning": str,
                "recall": float
            }
        """
        _vprint(verbose, "\n── context_recall ──────────────────────────────────")
        _vprint(verbose, f"  Question    : {question}")
        _vprint(verbose, f"  Ground truth: {ground_truth}")
        _vprint(verbose, f"  Context chunks retrieved: {len(retrieved_context)}")

        def _has_context(ctx: List[str]) -> bool:
            if not ctx:
                return False
            return any(str(x).strip() for x in ctx)

        context_exists = _has_context(retrieved_context)
        # Prepend the question so the LLM can interpret bare-fact ground truths
        # (e.g. GT="Germany" + Q="Where was Einstein born?" → "born in Germany")
        # and so structured graph results (e.g. "{'count(p)': 2}") get question framing.
        context_str = f"[Question: {question}]\n\n" + _truncate_context(retrieved_context)

        # ---------------------------------------------------------
        # CASE A: Empty retrieval
        # ---------------------------------------------------------
        if not context_exists:
            _vprint(verbose, "\n  [Step 1] No retrieved context detected.")

            # Greetings, scope refusals, and KB-abstention responses all require
            # no retrieval — empty context is the correct system behaviour here.
            if _is_no_retrieval_needed(ground_truth):
                result = {
                    "sentences": [ground_truth],
                    "attributions": [1],
                    "reasoning": (
                        "No context was retrieved, and the ground truth is a "
                        "conversational, scope-refusal, or KB-abstention response "
                        "that requires no knowledge lookup. Recall = 1.0."
                    ),
                    "recall": 1.0,
                }

            else:
                result = {
                    "sentences": [ground_truth],
                    "attributions": [0],
                    "reasoning": (
                        "No context was retrieved, but the ground truth contains "
                        "answerable factual information. Retrieval failed to surface evidence."
                    ),
                    "recall": 0.0,
                }

            if verbose:
                for s, a in zip(result["sentences"], result["attributions"]):
                    label = "✓ attributed" if a else "✗ not found"
                    _vprint(verbose, f"    [{label}] {s}")

                _vprint(verbose, f"\n  Reasoning : {result['reasoning']}")
                _vprint(verbose, f"  Score     : {result['recall']:.3f}")
                _vprint(verbose, "────────────────────────────────────────────────────")

            return result

        # ---------------------------------------------------------
        # CASE B: Context exists -> normal attribution
        # ---------------------------------------------------------
        system_prompt = (
            "ROLE & DOMAIN:\n"
            "You are an expert evaluator system specializing in Zoology and animal biology.\n\n"
            "TASK:\n"
            "Your objective is to measure 'Context Recall'. You must determine to what extent the ground-truth "
            "answer is supported by the retrieved context. You will do this by breaking down the ground truth "
            "into individual claims and verifying them against the context.\n\n"
            "INSTRUCTIONS:\n"
            "1. Extract sentences: Break the ground truth answer down into individual, self-contained claims or sentences.\n"
            "2. Interpret short answers: If the ground truth is a single word or short phrase (e.g., '22 months'), "
            "use the question (e.g., 'What is the gestation period of an elephant?') to form a complete statement "
            "('The gestation period of an elephant is 22 months').\n"
            "3. Interpret structured data: The retrieved context may contain raw graph database results "
            "(e.g., {'habitat': 'Savanna'}). Interpret these conceptually to check if they support the ground truth claims.\n"
            "4. Use Context Headers: Context chunks may begin with metadata headers (e.g., 'Animal: Axolotl', 'Section: Evolution'). "
            "Use this metadata to identify the main subject of the chunk and resolve any pronouns (like 'they', 'it', 'these creatures') "
            "found in the text.\n"
            "5. Assign Attributions: For each extracted sentence, assign an attribution of 1 if the claim is explicitly "
            "present, semantically equivalent, or clearly implied by the context. Assign 0 if it is unsupported, "
            "contradicted, or missing.\n"           
            "CRITICAL:\n"
            "Base your evaluation strictly on semantic meaning, not just exact keyword matching."
        )
        user_message = (
            f"Question: {question}\n\n"
            f"Retrieved Context: {context_str}\n\n"
            f"Ground Truth Answer: {ground_truth}\n\n"
            "Analyze the ground truth sentence by sentence against the context and provide the final evaluation."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]     

        _vprint(
            verbose,
            "\n  [Step 1] Asking the LLM to attribute each ground-truth sentence to context..."
        )

        result_obj = self.client.structured_output_with_chat(
            messages,
            _AttributionResult
        )

        result = result_obj.model_dump()

        # Force consistency
        attributions = result.get("attributions", [])
        if attributions:
            result["recall"] = sum(attributions) / len(attributions)
        else:
            result["recall"] = 0.0

        # ---------------------------------------------------------
        # Verbose output
        # ---------------------------------------------------------
        if verbose:
            sentences = result.get("sentences", [])

            _vprint(verbose, f"\n  Ground-truth sentences ({len(sentences)}):")

            for s, a in zip(sentences, attributions):
                label = "✓ attributed" if a else "✗ not found"
                _vprint(verbose, f"    [{label}] {s}")

            _vprint(verbose, f"\n  Reasoning : {result.get('reasoning', '')}")
            _vprint(verbose, f"  Score     : {result['recall']:.3f}")
            _vprint(verbose, "────────────────────────────────────────────────────")

        return result

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: List[str],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Robust faithfulness evaluator.

        Handles two modes automatically:

        1. Retrieval-grounded mode (context exists)
        Measures whether factual claims in the answer are supported by context.

        2. No-context conversational mode
        Prevents unfair hallucination penalties for greetings, assistant-role
        statements, generic capability claims, and harmless conversational text.

        Returns:
            {
                "statements": [...],
                "verdicts": [...],     # 1 supported / 0 unsupported
                "faithfulness": float,
                "reasoning": [...]
            }
        """
        _vprint(verbose, "\n── faithfulness ────────────────────────────────────")
        _vprint(verbose, f"  Question: {question}")
        _vprint(verbose, f"  Answer  : {answer}")


        # ---------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------
        def _has_context(ctx: List[str]) -> bool:
            if not ctx:
                return False
            return any(str(x).strip() for x in ctx)

        # Prepend question so structured graph results are interpretable in context.
        context_str = f"[Question: {question}]\n\n" + _truncate_context(context)
        has_context = _has_context(context)

        # Remove inline citations like [1], [2]
        answer_clean = _re.sub(r"\s*\[\d+\]", "", answer).strip()

        # "Not in knowledge base" is a meta-statement, not a factual claim.
        # It is always faithful — verifying it against chunks would be meaningless.
        if _is_abstention_answer(answer_clean):
            _vprint(verbose, "\n  Answer is a KB-abstention — faithfulness = 1.0 by convention.")
            _vprint(verbose, "────────────────────────────────────────────────────")
            return {
                "statements": [answer_clean],
                "verdicts": [1],
                "faithfulness": 1.0,
                "reasoning": ["KB-abstention responses make no factual claim and are always faithful."],
            }

        # ---------------------------------------------------------
        # Step 1: Decompose answer into meaningful claims
        # ---------------------------------------------------------
        decompose_prompt = (
            "ROLE & DOMAIN:\n"
            "You are an expert evaluator system specializing in Zoology and animal biology.\n\n"
            "TASK:\n"
            "Given a question and a factual answer, break the answer down into a list of concise, "
            "standalone factual statements. This is the first step for a 'Faithfulness' evaluation.\n\n"
            "INSTRUCTIONS:\n"
            "1. Make statements standalone: Each statement must contain enough context to make sense "
            "in complete isolation.\n"
            "2. Resolve pronouns: Replace pronouns ('it', 'they', 'these creatures') with the specific animal "
            "or entity being discussed in the answer. (e.g., 'It hunts at night' -> 'The leopard hunts at night').\n"
            "3. Ignore citations: Remove any reference markers, brackets, or citations like [1], [2], or 'According to the text'.\n"
            "4. Avoid oversplitting: Combine closely related attributes into a single factual statement rather "
            "than making many redundant tiny statements (e.g., 'The frog is small and green' instead of "
            "'The frog is small' and 'The frog is green').\n"
            "5. Do not alter meaning: Extract the facts exactly as they are presented in the answer, even if "
            "you know them to be biologically incorrect. Your job here is only extraction, not verification."
        )

        _vprint(verbose, "\n  [Step 1] Decomposing the answer into statements...")

        try:
            stmt_obj = self.client.structured_output_with_chat(
                [
                    {"role": "system", "content": decompose_prompt},
                    {
                        "role": "user",
                        "content": f"Question: {question}\nAnswer: {answer_clean}",
                    },
                ],
                _Statements,
            )
            statements = stmt_obj.statements
            self.decomposed_answer = statements  # Store for potential later use in correctness evaluation
        except Exception as exc:
            _vprint(verbose, f"  [Step 1 ERROR] {exc}")
            return {
                "statements": [],
                "verdicts": [],
                "faithfulness": 0.0,
                "reasoning": [],
            }

        if verbose:
            _vprint(verbose, f"  Statements extracted ({len(statements)}):")
            for i, s in enumerate(statements, 1):
                _vprint(verbose, f"    {i}. {s}")

        if not statements:
            return {
                "statements": [],
                "verdicts": [],
                "faithfulness": 1.0,
                "reasoning": [],
            }

        # ---------------------------------------------------------
        # MODE A: No context supplied
        # ---------------------------------------------------------
        if not has_context:
            _vprint(verbose, "\n  [Step 2] No external context detected.")
            _vprint(verbose, "           Using conversational truthfulness mode...")

            verify_prompt = (
                "ROLE & DOMAIN:\n"
                "You are an expert evaluator for a Zoology-focused Retrieval-Augmented Generation (RAG) system.\n\n"
                
                "TASK:\n"
                "Evaluate the 'Conversational Truthfulness' of statements generated when NO external database context "
                "was retrieved. Your goal is to penalize factual hallucinations while allowing safe conversational filler "
                "and intended animal-based roleplay.\n\n"
                
                "INSTRUCTIONS:\n"
                "1. Assign Verdict = 1 (Acceptable) for:\n"
                "   - Greetings, farewells, and polite conversational filler.\n"
                "   - Animal persona roleplay or thematic identity claims (e.g., 'I am a dolphin navigating the data ocean', 'I use my gorilla muscles to fetch data'). These are designed UI features, NOT factual hallucinations.\n"
                "   - Generic capability statements or offers to help.\n"
                "   - Scope boundary setting and refusals (e.g., 'I cannot answer that', 'I only focus on zoology').\n\n"
                
                "2. Assign Verdict = 0 (Unacceptable / Hallucination) for:\n"
                "   - Any specific zoological fact, biological trait, or claim about the animal kingdom. (Since there is no retrieved context, the system is forbidden from asserting zoological facts from its internal memory).\n"
                "   - Specific numbers, dates, geographical data, or statistics about the real world.\n\n"
                
                "3. Output:\n"
                "   - Ensure you return exactly one verdict (1 or 0) and a brief reasoning for each statement."
            )

            statements_str = "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(statements)
            )

            verif_obj = self.client.structured_output_with_chat(
                [
                    {"role": "system", "content": verify_prompt},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nStatements:\n{statements_str}",
                    },
                ],
                _Verification,
            )

            verdicts = verif_obj.verdicts
            reasoning = verif_obj.reasoning

        # ---------------------------------------------------------
        # MODE B: Context supplied
        # ---------------------------------------------------------
        else:
            _vprint(verbose, "\n  [Step 2] Context detected.")
            _vprint(verbose, "           Using retrieval-grounded mode...")
            
            verify_prompt = (
                "ROLE & DOMAIN:\n"
                "You are an expert evaluator for a Zoology-focused Retrieval-Augmented Generation (RAG) system.\n\n"
                
                "TASK:\n"
                "Evaluate the 'Faithfulness' of generated statements against the retrieved context. Your goal is to "
                "determine if every statement is strictly grounded in the provided context, penalizing any ungrounded claims.\n\n"
                
                "INSTRUCTIONS:\n"
                "1. Use Context Headers: The context may contain metadata headers (e.g., 'Animal: Axolotl'). Use these "
                "to resolve pronouns or implicit references when checking the statements.\n"
                "2. Assign Verdict = 1 (Supported) if the statement is:\n"
                "   - Explicitly supported by the provided context.\n"
                "   - Semantically equivalent to information in the context.\n"
                "   - Clearly and directly implied by the context using basic logic.\n"
                "3. Assign Verdict = 0 (Unsupported/Hallucinated) if the statement:\n"
                "   - Adds new zoological facts or details NOT present in the context (CRITICAL: Even if you know the fact is biologically true in the real world, mark it 0 if the context does not state it).\n"
                "   - Contradicts the context.\n"
                "   - Exaggerates certainty or requires unwarranted speculation.\n"
                "4. Evaluation Rules:\n"
                "   - Use semantic meaning rather than exact keyword matching.\n"
                "   - Use the original question to interpret shorthand or fragmented statements.\n"
                "5. Format Requirements:\n"
                "   - Ensure you return exactly one verdict (1 or 0) and one brief reasoning string for each statement."
            )

            statements_str = "\n".join(
                f"{i + 1}. {s}" for i, s in enumerate(statements)
            )

            verif_obj = self.client.structured_output_with_chat(
                [
                    {"role": "system", "content": verify_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Context:\n{context_str}\n\n"
                            f"Statements:\n{statements_str}"
                        ),
                    },
                ],
                _Verification,
            )

            verdicts = verif_obj.verdicts
            reasoning = verif_obj.reasoning

        # ---------------------------------------------------------
        # Score
        # ---------------------------------------------------------
        faithfulness_score = (
            sum(verdicts) / len(verdicts)
            if verdicts
            else 1.0
        )

        # ---------------------------------------------------------
        # Verbose output
        # ---------------------------------------------------------
        if verbose:
            _vprint(verbose, "\n  Verdicts:")
            for s, v, r in zip(statements, verdicts, reasoning):
                label = "✓ supported" if v else "✗ unsupported"
                _vprint(verbose, f"    [{label}] {s}")
                if r:
                    _vprint(verbose, f"              → {r}")

            _vprint(verbose, f"\n  Score : {faithfulness_score:.3f}")
            _vprint(verbose, "────────────────────────────────────────────────────")

        return {
            "statements": statements,
            "verdicts": verdicts,
            "faithfulness": faithfulness_score,
            "reasoning": reasoning,
        }

    def evaluate_answer_correctness(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Goal: Given a ground truth and an answer statement, analyze each statement
        and classify it into one of the following categories:

        TP (true positive): Statements present in the answer that are also directly
            supported by one or more statements in the ground truth.
        FP (false positive): Statements present in the answer but not directly
            supported by any statement in the ground truth.
        FN (false negative): Statements found in the ground truth but not present
            in the answer.

        Each statement can only belong to one of these categories.
        Provide a reason for each classification.

        verbose=True prints each step so students can follow the reasoning.
        """
        _vprint(verbose, "\n── answer_correctness ──────────────────────────────")
        _vprint(verbose, f"  Question    : {question}")
        _vprint(verbose, f"  Answer      : {answer}")
        _vprint(verbose, f"  Ground truth: {ground_truth}")

        breakdown_prompt = (
            "ROLE & DOMAIN:\n"
            "You are an expert evaluator system specializing in Zoology and animal biology.\n\n"
            
            "TASK:\n"
            "Given a question and its verified Ground Truth answer, break the Ground Truth down into a list "
            "of concise, standalone factual statements. This is a crucial step for an 'Answer Correctness' evaluation.\n\n"
            
            "INSTRUCTIONS:\n"
            "1. Interpret short answers (CRITICAL): Ground Truths are often brief fragments (e.g., '22 months' or 'Savanna'). "
            "You MUST use the provided original question to expand these fragments into fully formed, standalone statements. "
            "(e.g., Question: 'Where does the lion live?', Ground Truth: 'Savanna' -> Output: 'The lion lives in the savanna').\n"
            "2. Make statements standalone: Each statement must contain enough context to make sense in complete isolation.\n"
            "3. Resolve pronouns: Replace any pronouns ('it', 'they') with the specific animal or entity being discussed.\n"
            "4. Avoid oversplitting: Combine closely related attributes into a single factual statement rather than creating "
            "artificially fragmented sentences.\n"
            "5. Pure Extraction: Extract only what is present in the Ground Truth. Do not add your own external knowledge."
        )

        def _get_statements(text: str) -> List[str]:
            # Limpiamos las citas por regex como medida de seguridad adicional antes de enviar al LLM
            clean = _re.sub(r"\s*\[\d+\]", "", text).strip()
            
            try:
                obj = self.client.structured_output_with_chat(
                    [
                        {"role": "system", "content": breakdown_prompt},
                        {"role": "user", "content": f"Question: {question}\nText: {clean}"},
                    ],
                    _Statements,
                )
                return obj.statements
            except Exception as exc:
                _vprint(verbose, f"  [Breakdown ERROR] {exc}")
                return []

        _vprint(verbose, f"  Answer statements ({len(self.decomposed_answer)}):")
        for i, s in enumerate(self.decomposed_answer, 1):
            _vprint(verbose, f"    {i}. {s}")

        _vprint(verbose, "\n  [Step 2] Decomposing the ground truth into statements...")
        truth_statements = _get_statements(ground_truth)
        _vprint(verbose, f"  Ground truth statements ({len(truth_statements)}):")
        for i, s in enumerate(truth_statements, 1):
            _vprint(verbose, f"    {i}. {s}")

        classify_prompt = (
            "ROLE & DOMAIN:\n"
            "You are an expert evaluator system specializing in Zoology and animal biology.\n\n"
            
            "TASK:\n"
            "Compare a list of generated 'Answer Statements' against a list of 'Ground Truth Statements' "
            "to measure Answer Correctness. You must classify the semantic relationship using TP, FP, and FN.\n\n"
            
            "INSTRUCTIONS:\n"
            "1. Evaluate 'Answer Statements' (TP vs. FP):\n"
            "   - Assign TP (True Positive): The Answer Statement is explicitly stated, clearly implied, or "
            "semantically equivalent to ANY of the Ground Truth Statements. CRITICAL: ALSO assign TP to any "
            "harmless conversational filler, greetings, or persona roleplay (e.g., 'Hello!', 'I am a dolphin', "
            "'Let me check my database'). Do not penalize the system for being polite or playing its character.\n"
            "   - Assign FP (False Positive): The Answer Statement introduces materially new factual zoological "
            "claims, unsupported details, or contradictions NOT justified by the Ground Truth.\n"
            "2. Evaluate 'Ground Truth Statements' (FN):\n"
            "   - Assign FN (False Negative): A Ground Truth Statement contains important information that is "
            "entirely missing from the Answer Statements.\n"
            "3. Rules for Semantic Matching:\n"
            "   - Evaluate biological meaning, not exact wording.\n"
            "   - Specific examples named in the Ground Truth may be restated as examples without penalty.\n"
            "4. Formatting & Counting:\n"
            "   - Provide a concise reasoning.\n"
            "   - CRITICAL: The tp_count, fp_count, and fn_count integers MUST exactly match the total number of "
            "TP, FP, and FN items in your classifications list."
        )

        answer_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.decomposed_answer))
        truth_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(truth_statements))

        _vprint(verbose, "\n  [Step 3] Classifying each statement as TP / FP / FN...")
        classif_obj = self.client.structured_output_with_chat(
            [
                {"role": "system", "content": classify_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Answer statements:\n{answer_str}\n\n"
                        f"Ground truth statements:\n{truth_str}"
                    ),
                },
            ],
            _Classification,
        )

        tp = classif_obj.tp_count
        fp = classif_obj.fp_count
        fn = classif_obj.fn_count

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        if verbose:
            _vprint(verbose, f"  Classifications:")
            for c in classif_obj.classifications:
                _vprint(verbose, f"    [{c.category}] {c.statement} — {c.reason}")
            _vprint(verbose, f"\n  TP={tp}  FP={fp}  FN={fn}")
            _vprint(verbose, f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
            _vprint(verbose, "────────────────────────────────────────────────────")

        return {
            "classifications": [c.model_dump() for c in classif_obj.classifications],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "answer_correctness": f1,
        }

    # ------------------------------------------------------------------
    # Pipeline: run_benchmark → evaluate_results → print_summary
    # ------------------------------------------------------------------

    def run_benchmark(self, dataset: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Execute the ground-truth Cypher queries, call the agent for each question,
        and record answers, contexts and latency (mirrors Listing 8.2).

        Input DataFrame must have columns: question, cypher, tool
        Output DataFrame adds:  ground_truth, answer, latency, retrieved_contexts, used_tools, expected_tool
        Pass verbose=True to print each question, its ground truth and the agent's answer.
        """
        answers: List = []
        ground_truths: List = []
        latencies: List = []
        contexts: List = []
        used_tools_list: List = []
        expected_tools_list: List = []

        import os
        import ast
        start_idx_bench = 0
        if os.path.exists("benchmark_backup.csv"):
            try:
                b_df = pd.read_csv("benchmark_backup.csv", sep=";")
                start_idx_bench = len(b_df)
                if start_idx_bench > 0:
                    print(f"Resuming benchmark from row {start_idx_bench + 1}...")
                    answers = b_df["answer"].tolist()
                    ground_truths = b_df["ground_truth"].tolist()
                    latencies = b_df["latency"].tolist()
                    contexts = [ast.literal_eval(c) if isinstance(c, str) and c.startswith('[') else c for c in b_df["retrieved_contexts"]]
                    used_tools_list = b_df["used_tools"].tolist()
                    expected_tools_list = b_df["expected_tool"].tolist()
            except Exception as e:
                print(f"Failed to load benchmark backup: {e}")
                start_idx_bench = 0

        for i, (_, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset), desc="Processing rows"), 1):
            if (i - 1) < start_idx_bench:
                continue

            time.sleep(5)
            _vprint(verbose, f"\n{'━'*60}")
            _vprint(verbose, f"  [{i}/{len(dataset)}] {row['question']}")
            _vprint(verbose, f"{'━'*60}")

            # Execute Cypher to obtain the ground truth dynamically
            _vprint(verbose, f"  [Step 1] Executing Cypher ground truth...")
            _vprint(verbose, f"           {row['cypher']}")
            gt_records = self.neo4j.execute_query(row["cypher"])
            gt_values = []
            for r in gt_records:
                # Convertimos el registro de Neo4j a un diccionario estándar
                r_dict = dict(r)
                
                # Si existe la columna ground_truth, la usamos. Si no, tomamos todo el diccionario.
                if "ground_truth" in r_dict:
                    val = r_dict["ground_truth"]
                else:
                    # Formatea las columnas en pares "clave: valor" (ej: "species: Elephant, preys_on: []")
                    val = ", ".join([f"{k}: {v}" for k, v in r_dict.items() if v is not None])
                
                if val is not None and str(val) not in ("None", "", "{}"):
                    gt_values.append(str(val))
            
            gt_str = "; ".join(gt_values) if gt_values else "This information is not in the knowledge base."
            ground_truths.append(gt_str)
            _vprint(verbose, f"  Ground truth: {gt_str}")
            # Call the agent
            _vprint(verbose, f"\n  [Step 2] Calling the RAG agent...")
            start = datetime.now()
            try:
                answer, context, used_tools_val = self.get_answer(row["question"])
            except Exception:
                answer, context, used_tools_val = None, [], []
            elapsed = (datetime.now() - start).total_seconds()
            latencies.append(elapsed)

            _vprint(verbose, f"  Answer  : {answer}")
            _vprint(verbose, f"  Context chunks: {len(context)}")
            _vprint(verbose, f"  Latency : {elapsed:.2f}s")
            _vprint(verbose, f"  Used tools: {used_tools_val}")
            
            expected_tool = row.get("tool", None)
            _vprint(verbose, f"  Expected tool: {expected_tool}")

            answers.append(answer)
            contexts.append(context)
            used_tools_list.append(used_tools_val)
            expected_tools_list.append(expected_tool)

            # Guardar backup intermedio de los resultados
            temp_results = dataset.iloc[:i].copy()
            temp_results["ground_truth"] = ground_truths
            temp_results["answer"] = answers
            temp_results["latency"] = latencies
            temp_results["retrieved_contexts"] = contexts
            temp_results["used_tools"] = used_tools_list
            temp_results["expected_tool"] = expected_tools_list
            temp_results.to_csv("benchmark_backup.csv", sep=";", index=False)

        results = dataset.copy()
        results["ground_truth"] = ground_truths
        import os
        start_idx_eval = 0
        if os.path.exists("evaluation_backup.csv"):
            try:
                e_df = pd.read_csv("evaluation_backup.csv", sep=";")
                if "context_recall" in e_df.columns:
                    valid_e_df = e_df.dropna(subset=["context_recall"])
                    start_idx_eval = len(valid_e_df)
                    if start_idx_eval > 0:
                        print(f"Resuming evaluation from row {start_idx_eval + 1}...")
                        recall_scores = valid_e_df["context_recall"].tolist()
                        faithfulness_scores = valid_e_df["faithfulness"].tolist()
                        correctness_scores = valid_e_df["answer_correctness"].tolist()
                        tool_correct_scores = valid_e_df["is_tool_correct"].tolist()
            except Exception as e:
                print(f"Failed to load evaluation backup: {e}")
                start_idx_eval = 0

        for idx_ev, (_, row) in enumerate(tqdm(results.iterrows(), total=len(results), desc="Evaluating")):
            if idx_ev < start_idx_eval:
                continue

        results["latency"] = latencies
        results["retrieved_contexts"] = contexts
        results["used_tools"] = used_tools_list
        results["expected_tool"] = expected_tools_list
        return results

    def evaluate_results(self, results_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Apply the three RAGAS metrics to every row in the results DataFrame
        and add the scores as new columns (mirrors Listing 8.3 / 8.4).

        Missing answers are replaced with "I don't know" before scoring.
        Pass verbose=True to print the step-by-step reasoning for every row.
        """
        df = results_df.fillna("I don't know").copy()

        recall_scores: List[float] = []
        faithfulness_scores: List[float] = []
        correctness_scores: List[float] = []
        tool_correct_scores: List[bool] = []

        import os
        start_idx_eval = 0
        if os.path.exists("evaluation_backup.csv"):
            try:
                e_df = pd.read_csv("evaluation_backup.csv", sep=";")
                if "context_recall" in e_df.columns:
                    valid_e_df = e_df.dropna(subset=["context_recall"])
                    start_idx_eval = len(valid_e_df)
                    if start_idx_eval > 0:
                        print(f"Resuming evaluation from row {start_idx_eval + 1}...")
                        recall_scores = valid_e_df["context_recall"].tolist()
                        faithfulness_scores = valid_e_df["faithfulness"].tolist()
                        correctness_scores = valid_e_df["answer_correctness"].tolist()
                        tool_correct_scores = valid_e_df["is_tool_correct"].tolist()
            except Exception as e:
                print(f"Failed to load evaluation backup: {e}")
                start_idx_eval = 0

        for idx_ev, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Evaluating")):
            if idx_ev < start_idx_eval:
                continue

            time.sleep(5)
            self.decomposed_answer = None  # Reset before each question
            if verbose:
                print(f"\n{'━'*60}")
                print(f"  Q: {row['question']}")
                print(f"{'━'*60}")

            contexts = row["retrieved_contexts"]
            if isinstance(contexts, str):
                contexts = [contexts]

            # Compare exact match between used and expected
            # Ensure used_tools is converted to a comparable format if needed. Here we assume lists or strings.
            used = row.get("used_tools", [])
            expected = row.get("expected_tool", "")
            
            # Simple check if expected tool name represents what was used. Customise if used_tools has format changes
            if isinstance(used, list) and len(used) > 0:
                 is_tool_correct = (expected.strip().lower() == str(used[-1]).strip().lower())
                 # Adjust matching logic if used_tools format differs e.g. list of dicts.
            elif isinstance(used, str):
                 is_tool_correct = (expected.strip().lower() == used.strip().lower())
            else:
                 is_tool_correct = False

            recall = None
            for _ in range(3):
                try:
                    recall = self.evaluate_context_recall(
                        row["question"], row["ground_truth"], contexts, verbose=verbose
                    )
                    break
                except Exception as e:
                    print(f"Error en evaluate_context_recall: {e}. Reintentando en 10 segundos...")
                    time.sleep(10)
            if recall is None:
                recall = {"recall": 0.0}

            faith = None
            for _ in range(3):
                try:
                    faith = self.evaluate_faithfulness(
                        row["question"], row["answer"], contexts, verbose=verbose
                    )
                    break
                except Exception as e:
                    print(f"Error en evaluate_faithfulness: {e}. Reintentando en 10 segundos...")
                    time.sleep(10)
            if faith is None:
                faith = {"faithfulness": 0.0}

            corr = None
            for _ in range(3):
                try:
                    corr = self.evaluate_answer_correctness(
                        row["question"], row["answer"], row["ground_truth"], verbose=verbose
                    )
                    break
                except Exception as e:
                    print(f"Error en evaluate_answer_correctness: {e}. Reintentando en 10 segundos...")
                    time.sleep(10)
            if corr is None:
                corr = {"answer_correctness": 0.0}

            recall_scores.append(float(recall.get("recall", 0.0)))
            faithfulness_scores.append(float(faith.get("faithfulness", 0.0)))
            correctness_scores.append(float(corr.get("answer_correctness", 0.0)))
            tool_correct_scores.append(is_tool_correct)

            # Guardar backup intermedio de las evaluaciones
            temp_df = df.iloc[:len(recall_scores)].copy()
            temp_df["context_recall"] = recall_scores
            temp_df["faithfulness"] = faithfulness_scores
            temp_df["answer_correctness"] = correctness_scores
            temp_df["is_tool_correct"] = tool_correct_scores
            temp_df.to_csv("evaluation_backup.csv", sep=";", index=False)

        df["context_recall"] = recall_scores
        df["faithfulness"] = faithfulness_scores
        df["answer_correctness"] = correctness_scores
        df["is_tool_correct"] = tool_correct_scores
        
        # Calculate overall accuracy
        total_evaluations = len(tool_correct_scores)
        correct_evaluations = sum(tool_correct_scores)
        tool_accuracy = correct_evaluations / total_evaluations if total_evaluations > 0 else 0.0
        
        if verbose:
            print(f"\n{'━'*60}")
            print(f"  Tool Selection Accuracy: {tool_accuracy:.2%} ({correct_evaluations}/{total_evaluations})")
            print(f"{'━'*60}")

        return df

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print a benchmark summary table matching Table 8.5 in the book."""
        print("\n=== Benchmark Summary ===")
        metric_cols = ["answer_correctness", "context_recall", "faithfulness"]
        for col in metric_cols:
            if col in results_df.columns:
                print(f"{col:25s}: {results_df[col].mean():.4f}")
        if "latency" in results_df.columns:
            print(f"{'avg_latency_s':25s}: {results_df['latency'].mean():.2f}")
