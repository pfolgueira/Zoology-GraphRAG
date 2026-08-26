from typing import List, Dict, Any

import requests

from graphrag.config import get_settings
from graphrag.graph.neo4j_manager import Neo4jManager
from graphrag.retrieval.fulltext_retriever import FullTextRetriever
from graphrag.retrieval.vector_retriever import VectorRetriever

class HybridRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.settings = get_settings()
        self.vector_retriever = VectorRetriever(neo4j_manager)
        self.fulltext_retriever = FullTextRetriever(neo4j_manager)
        self.rerank_model = self.settings.openrouter_rerank_model

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Combina resultados de búsqueda vectorial y full-text en un ranking único.
        """
        top_k_candidates = self.settings.top_k_candidates
        top_k_results = top_k or self.settings.top_k_results

        vector_results = self.vector_retriever.retrieve(query=query, top_k=top_k_candidates)
        fulltext_results = self.fulltext_retriever.retrieve(query=query, top_k=top_k_candidates)

        fused_candidates = self._fuse_results(vector_results, fulltext_results, top_k=top_k_candidates)

        reranked_results = self._rerank_results(query, fused_candidates)

        return reranked_results[:top_k_results]

    def _rerank_results(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reordena los candidatos usando el reranker de OpenRouter."""

        if not candidates:
            return []

        documents = [
            doc.get("text", "")
            for doc in candidates
        ]

        response = requests.post(
            "https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        reranked_results = []

        for result in data["results"]:
            index = result["index"]
            score = float(result["relevance_score"])

            doc = candidates[index].copy()

            doc["rerank_score"] = score
            doc["hybrid_score"] = doc.get("score")
            doc["score"] = score

            reranked_results.append(doc)

        return reranked_results

    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        fulltext_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fusiona resultados normalizando scores y ponderando cada retriever."""
        weight_vector = 0.6
        weight_fulltext = 0.4

        vector_scores = self._normalize_scores(vector_results)
        fulltext_scores = self._normalize_scores(fulltext_results)

        merged: Dict[str, Dict[str, Any]] = {}

        for row in vector_results:
            chunk_id = str(row.get("chunk_id") or row.get("text"))
            merged[chunk_id] = {
                "text": row.get("text", ""),
                "chunk_id": row.get("chunk_id"),
                "vector_score": vector_scores.get(chunk_id, 0.0),
                "fulltext_score": 0.0,
            }

        for row in fulltext_results:
            chunk_id = str(row.get("chunk_id") or row.get("text"))
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "text": row.get("text", ""),
                    "chunk_id": row.get("chunk_id"),
                    "vector_score": 0.0,
                    "fulltext_score": fulltext_scores.get(chunk_id, 0.0),
                }
            else:
                merged[chunk_id]["fulltext_score"] = fulltext_scores.get(chunk_id, 0.0)

        for row in merged.values():
            row["score"] = (
                weight_vector * row["vector_score"]
                + weight_fulltext * row["fulltext_score"]
            )

        sorted_results = sorted(
            merged.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )

        return sorted_results[:top_k]

    def _normalize_scores(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Normaliza los scores al rango [0, 1] para hacerlos comparables."""
        if not rows:
            return {}

        raw_scores = [float(row.get("score", 0.0)) for row in rows]
        max_score = max(raw_scores)
        min_score = min(raw_scores)

        if max_score == min_score:
            return {
                str(row.get("chunk_id") or row.get("text")): 1.0
                for row in rows
            }

        normalized: Dict[str, float] = {}
        for row in rows:
            chunk_id = str(row.get("chunk_id") or row.get("text"))
            score = float(row.get("score", 0.0))
            normalized[chunk_id] = (score - min_score) / (max_score - min_score)

        return normalized
