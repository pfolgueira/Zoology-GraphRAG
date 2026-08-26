from graphrag.retrieval.vector_retriever import VectorRetriever
from graphrag.graph.neo4j_manager import Neo4jManager


def main():
    neo4j = Neo4jManager()

    try:
        retriever = VectorRetriever(neo4j)

        query = "Are lions social animals?"

        results = retriever.retrieve(
            query=query,
            top_k=5
        )

        print("=" * 60)
        print("VECTOR RETRIEVAL TEST")
        print("=" * 60)
        print(f"\nQuery: {query}\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.get('score')}")
            print(f"   Chunk ID: {result.get('chunk_id')}")
            print(f"   Text: {result.get('text')}")
            print(f"   Questions: {result.get('matched_questions')}")
            print()

    finally:
        neo4j.close()


if __name__ == "__main__":
    main()