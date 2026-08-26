from graphrag.config import get_settings
import requests


def main():
    settings = get_settings()

    query = "What are the social characteristics of lions?"

    documents = [
        "Lions are highly social big cats that live in groups called prides.",
        "Tigers are generally solitary animals.",
        "African elephants live in complex social groups.",
    ]

    response = requests.post(
        "https://openrouter.ai/api/v1/rerank",
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.openrouter_rerank_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    print("=" * 60)
    print("RERANK TEST")
    print("=" * 60)

    for result in data["results"]:
        print()
        print("Index:", result["index"])
        print("Score:", result["relevance_score"])
        print("Document:", result["document"]["text"])


if __name__ == "__main__":
    main()