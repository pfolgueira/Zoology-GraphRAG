from graphrag.config import get_settings
from graphrag.utils.embeddings import EmbeddingGenerator

def main():
    settings = get_settings()

    print("=" * 60)
    print("TEST DE EMBEDDINGS")
    print("=" * 60)

    print(f"Modelo:       {settings.embedding_model}")
    print(f"Dimensiones:  {settings.embedding_dimensions}")
    print()

    generator = EmbeddingGenerator()

    # ---------------------------------------------------------
    # 1. Prueba de un único texto
    # ---------------------------------------------------------

    text = "What is the habitat of the lion?"

    print("1. Generando embedding individual...")
    embedding = generator.embed_text(text)

    print(f"   Texto: {text}")
    print(f"   Dimensiones obtenidas: {len(embedding)}")
    print(f"   Primeros valores: {embedding[:5]}")
    print()

    assert len(embedding) == settings.embedding_dimensions, (
        f"Dimensiones incorrectas: "
        f"esperadas {settings.embedding_dimensions}, "
        f"obtenidas {len(embedding)}"
    )

    print("   ✓ Embedding individual correcto")
    print()

    # ---------------------------------------------------------
    # 2. Prueba de batch
    # ---------------------------------------------------------

    texts = [
        "What is the habitat of the lion?",
        "How do penguins adapt to cold environments?",
        "What do elephants eat?",
    ]

    print("2. Generando embeddings en batch...")
    embeddings = generator.embed_texts(texts)

    print(f"   Textos enviados: {len(texts)}")
    print(f"   Embeddings recibidos: {len(embeddings)}")

    assert len(embeddings) == len(texts), (
        f"Se esperaban {len(texts)} embeddings, "
        f"pero se recibieron {len(embeddings)}"
    )

    for i, emb in enumerate(embeddings):
        print(
            f"   Embedding {i + 1}: "
            f"{len(emb)} dimensiones"
        )

        assert len(emb) == settings.embedding_dimensions, (
            f"Embedding {i + 1}: "
            f"esperadas {settings.embedding_dimensions}, "
            f"obtenidas {len(emb)}"
        )

    print()
    print("   ✓ Batch correcto")
    print()

    # ---------------------------------------------------------
    # 3. Comprobar valores
    # ---------------------------------------------------------

    print("3. Comprobando valores...")

    for i, emb in enumerate(embeddings):
        assert all(isinstance(x, (int, float)) for x in emb), (
            f"El embedding {i + 1} contiene valores no numéricos"
        )

    print("   ✓ Todos los valores son numéricos")
    print()

    # ---------------------------------------------------------
    # Resultado
    # ---------------------------------------------------------

    print("=" * 60)
    print("✓ TODAS LAS PRUEBAS HAN PASADO")
    print("=" * 60)


if __name__ == "__main__":
    main()