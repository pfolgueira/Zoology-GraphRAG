from graphrag.config import get_settings
from graphrag.graph.neo4j_manager import Neo4jManager
from graphrag.utils.embeddings import EmbeddingGenerator


BATCH_SIZE = 100


def get_questions(neo4j: Neo4jManager):
    query = """
    MATCH (q:Question)
    WHERE q.text IS NOT NULL
      AND q.question_embedding_new IS NULL
    RETURN
        elementId(q) AS element_id,
        q.text AS text
    ORDER BY elementId(q)
    """

    return neo4j.execute_query(query)


def save_embeddings(
    neo4j: Neo4jManager,
    questions,
    embeddings,
):
    query = """
    UNWIND $items AS item
    MATCH (q)
    WHERE elementId(q) = item.element_id
    SET q.question_embedding_new = item.embedding
    RETURN count(q) AS updated
    """

    items = [
        {
            "element_id": question["element_id"],
            "embedding": embedding,
        }
        for question, embedding in zip(questions, embeddings)
    ]

    result = neo4j.execute_query(
        query,
        {"items": items},
    )

    updated = result[0]["updated"]

    if updated != len(items):
        raise RuntimeError(
            f"Se esperaban actualizar {len(items)} nodos, "
            f"pero Neo4j actualizó {updated}."
        )

    return updated

def check_embeddings(
    neo4j: Neo4jManager,
    dimensions: int,
):
    query = """
    MATCH (q:Question)
    WHERE q.text IS NOT NULL
    RETURN
        count(q) AS total,
        count(q.question_embedding_new) AS with_embedding,
        count(CASE
            WHEN q.question_embedding_new IS NOT NULL
             AND size(q.question_embedding_new) = $dimensions
            THEN 1
        END) AS with_correct_dimensions
    """

    result = neo4j.execute_query(
        query,
        {"dimensions": dimensions},
    )

    return result[0]


def main():
    settings = get_settings()

    print("=" * 60)
    print("REGENERACIÓN DE QUESTION EMBEDDINGS")
    print("=" * 60)
    print(f"Modelo:       {settings.embedding_model}")
    print(f"Dimensiones:  {settings.embedding_dimensions}")
    print(f"Batch size:   {BATCH_SIZE}")
    print()

    neo4j = Neo4jManager()
    embedding_generator = EmbeddingGenerator()

    # ---------------------------------------------------------
    # Obtener Questions pendientes
    # ---------------------------------------------------------

    questions = get_questions(neo4j)

    total = len(questions)

    print(f"Questions pendientes: {total}")

    if total == 0:
        print("No hay Questions pendientes.")
        print()
        print("Comprobando estado final...")
    else:
        print()

        # -----------------------------------------------------
        # Procesar por batches
        # -----------------------------------------------------

        processed = 0

        for start in range(0, total, BATCH_SIZE):
            batch = questions[start:start + BATCH_SIZE]

            print(
                f"[{start + 1}-{start + len(batch)} / {total}] "
                "Generando embeddings..."
            )

            texts = [
                question["text"]
                for question in batch
            ]

            embeddings = embedding_generator.embed_texts(texts)

            # -------------------------------------------------
            # Validar respuesta de OpenRouter
            # -------------------------------------------------

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"OpenRouter devolvió {len(embeddings)} embeddings "
                    f"para {len(batch)} Questions."
                )

            for question, embedding in zip(batch, embeddings):
                if len(embedding) != settings.embedding_dimensions:
                    raise RuntimeError(
                        f"Dimensiones incorrectas para "
                        f"Question {question['id']}: "
                        f"esperadas {settings.embedding_dimensions}, "
                        f"obtenidas {len(embedding)}."
                    )

            # -------------------------------------------------
            # Guardar en Neo4j
            # -------------------------------------------------

            updated = save_embeddings(
                neo4j,
                batch,
                embeddings,
            )

            processed += updated

            print(
                f"  ✓ Guardadas {processed}/{total}"
            )

        print()

    # ---------------------------------------------------------
    # Comprobación final
    # ---------------------------------------------------------

    stats = check_embeddings(
        neo4j,
        settings.embedding_dimensions,
    )

    total_questions = stats["total"]
    with_embedding = stats["with_embedding"]
    correct_dimensions = stats["with_correct_dimensions"]

    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    print(f"Questions:              {total_questions}")
    print(f"Con embedding:          {with_embedding}")
    print(f"Con dimensiones correctas: {correct_dimensions}")

    if (
        total_questions != with_embedding
        or total_questions != correct_dimensions
    ):
        raise RuntimeError(
            "La migración no se ha completado correctamente."
        )

    print()
    print("✓ Todas las Questions tienen un embedding de")
    print(f"  {settings.embedding_dimensions} dimensiones.")
    print()
    print("El campo utilizado es:")
    print("  Question.question_embedding_new")
    print()
    print("El embedding original NO ha sido modificado.")
    print("=" * 60)


if __name__ == "__main__":
    main()