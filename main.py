
import argparse

from docmind.chunker import chunk_pages
from docmind.embeddings import DEFAULT_MODEL_NAME, embed_chunks
from docmind.pdf_reader import extract_pages
from docmind.search import search_chunks


def main() -> None:
    """Запускает обработку PDF и семантический поиск."""

    parser = argparse.ArgumentParser(
        description="DocMind — семантический поиск по PDF"
    )

    parser.add_argument(
        "pdf_path",
        help="Путь к PDF-документу",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Максимальный размер текстового фрагмента",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Количество перекрывающихся символов",
    )

    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Создать эмбеддинги документа",
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Поисковый запрос",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Максимальное количество результатов поиска",
    )

    args = parser.parse_args()

    # Проверяем поисковые параметры до загрузки модели
    if args.query is not None and not args.query.strip():
        parser.error("Поисковый запрос не должен быть пустым")

    if args.top_k < 1:
        parser.error("--top-k должен быть больше нуля")

    # Извлекаем текст документа и создаём фрагменты
    try:
        pages = extract_pages(args.pdf_path)

        chunks = chunk_pages(
            pages,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"\nОбработано страниц: {len(pages)}")
    print(f"Создано фрагментов: {len(chunks)}")

    # Выводим первые три фрагмента для диагностики
    for chunk in chunks[:3]:
        preview = chunk["text"][:160].replace("\n", " ")

        print(f"\n--- Фрагмент №{chunk['chunk_id']} ---")
        print(f"Страница: {chunk['page']}")
        print(f"Позиции: {chunk['start']}–{chunk['end']}")
        print(f"Текст: {preview}")

    # Если эмбеддинги и поиск не нужны, завершаем работу
    if not args.embeddings and args.query is None:
        return

    if not chunks:
        print("\nДокумент не содержит текста для поиска.")
        return

    # Загружаем предобученную модель
    from sentence_transformers import SentenceTransformer

    print("\nЗагружаем модель эмбеддингов...")
    encoder = SentenceTransformer(DEFAULT_MODEL_NAME)

    # Создаём эмбеддинги всех фрагментов
    embeddings = embed_chunks(chunks, encoder)

    print(f"Размер матрицы: {embeddings.shape}")

    # Если запрос отсутствует, показываем информацию о векторах
    if args.query is None:
        print(
            "Первые 5 значений:",
            embeddings[0][:5].round(4).tolist(),
        )
        return

    # Выполняем семантический поиск
    try:
        results = search_chunks(
            query=args.query,
            chunks=chunks,
            embeddings=embeddings,
            encoder=encoder,
            top_k=args.top_k,
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"\nРезультаты поиска: {args.query}")

    for position, result in enumerate(results, start=1):
        print(f"\n--- Результат №{position} ---")
        print(f"Страница: {result['page']}")
        print(f"Сходство: {result['score']:.4f}")
        print(f"Текст:\n{result['text']}")


if __name__ == "__main__":
    main()
