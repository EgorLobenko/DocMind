import argparse

from docmind.pdf_reader import extract_pages


def main() -> None:
    """Запускает чтение PDF и выводит результат."""

    parser = argparse.ArgumentParser(
        description="DocMind — извлечение текста из PDF"
    )

    parser.add_argument(
        "pdf_path",
        help="Путь к PDF-документу"
    )

    args = parser.parse_args()

    # Получаем текст всех страниц документа
    try:
        pages = extract_pages(args.pdf_path)
    except FileNotFoundError as error:
        parser.error(str(error))

    print(f"\nОбработано страниц: {len(pages)}")

    # Показываем краткую информацию о каждой странице
    for page in pages:
        text = page["text"]

        print(f"\n--- Страница {page['page']} ---")
        print(f"Количество символов: {len(text)}")

        if text:
            print(f"Фрагмент текста: {text[:200]}")
        else:
            print("Текст не обнаружен.")


if __name__ == "__main__":
    main()