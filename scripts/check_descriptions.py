import difflib
import json


INPUT_FILE = "shl_catalog_clean.json"


def escape_control_characters(text):
    return (
        text.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def stripped_text(description, description_clean):
    matcher = difflib.SequenceMatcher(None, description, description_clean)
    removed_parts = []

    for tag, original_start, original_end, _clean_start, _clean_end in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            removed_parts.append(description[original_start:original_end])

    return escape_control_characters("".join(removed_parts))


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        products = json.load(input_file)

    affected_count = 0

    for product in products:
        if product["description_clean"] != product["description"]:
            affected_count += 1
            print(f'{product["entity_id"]} | {product["name"]}')
            print(
                "STRIPPED: "
                f'{stripped_text(product["description"], product["description_clean"])}'
            )
            print("---")

    print(f"Total affected products: {affected_count}")


if __name__ == "__main__":
    main()
