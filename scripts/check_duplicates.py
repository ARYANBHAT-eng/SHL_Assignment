import json
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path(__file__).with_name("shl_catalog.json")


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        products = json.load(input_file, strict=False)

    products_by_entity_id = defaultdict(list)
    for product in products:
        products_by_entity_id[product.get("entity_id")].append(
            product.get("name", "<missing name>")
        )

    duplicate_entity_ids = {
        entity_id: names
        for entity_id, names in products_by_entity_id.items()
        if len(names) > 1
    }

    if not duplicate_entity_ids:
        print(f"No duplicate entity_ids found. Total products: {len(products)}")
        return

    for entity_id, names in duplicate_entity_ids.items():
        print(f"entity_id: {entity_id}")
        for name in names:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
