import json
from pathlib import Path


INPUT_FILE = Path(__file__).with_name("shl_catalog_clean.json")


def print_product(product, fields):
    parts = [f"{field}: {product.get(field)}" for field in fields]
    print(", ".join(parts))


def only_general_population(job_levels):
    return (
        isinstance(job_levels, list)
        and len(job_levels) > 0
        and all(job_level == "General Population" for job_level in job_levels)
    )


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        products = json.load(input_file)

    job_levels_all_products = [
        product for product in products if product.get("job_levels_all") is True
    ]
    general_population_only_products = [
        product
        for product in products
        if only_general_population(product.get("job_levels"))
    ]

    print("List 1 — Products where job_levels_all is True:")
    for product in job_levels_all_products:
        print_product(product, ("entity_id", "name", "keys"))
    print(f"Count: {len(job_levels_all_products)}")

    print("List 2 — Products where job_levels contains only General Population:")
    for product in general_population_only_products:
        print_product(product, ("entity_id", "name", "job_levels", "keys"))
    print(f"Count: {len(general_population_only_products)}")


if __name__ == "__main__":
    main()
