import json
import re


INPUT_FILE = "shl_catalog.json"
OUTPUT_FILE = "shl_catalog_clean.json"
DROP_FIELDS = (
    "remote",
    "status",
    "scraped_at",
    "job_levels_raw",
    "languages_raw",
    "duration_raw",
)

BOILERPLATE_PHRASES = (
    "This tool has been validated",
    "In accordance with New York City Local Law 144",
    "Pursuant to New York City Local Law",
    "NYC Local Law 144",
)

DESCRIPTION_RESTORE_ENTITY_IDS = ("3746", "3472", "3942", "4205")


def dedupe_preserve_order(items):
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def parse_duration(duration):
    if duration == "0 minutes":
        return None, "unknown"
    if duration == "Untimed":
        return None, "untimed"
    if duration == "Variable":
        return None, "variable"

    match = re.search(r"\b(\d+)\s+minutes\b", duration)
    if match:
        return int(match.group(1)), "timed"

    return None, "unknown"


def strip_description_boilerplate(description):
    positions = [
        position
        for phrase in BOILERPLATE_PHRASES
        if (position := description.find(phrase)) != -1
    ]

    if not positions:
        return description

    return description[: min(positions)].rstrip()


def normalize_name(name):
    return re.sub(r"\s+", " ", name).strip()


def normalize_description(description):
    description = description.replace("\r\n", " ")
    description = description.replace("\r", " ").replace("\n", " ")
    description = re.sub(r" {2,}", " ", description)
    return description.strip()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        products = json.load(input_file, strict=False)

    for product in products:
        for field in DROP_FIELDS:
            del product[field]

        if product["entity_id"] == "4207":
            product["name"] = "Microsoft Excel 365 (New)"

        product["keys"] = dedupe_preserve_order(product["keys"])
        product["job_levels"] = dedupe_preserve_order(product["job_levels"])
        product["languages"] = dedupe_preserve_order(product["languages"])

        if product["entity_id"] == "726":
            product["keys"] = ["Personality & Behavior"]
        if product["entity_id"] == "759":
            product["keys"] = ["Personality & Behavior"]

        product["job_levels_all"] = product["job_levels"] == []
        product["languages_agnostic"] = product["languages"] == []

        duration_minutes, duration_category = parse_duration(product["duration"])
        product["duration_minutes"] = duration_minutes
        product["duration_category"] = duration_category

        product["description_clean"] = strip_description_boilerplate(
            product["description"]
        )
        if product["entity_id"] in DESCRIPTION_RESTORE_ENTITY_IDS:
            product["description_clean"] = product["description"]

        product["name"] = normalize_name(product["name"])
        product["description_clean"] = normalize_description(
            product["description_clean"]
        )

        product["adaptive_bool"] = product["adaptive"] == "yes"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        json.dump(products, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    job_levels_all_count = sum(product["job_levels_all"] for product in products)
    languages_agnostic_count = sum(
        product["languages_agnostic"] for product in products
    )
    duration_timed_count = sum(
        product["duration_category"] == "timed" for product in products
    )
    duration_unknown_count = sum(
        product["duration_category"] == "unknown" for product in products
    )
    description_changed_count = sum(
        product["description_clean"] != product["description"] for product in products
    )
    product_4207_name = next(
        product["name"] for product in products if product["entity_id"] == "4207"
    )

    print(f"Done. {len(products)} products written.")
    print("Validation summary:")
    print(f"job_levels_all is True: {job_levels_all_count}")
    print(f"languages_agnostic is True: {languages_agnostic_count}")
    print(f'duration_category == "timed": {duration_timed_count}')
    print(f'duration_category == "unknown": {duration_unknown_count}')
    print(f"description_clean != description: {description_changed_count}")
    print(f'entity_id == "4207" name: {product_4207_name}')
    print(
        "Patched entity_ids: 3746, 3472, 3942, 4205 — "
        "description_clean restored from original."
    )


if __name__ == "__main__":
    main()
