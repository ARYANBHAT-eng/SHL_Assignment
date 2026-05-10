import json


CATALOG_FILE = "shl_catalog_clean.json"
PRODUCT_ROLES_FILE = "product_roles.json"
OUTPUT_FILE = "product_ontology.json"

PRODUCT_FAMILY_RULES = (
    ("Automata", ("Automata",)),
    ("Verify Interactive", ("Verify Interactive", "SHL Verify Interactive")),
    ("Verify", ("Verify",)),
    ("OPQ", ("OPQ",)),
    ("MQ", ("MQ", "Motivation Questionnaire")),
    ("Scenarios", ("Scenarios",)),
    ("SVAR", ("SVAR",)),
    ("MFS 360", ("MFS 360", "360\u00b0", "360 MFS")),
    ("HiPo", ("HiPo",)),
    ("RemoteWorkQ", ("RemoteWorkQ",)),
    ("Entry Level", ("Entry Level",)),
    ("Focus 8.0", ("8.0",)),
    ("SAP", ("SAP",)),
    ("WriteX", ("WriteX",)),
    ("Sales Transformation", ("Sales Transformation",)),
    (
        "Microsoft Office",
        (
            "Microsoft Excel",
            "Microsoft Word",
            "Microsoft PowerPoint",
            "Microsoft Outlook",
        ),
    ),
    (
        "Contact Center",
        (
            "Contact Center",
            "Customer Service Phone",
            "Sales & Service Phone",
            "Conversational Multichat",
        ),
    ),
    ("Global Skills", ("Global Skills",)),
    ("Digital Readiness", ("Digital Readiness",)),
    ("PJM", ("PJM",)),
    ("Enterprise Leadership", ("Enterprise Leadership",)),
)


def assign_product_family(product):
    name = product["name"]

    for family, patterns in PRODUCT_FAMILY_RULES:
        if any(pattern in name for pattern in patterns):
            return family

    if product["entity_id"] in {"4301", "4302"}:
        return "GSA"

    return "Independent"


def is_legacy_product(name, product_family):
    return (
        product_family != "Independent"
        and "(New)" not in name
        and "Interactive" not in name
        and "8.0" not in name
        and "2.0" not in name
        and "365" not in name
    )


def assign_version_preference(name):
    if (
        "Interactive" in name
        or "365" in name
        or "2.0" in name
        or "(New)" in name
        or "8.0" in name
    ):
        return 3
    if "1.0" in name:
        return 2
    return 1


def print_family_distribution(family_counts):
    distribution = sorted(
        family_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    print("Family distribution:")
    for family, count in distribution:
        print(f"{family}: {count}")


def main():
    with open(CATALOG_FILE, "r", encoding="utf-8") as input_file:
        products = json.load(input_file)

    with open(PRODUCT_ROLES_FILE, "r", encoding="utf-8") as input_file:
        product_roles = json.load(input_file)

    ontology = []
    family_counts = {}
    legacy_true_count = 0
    version_preference_counts = {1: 0, 2: 0, 3: 0}

    for product in products:
        entity_id = product["entity_id"]
        name = product["name"]
        product_family = assign_product_family(product)
        is_legacy = is_legacy_product(name, product_family)
        version_preference = assign_version_preference(name)
        is_report = product_roles.get(entity_id) == "report"

        ontology.append(
            {
                "entity_id": entity_id,
                "product_family": product_family,
                "is_legacy": is_legacy,
                "version_preference": version_preference,
                "is_report": is_report,
            }
        )

        family_counts[product_family] = family_counts.get(product_family, 0) + 1
        if is_legacy:
            legacy_true_count += 1
        version_preference_counts[version_preference] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        json.dump(ontology, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    total_products = len(products)
    matched_entity_ids = sum(
        1 for product in products if product["entity_id"] in product_roles
    )

    print(f"Total products: {total_products}")
    print_family_distribution(family_counts)
    print(f"is_legacy True count: {legacy_true_count}")
    print(f"is_legacy False count: {total_products - legacy_true_count}")
    print("version_preference distribution:")
    print(f"1: {version_preference_counts[1]}")
    print(f"2: {version_preference_counts[2]}")
    print(f"3: {version_preference_counts[3]}")
    print(
        "Confirm: product_roles.json loaded successfully and matched "
        f"{matched_entity_ids} entity_ids"
    )


if __name__ == "__main__":
    main()
