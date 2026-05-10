import json


INPUT_FILE = "shl_catalog_clean.json"
RELATIONSHIP_MAP_FILE = "relationship_map.json"
PRODUCT_ROLES_FILE = "product_roles.json"

ASSESSMENT_TO_REPORTS = {
    "720": [
        "727",
        "748",
        "749",
        "750",
        "752",
        "753",
        "754",
        "756",
        "757",
        "758",
        "759",
        "1050",
        "1058",
        "1059",
        "1060",
        "1061",
        "1067",
        "4286",
        "4289",
        "4298",
        "4300",
        "4307",
    ],
    "724": ["1048", "1306", "1308", "1309"],
    "731": ["1102"],
    "741": ["3900", "3901"],
    "742": ["3902", "3903", "3904"],
    "743": ["3849", "3899"],
    "744": ["3948", "3949", "3950", "3951", "4297"],
    "3908": ["3945", "3970"],
    "3971": ["3969", "3974", "3976"],
    "4204": ["4202", "4203"],
    "4301": ["4302"],
    "3845": ["4284", "4285"],
    "3856": ["4287"],
    "4230": ["4283"],
    "4233": ["4288"],
    "3484": ["3746"],
}

BUNDLE_TO_COMPONENTS = {
    "3931": ["3933"],
    "3930": ["3932"],
}

SIMULATION_NAME_TERMS = ("Simulation", "Automata", "WriteX", "SVAR", "Coding")
GUIDE_NAME_TERMS = ("Guide", "Cards", "Framework", "Profiler", "Planner")
ROLE_ORDER = ("report", "bundle", "simulation", "guide", "base_assessment")


def invert_mapping(parent_to_children):
    child_to_parent = {}

    for parent_id, child_ids in parent_to_children.items():
        for child_id in child_ids:
            child_to_parent[child_id] = parent_id

    return child_to_parent


def is_simulation(product):
    keys = product["keys"]
    name = product["name"]

    return (
        keys == ["Simulations"]
        or "Simulations" in keys
        and any(term in name for term in SIMULATION_NAME_TERMS)
    )


def is_guide(product):
    return any(term in product["name"] for term in GUIDE_NAME_TERMS)


def assign_role(product, report_to_assessment, bundle_to_components):
    entity_id = product["entity_id"]

    if entity_id in report_to_assessment:
        return "report"
    if entity_id in bundle_to_components:
        return "bundle"
    if is_simulation(product):
        return "simulation"
    if is_guide(product):
        return "guide"
    return "base_assessment"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        products = json.load(input_file)

    report_to_assessment = invert_mapping(ASSESSMENT_TO_REPORTS)
    component_to_bundle = invert_mapping(BUNDLE_TO_COMPONENTS)
    relationship_map = {
        "assessment_to_reports": ASSESSMENT_TO_REPORTS,
        "report_to_assessment": report_to_assessment,
        "bundle_to_components": BUNDLE_TO_COMPONENTS,
        "component_to_bundle": component_to_bundle,
    }

    product_roles = {
        product["entity_id"]: assign_role(
            product, report_to_assessment, BUNDLE_TO_COMPONENTS
        )
        for product in products
    }

    with open(RELATIONSHIP_MAP_FILE, "w", encoding="utf-8") as output_file:
        json.dump(relationship_map, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    with open(PRODUCT_ROLES_FILE, "w", encoding="utf-8") as output_file:
        json.dump(product_roles, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    catalog_entity_ids = {product["entity_id"] for product in products}
    missing_parents = [
        entity_id
        for entity_id in ASSESSMENT_TO_REPORTS
        if entity_id not in catalog_entity_ids
    ]
    missing_reports = [
        entity_id
        for report_ids in ASSESSMENT_TO_REPORTS.values()
        for entity_id in report_ids
        if entity_id not in catalog_entity_ids
    ]

    role_counts = {
        role: sum(assigned_role == role for assigned_role in product_roles.values())
        for role in ROLE_ORDER
    }

    print(f"Total products assigned a role: {len(product_roles)}")
    for role in ROLE_ORDER:
        print(f"{role}: {role_counts[role]}")
    print(f"assessment_to_reports entries: {len(ASSESSMENT_TO_REPORTS)}")
    print(f"report_to_assessment entries: {len(report_to_assessment)}")
    print(f"MISSING PARENTS: {', '.join(missing_parents) if missing_parents else 'none'}")
    print(f"MISSING REPORTS: {', '.join(missing_reports) if missing_reports else 'none'}")


if __name__ == "__main__":
    main()
