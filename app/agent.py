import json
import os

import itertools as _itertools

from dotenv import load_dotenv
from groq import Groq

from app.context_extractor import extract_context
from app.retrieval import get_product_by_id, hybrid_search

load_dotenv()

_GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",") if k.strip()]
if not _GROQ_KEYS:
    raise RuntimeError("No Groq API keys found. Set GROQ_API_KEYS or GROQ_API_KEY in .env")
_CLIENTS = [Groq(api_key=k) for k in _GROQ_KEYS]
_client_cycle = _itertools.cycle(_CLIENTS)
_MODEL_ID = "llama-3.3-70b-versatile"

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

with open(os.path.join(_DATA_DIR, "relationship_map.json"), "r", encoding="utf-8") as _f:
    _rel = json.load(_f)
_REPORT_TO_ASSESSMENT: dict[str, str] = _rel["report_to_assessment"]

_KEY_PRIORITY = [
    ("Ability & Aptitude", "A"),
    ("Personality & Behavior", "P"),
    ("Knowledge & Skills", "K"),
    ("Simulations", "S"),
    ("Biodata & Situational Judgment", "B"),
    ("Competencies", "C"),
    ("Development & 360", "D"),
    ("Assessment Exercises", "E"),
]

_SENIORITY_TO_JOB_LEVELS: dict[str, list[str]] = {
    "entry":     ["Entry-Level"],
    "graduate":  ["Graduate"],
    "mid":       ["Mid-Professional", "Professional Individual Contributor"],
    "senior":    ["Manager", "Professional Individual Contributor", "Mid-Professional"],
    "executive": ["Director", "Executive"],
}

_SVAR_LANGUAGE_MAP: dict[str, str] = {
    "English (USA)": "3987",
    "English":       "3987",
    "US":            "3987",
    "English (UK)":  "4217",
    "UK":            "4217",
    "English (Australia)": "4216",
    "English (India)":     "3986",
    "Indian":              "3986",
    "French (Canada)":     "4197",
    "French":              "4198",
    "Spanish":             "4200",
    "Spanish (Castilian)": "4199",
}

_REFUSING_REPLY = (
    "I can only help with SHL assessment selection and comparison. "
    "For legal, compliance, or general HR questions, please consult the appropriate professional."
)

_LLM_SYSTEM_PROMPT = (
    "You are a concise SHL assessment consultant. You help hiring managers select the right assessments. "
    "Be direct and transactional. No conversational filler. No excessive empathy. "
    "Maximum 3 sentences in your reply. Never mention product URLs. Never invent assessments. "
    "Only refer to products that are explicitly in the provided shortlist. "
    "Never suggest, name, or imply any product that is not in the shortlist."
)


def _derive_test_type(keys: list[str]) -> str:
    for key, code in _KEY_PRIORITY:
        if key in keys:
            return code
    return "K"


def _recalc_confidence(ctx: dict) -> float:
    # Deterministic confidence from extracted fields
    score = 0.0
    if ctx.get("role"):
        score += 0.35
    if ctx.get("seniority"):
        score += 0.20
    if ctx.get("assessment_purpose"):
        score += 0.15
    if ctx.get("skills") or ctx.get("jd_provided"):
        score += 0.15
    if ctx.get("languages") or not ctx.get("voice_role", False):
        score += 0.15
    if ctx.get("voice_role") and not ctx.get("languages"):
        score = min(score, 0.59)
    return round(score, 2)


def _detect_state(ctx: dict, messages: list[dict]) -> str:
    user_turn_count = sum(1 for m in messages if m["role"] == "user")
    # Hard cap: force recommendation after 4 user turns
    if user_turn_count >= 4:
        return "RECOMMENDING"
    # Scope guard: only SHL assessment topics
    if ctx["off_topic"]:
        return "REFUSING"
    # Require minimum context before first recommendation
    if len(messages) == 1:
        if not ctx.get("role") or (not ctx.get("seniority") and not ctx.get("skills")):
            return "CLARIFYING"
    if ctx["confidence_score"] < 0.60:
        return "CLARIFYING"
    if ctx["comparison_request"]:
        return "EXPLAINING"
    # Detect refinement requests after initial recommendation
    if ctx["refinement_action"] is not None and len(messages) >= 3:
        return "REFINING"
    return "RECOMMENDING"


def _detect_archetype(ctx: dict) -> str:
    if ctx["seniority"] == "executive":
        return "executive"
    if ctx["seniority"] == "graduate":
        return "graduate"
    if ctx["assessment_purpose"] in ["development", "audit"]:
        return "development"
    if ctx["industry"] in ["industrial", "manufacturing", "chemical", "plant"]:
        return "safety_industrial"
    role_lower = (ctx["role"] or "").lower()
    if ctx["voice_role"] or "contact centre" in role_lower or "contact center" in role_lower:
        return "frontline_entry"
    if ctx["seniority"] in ["senior", "mid"] and len(ctx["skills"]) > 0:
        return "senior_ic_technical"
    if ctx["seniority"] == "entry" and (
        ctx["voice_role"]
        or "contact centre" in (ctx["role"] or "").lower()
        or "contact center" in (ctx["role"] or "").lower()
        or "customer service" in (ctx["role"] or "").lower()
    ):
        return "frontline_entry"
    return "general"


def _build_anchor_ids(ctx: dict, archetype: str) -> list[str]:
    anchor_ids: list[str] = []

    if archetype not in ("frontline_entry", "safety_industrial"):
        anchor_ids.append("720")  # OPQ32r

    if archetype == "executive" and ctx["assessment_purpose"] == "selection":
        anchor_ids.append("749")   # OPQ Leadership Report
        anchor_ids.append("4289")  # OPQ Universal Competency Report 2.0

    if archetype in ("senior_ic_technical", "graduate"):
        anchor_ids.append("3971")  # Verify G+

    if archetype == "senior_ic_technical":
        anchor_ids.append("4218")  # Smart Interview Live Coding

    if archetype == "graduate":
        anchor_ids.append("741")  # Graduate Scenarios
        from app.retrieval import CATALOG
        for p in CATALOG:
            if "basic statistics" in p["name"].lower():
                anchor_ids.append(p["entity_id"])
                break

    if archetype == "senior_ic_technical":
        skills_lower = [s.lower() for s in ctx["skills"]]
        if any("java" in s for s in skills_lower):
            from app.retrieval import CATALOG
            for p in CATALOG:
                if "core java (advanced level)" in p["name"].lower():
                    anchor_ids.append(p["entity_id"])
                    break
        if any(s == "sql" or s.startswith("sql") for s in skills_lower):
            from app.retrieval import CATALOG
            for p in CATALOG:
                if p["name"].lower() == "sql (new)":
                    anchor_ids.append(p["entity_id"])
                    break

    if archetype == "development":
        anchor_ids.extend(["4301", "4302"])  # GSA + GSA Development Report
        role_lower = (ctx["role"] or "").lower()
        if any(kw in role_lower for kw in ["sales", "account", "revenue", "business development"]):
            anchor_ids.extend(["754", "4283"])  # OPQ MQ Sales Report + Sales Transformation 2.0

    if archetype == "safety_industrial":
        from app.retrieval import CATALOG
        for p in CATALOG:
            if "workplace health and safety" in p["name"].lower():
                anchor_ids.append(p["entity_id"])
                break

    if ctx["voice_role"] and ctx["languages"] and any(
        kw in (ctx["role"] or "").lower()
        for kw in ["contact centre", "contact center", "customer service phone", "call centre", "call center", "spoken", "voice"]
    ):
        svar_id = _SVAR_LANGUAGE_MAP.get(ctx["languages"][0])
        if svar_id:
            anchor_ids.append(svar_id)

    if archetype == "frontline_entry" and ctx["voice_role"]:
        anchor_ids.append("4189")  # Contact Center Call Simulation (New)
        anchor_ids.append("3933")  # Customer Service Phone Simulation
        anchor_ids.append("3939")  # Entry Level Customer Serv-Retail & Contact Center

    if ctx["industry"] in ("healthcare", "medical") or "hipaa" in (ctx["role"] or "").lower() or "hipaa" in " ".join(ctx["skills"]).lower():
        anchor_ids.append("731")  # Dependability and Safety Instrument (DSI)

    if ctx["industry"] in ("healthcare", "medical") or any(
        kw in (ctx["role"] or "").lower()
        for kw in ["healthcare", "medical", "patient", "hipaa", "clinical"]
    ):
        from app.retrieval import CATALOG
        for p in CATALOG:
            if "hipaa" in p["name"].lower():
                anchor_ids.append(p["entity_id"])
                break
        for p in CATALOG:
            if "medical terminology" in p["name"].lower():
                anchor_ids.append(p["entity_id"])
                break

    role_lower = (ctx["role"] or "").lower()
    skills_lower_str = " ".join(ctx["skills"]).lower()
    if any(kw in role_lower for kw in ["admin", "administrator", "assistant", "clerical"]) or \
       any(kw in skills_lower_str for kw in ["excel", "word", "office"]):
        from app.retrieval import CATALOG
        office_targets = [
            "microsoft excel 365 (new)",
            "microsoft word 365 (new)",
            "microsoft word 365 - essentials (new)",
            "ms excel (new)",
            "ms word (new)",
        ]
        for target in office_targets:
            for p in CATALOG:
                if p["name"].lower() == target:
                    anchor_ids.append(p["entity_id"])
                    break

    return anchor_ids


def _build_shortlist(ctx: dict, archetype: str) -> list[dict]:
    anchor_ids = _build_anchor_ids(ctx, archetype)
    anchor_id_set = set(anchor_ids)

    anchor_products = [p for eid in anchor_ids if (p := get_product_by_id(eid))]

    if archetype == "safety_industrial":
        query = "safety industrial workplace plant operator dependability"
    elif ctx["industry"] in ("healthcare", "medical", "hospital") or any(
        kw in (ctx["role"] or "").lower()
        for kw in ["healthcare", "medical", "hospital", "clinical", "patient", "admin", "hipaa", "health"]
    ):
        skills_str = " ".join(ctx["skills"])
        query = f"HIPAA medical terminology healthcare compliance {ctx['role'] or ''} {skills_str}".strip()
    elif any(kw in (ctx["role"] or "").lower() for kw in ["admin", "administrator", "assistant", "clerical", "office"]) or any(kw in " ".join(ctx["skills"]).lower() for kw in ["excel", "word", "office", "powerpoint"]):
        skills_str = " ".join(ctx["skills"])
        query = f"Microsoft Office 365 Excel Word admin assistant {skills_str}".strip()
    else:
        query_parts = [ctx["role"] or ""] + ctx["skills"]
        query = " ".join(p for p in query_parts if p).strip() or "assessment"

    seniority = ctx["seniority"]
    filter_job_levels = _SENIORITY_TO_JOB_LEVELS.get(seniority) if seniority else None
    if archetype == "frontline_entry":
        filter_job_levels = None  # entry-level products have inconsistent job_levels data
    filter_languages = ctx["languages"] if ctx["languages"] else None
    if archetype == "development":
        top_domain = 3
    elif archetype == "senior_ic_technical":
        top_domain = 9
    else:
        top_domain = 5

    retrieved = hybrid_search(
        query=query,
        top_k=20,
        filter_job_levels=filter_job_levels,
        filter_languages=filter_languages,
        exclude_roles=["report", "guide"],
    )

    seniority = ctx["seniority"]
    if seniority in ("mid", "senior", "executive"):
        retrieved = [
            p for p in retrieved
            if "entry level" not in p["name"].lower()
        ]

    domain_products: list[dict] = []
    for p in retrieved:
        if p["entity_id"] not in anchor_id_set:
            domain_products.append(p)
        if len(domain_products) >= top_domain:
            break

    if archetype == "senior_ic_technical" and ctx["skills"]:
        skills_lower = [s.lower() for s in ctx["skills"]]
        filtered = [
            p for p in domain_products
            if any(skill in p["name"].lower() for skill in skills_lower)
        ]
        if len(filtered) >= 3:
            domain_products = filtered[:top_domain]
        # else keep original domain_products unfiltered

    shortlist = anchor_products + domain_products
    if len(shortlist) > 10:
        shortlist = anchor_products[:10] + domain_products[:max(0, 10 - len(anchor_products))]

    return shortlist


def _apply_refinements(shortlist: list[dict], ctx: dict) -> list[dict]:
    action = ctx["refinement_action"]
    targets = ctx["refinement_target"]

    if action == "remove" and targets:
        targets_lower = [t.lower() for t in targets]
        shortlist = [
            p for p in shortlist
            if not any(t in p["name"].lower() for t in targets_lower)
        ]

    elif action == "add" and targets:
        existing_ids = {p["entity_id"] for p in shortlist}
        results = hybrid_search(query=targets[0], top_k=3, exclude_roles=["report"])
        for p in results:
            if p["entity_id"] not in existing_ids:
                shortlist.append(p)
                break

    return shortlist


def _validate_shortlist(shortlist: list[dict]) -> list[dict]:
    shortlist_ids = {p["entity_id"] for p in shortlist}
    return [
        p for p in shortlist
        if p["entity_id"] not in _REPORT_TO_ASSESSMENT
        or _REPORT_TO_ASSESSMENT[p["entity_id"]] in shortlist_ids
    ]


def _llm_reply(user_prompt: str) -> str:
    response = next(_client_cycle).chat.completions.create(
        model=_MODEL_ID,
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()


def process_chat(messages: list[dict]) -> dict:
    ctx = extract_context(messages)
    ctx["confidence_score"] = _recalc_confidence(ctx)
    state = _detect_state(ctx, messages)
    archetype = _detect_archetype(ctx)

    shortlist_products: list[dict] = []
    if state in ("RECOMMENDING", "REFINING", "EXPLAINING"):
        shortlist_products = _build_shortlist(ctx, archetype)
        if state == "REFINING":
            shortlist_products = _apply_refinements(shortlist_products, ctx)
        shortlist_products = _validate_shortlist(shortlist_products)

    if state == "REFUSING":
        reply = _REFUSING_REPLY
    elif state == "CLARIFYING":
        reply = ctx["clarification_question"] or "Could you tell me more about the role you're hiring for?"
    elif state == "EXPLAINING":
        user_prompt = (
            f"The user asked: '{messages[-1]['content']}'. "
            f"Answer using only this catalog context: "
            f"{[p['name'] + ': ' + p.get('description_clean', '') for p in shortlist_products]}"
        )
        reply = _llm_reply(user_prompt)
    elif state == "RECOMMENDING":
        user_prompt = (
            f"Confirm this shortlist for {ctx['role']} ({ctx['seniority']} level): "
            f"{[p['name'] for p in shortlist_products]}"
        )
        reply = _llm_reply(user_prompt)
    else:  # REFINING
        user_prompt = (
            f"Acknowledge this update to the shortlist: "
            f"{ctx['refinement_action']} {ctx['refinement_target']}. "
            f"Updated list: {[p['name'] for p in shortlist_products]}"
        )
        reply = _llm_reply(user_prompt)

    end_of_conversation = state in ("RECOMMENDING", "REFINING") and len(messages) >= 6

    return {
        "reply": reply,
        "recommendations": [
            {
                "name": p["name"],
                "url": p["link"],
                "test_type": _derive_test_type(p["keys"]),
            }
            for p in shortlist_products
        ],
        "end_of_conversation": end_of_conversation,
    }


if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "I am hiring a mid-level Java developer who works with stakeholders"},
        {"role": "assistant", "content": "What specific Java frameworks or skills are required?"},
        {"role": "user", "content": "Spring Boot, SQL, and they need to communicate with business teams"},
    ]
    result = process_chat(test_messages)
    print(result)
