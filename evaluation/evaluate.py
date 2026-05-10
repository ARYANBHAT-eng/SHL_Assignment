import os
import re
import sys

import requests

EXPECTED = {
    "C1":  ["Occupational Personality Questionnaire OPQ32r", "OPQ Universal Competency Report 2.0", "OPQ Leadership Report"],
    "C2":  ["Smart Interview Live Coding", "Linux Programming (General)", "Networking and Implementation (New)", "SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
    "C3":  ["SVAR - Spoken English (US) (New)", "Contact Center Call Simulation (New)", "Entry Level Customer Serv-Retail & Contact Center", "Customer Service Phone Simulation"],
    "C4":  ["SHL Verify Interactive – Numerical Reasoning", "Financial Accounting (New)", "Basic Statistics (New)", "Graduate Scenarios", "Occupational Personality Questionnaire OPQ32r"],
    "C5":  ["Global Skills Assessment", "Global Skills Development Report", "Occupational Personality Questionnaire OPQ32r", "OPQ MQ Sales Report", "Sales Transformation 2.0 - Individual Contributor"],
    "C6":  ["Manufac. & Indust. - Safety & Dependability 8.0", "Workplace Health and Safety (New)"],
    "C7":  ["HIPAA (Security)", "Medical Terminology (New)", "Microsoft Word 365 - Essentials (New)", "Dependability and Safety Instrument (DSI)", "Occupational Personality Questionnaire OPQ32r"],
    "C8":  ["Microsoft Excel 365 (New)", "Microsoft Word 365 (New)", "MS Excel (New)", "MS Word (New)", "Occupational Personality Questionnaire OPQ32r"],
    "C9":  ["Core Java (Advanced Level) (New)", "Spring (New)", "SQL (New)", "Amazon Web Services (AWS) Development (New)", "Docker (New)", "SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
    "C10": ["SHL Verify Interactive G+", "Graduate Scenarios"],
}

BASE_URL = "http://localhost:8000"


def check_server():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            print(f"WARNING: /health returned {r.status_code}. Is the server running?")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("WARNING: Cannot connect to server at http://localhost:8000. Start main.py first.")
        sys.exit(1)


def detect_trace_id(filename: str) -> str | None:
    name = os.path.splitext(os.path.basename(filename))[0].upper()
    # Match C10 before C1 to avoid prefix collision
    for key in ["C10", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]:
        if key in name:
            return key
    return None


def parse_user_turns(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    turns = []
    in_user_block = False
    current_lines = []

    for line in lines:
        stripped = line.rstrip("\n")
        if re.search(r"\*\*User\*\*", stripped):
            in_user_block = True
            current_lines = []
            continue
        if in_user_block:
            if stripped.startswith("> ") or stripped == ">":
                current_lines.append(stripped.lstrip("> ").strip())
            elif stripped.startswith("**") or stripped.startswith("_") or stripped.startswith("###"):
                if current_lines:
                    turns.append(" ".join(current_lines).strip())
                    current_lines = []
                in_user_block = False
            # skip blank lines inside block
        if not in_user_block and current_lines:
            turns.append(" ".join(current_lines).strip())
            current_lines = []

    if current_lines:
        turns.append(" ".join(current_lines).strip())

    return [t for t in turns if t]


def recall_at_10(predicted: list[dict], expected: list[str]) -> float:
    pred_names = {p["name"].lower().strip() for p in predicted[:10]}
    exp_names = {e.lower().strip() for e in expected}
    if not exp_names:
        return 0.0
    return len(pred_names & exp_names) / len(exp_names)


def replay_trace(user_turns: list[str]) -> list[dict]:
    messages = []
    last_recs = []

    for turn in user_turns:
        messages.append({"role": "user", "content": turn})
        try:
            r = requests.post(
                f"{BASE_URL}/chat",
                json={"messages": messages},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ERROR calling /chat: {e}")
            break

        recs = data.get("recommendations", [])
        if recs:
            last_recs = recs

        messages.append({"role": "assistant", "content": data.get("reply", "")})

        if data.get("end_of_conversation"):
            break

    return last_recs


def evaluate_trace(filepath: str, trace_id: str) -> float:
    expected = EXPECTED[trace_id]
    user_turns = parse_user_turns(filepath)
    predicted = replay_trace(user_turns)

    score = recall_at_10(predicted, expected)

    pred_names = [p["name"] for p in predicted[:10]]
    exp_set = {e.lower().strip() for e in expected}
    missing = [e for e in expected if e.lower().strip() not in {n.lower().strip() for n in pred_names}]

    print(f"{trace_id}: Recall@10 = {score:.2f} | Got: {pred_names} | Missing: {missing}")
    return score


def main():
    folder = input("Enter path to trace folder: ").strip().strip('"')
    check_server()

    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".md")],
        key=lambda p: (
            int(re.search(r"\d+", os.path.basename(p)).group())
            if re.search(r"\d+", os.path.basename(p)) else 0
        ),
    )

    scores = []
    for filepath in files:
        trace_id = detect_trace_id(filepath)
        if trace_id is None or trace_id not in EXPECTED:
            print(f"Skipping {filepath} — no matching trace ID")
            continue
        score = evaluate_trace(filepath, trace_id)
        scores.append(score)

    if scores:
        print(f"\nMean Recall@10: {sum(scores) / len(scores):.2f} across {len(scores)} traces")


if __name__ == "__main__":
    main()
