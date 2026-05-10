import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import check_server, detect_trace_id, evaluate_trace, EXPECTED

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
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
