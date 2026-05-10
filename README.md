# SHL Assessment Recommendation Agent

Conversational agent that recommends SHL psychometric assessments to hiring managers.

## How it works

Takes a hiring manager's job requirements through a multi-turn conversation and returns a shortlist of SHL assessments. Hybrid retrieval (FAISS + BM25) over 377 catalog products handles the matching. Groq (llama-3.3-70b-versatile) extracts structured hiring intent and generates replies. An FSM manages conversation state: clarify when context is thin, recommend when confident, refine on edit requests, explain on comparison queries.

## Setup

```bash
git clone <repo>
pip install -r requirements.txt
```

Add a `.env` in the project root:

```
GROQ_API_KEYS=your_key1,your_key2
```

## Run

```bash
python main.py
```

Server starts on port 8000. Health check:

```
GET /health → {"status": "ok"}
```

## API

```
POST /chat
```

```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a senior Java developer with SQL and AWS skills"},
    {"role": "assistant", "content": "Here are my recommendations..."},
    {"role": "user", "content": "Remove the personality test"}
  ]
}
```

```json
{
  "reply": "Updated shortlist based on your request...",
  "recommendations": [
    {"name": "SHL Verify Interactive G+", "url": "https://www.shl.com/...", "test_type": "A"},
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is empty during clarification, 1-10 items once the agent commits to a shortlist. `end_of_conversation` is true when the agent considers the conversation resolved.

## Evaluation

```bash
python evaluation/run_eval.py
```

Replays 10 labeled conversation traces (C1-C10) against the live server and reports Recall@10 per trace.

Current: Mean Recall@10 = **0.98**

## Structure

```
main.py
app/
    agent.py
    context_extractor.py
    retrieval.py
data/
    shl_catalog_clean.json
    product_ontology.json
    product_roles.json
    relationship_map.json
evaluation/
    run_eval.py
    evaluate.py
    traces/
```
