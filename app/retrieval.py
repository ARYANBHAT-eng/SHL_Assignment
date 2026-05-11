import contextlib
import json
import logging
import os
import re
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings(
    "ignore",
    message=".*Found Intel OpenMP.*",
    category=RuntimeWarning,
)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

with open(os.devnull, "w", encoding="utf-8") as _devnull:
    with contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
        from sentence_transformers import SentenceTransformer


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CATALOG_FILE = os.path.join(_DATA_DIR, "shl_catalog_clean.json")
ONTOLOGY_FILE = os.path.join(_DATA_DIR, "product_ontology.json")
PRODUCT_ROLES_FILE = os.path.join(_DATA_DIR, "product_roles.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _load_catalog() -> list[dict]:
    with open(CATALOG_FILE, "r", encoding="utf-8") as input_file:
        catalog = json.load(input_file)

    with open(ONTOLOGY_FILE, "r", encoding="utf-8") as input_file:
        ontology = json.load(input_file)

    with open(PRODUCT_ROLES_FILE, "r", encoding="utf-8") as input_file:
        product_roles = json.load(input_file)

    ontology_by_id = {item["entity_id"]: item for item in ontology}
    merged_catalog = []

    for product in catalog:
        entity_id = product["entity_id"]
        ontology_item = ontology_by_id[entity_id]
        merged_product = dict(product)
        merged_product["product_family"] = ontology_item["product_family"]
        merged_product["is_legacy"] = ontology_item["is_legacy"]
        merged_product["version_preference"] = ontology_item["version_preference"]
        merged_product["is_report"] = ontology_item["is_report"]
        merged_product["role"] = product_roles[entity_id]
        merged_catalog.append(merged_product)

    return merged_catalog


def _run_quietly(function, *args, **kwargs):
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return function(*args, **kwargs)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32", copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = _run_quietly(
        EMBEDDING_MODEL.encode,
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return _normalize_vectors(embeddings)


def _build_faiss_index(catalog: list[dict]) -> faiss.IndexFlatIP:
    descriptions = [product["description_clean"] for product in catalog]
    embeddings = _embed_texts(descriptions)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def _build_bm25_index(catalog: list[dict]) -> tuple[list[list[str]], BM25Okapi]:
    tokenized_corpus = [
        _tokenize(f'{product["name"]} {product["description_clean"]}')
        for product in catalog
    ]
    return tokenized_corpus, BM25Okapi(tokenized_corpus)


def _top_bm25_positions(scores: np.ndarray, limit: int) -> list[int]:
    if limit >= len(scores):
        return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    candidate_positions = np.argpartition(scores, -limit)[-limit:]
    return sorted(candidate_positions, key=lambda index: scores[index], reverse=True)


def _normalize_score_map(scores_by_position: dict[int, float]) -> dict[int, float]:
    if not scores_by_position:
        return {}

    max_score = max(scores_by_position.values())
    if max_score <= 0:
        return {position: 0.0 for position in scores_by_position}

    return {
        position: max(0.0, score / max_score)
        for position, score in scores_by_position.items()
    }


def _shares_any(product_values: list[str], filter_values: list[str]) -> bool:
    return bool(set(product_values) & set(filter_values))


def _passes_filters(
    product: dict,
    filter_keys: list[str] | None,
    filter_job_levels: list[str] | None,
    filter_languages: list[str] | None,
    exclude_roles: list[str] | None,
) -> bool:
    if filter_keys is not None and not _shares_any(product["keys"], filter_keys):
        return False

    if filter_job_levels is not None and not (
        product["job_levels_all"]
        or _shares_any(product["job_levels"], filter_job_levels)
    ):
        return False

    if filter_languages is not None and not (
        product["languages_agnostic"]
        or _shares_any(product["languages"], filter_languages)
    ):
        return False

    if exclude_roles is not None and product["role"] in exclude_roles:
        return False

    return True


def _apply_version_boost(product: dict, score: float) -> float:
    boost = 1 + (product["version_preference"] - 1) * 0.1
    return score * boost


def _apply_legacy_penalty(product: dict, score: float) -> float:
    if product["is_legacy"]:
        return score * 0.80
    return score


def _format_result(product: dict, score: float) -> dict:
    return {
        "entity_id": product["entity_id"],
        "name": product["name"],
        "link": product["link"],
        "keys": product["keys"],
        "job_levels": product["job_levels"],
        "languages": product["languages"],
        "duration_category": product["duration_category"],
        "duration_minutes": product["duration_minutes"],
        "product_family": product["product_family"],
        "is_report": product["is_report"],
        "version_preference": product["version_preference"],
        "role": product["role"],
        "score": round(float(score), 4),
    }


def hybrid_search(
    query: str,
    top_k: int = 20,
    filter_keys: list[str] | None = None,
    filter_job_levels: list[str] | None = None,
    filter_languages: list[str] | None = None,
    exclude_roles: list[str] | None = None,
) -> list[dict]:
    search_limit = min(len(CATALOG), max(top_k * 5, top_k))

    query_embedding = _embed_texts([query])
    faiss_scores, faiss_positions = FAISS_INDEX.search(query_embedding, search_limit)
    faiss_score_map = {
        int(position): float(score)
        for position, score in zip(faiss_positions[0], faiss_scores[0])
        if position >= 0
    }

    query_tokens = _tokenize(query)
    bm25_scores = BM25_INDEX.get_scores(query_tokens)
    bm25_positions = _top_bm25_positions(bm25_scores, search_limit)
    bm25_score_map = {
        int(position): float(bm25_scores[position])
        for position in bm25_positions
    }

    normalized_faiss_scores = _normalize_score_map(faiss_score_map)
    normalized_bm25_scores = _normalize_score_map(bm25_score_map)
    candidate_positions = set(normalized_faiss_scores) | set(normalized_bm25_scores)

    scored_products = []
    for position in candidate_positions:
        product = CATALOG[position]
        if not _passes_filters(
            product,
            filter_keys,
            filter_job_levels,
            filter_languages,
            exclude_roles,
        ):
            continue

        combined_score = (
            0.6 * normalized_faiss_scores.get(position, 0.0)
            + 0.4 * normalized_bm25_scores.get(position, 0.0)
        )
        final_score = _apply_version_boost(product, combined_score)
        final_score = _apply_legacy_penalty(product, final_score)
        scored_products.append((final_score, position, product))

    scored_products.sort(key=lambda item: (-item[0], item[1]))

    return [
        _format_result(product, score)
        for score, _position, product in scored_products[:top_k]
    ]


def get_product_by_id(entity_id: str) -> dict | None:
    return PRODUCT_BY_ID.get(entity_id)


CATALOG = _load_catalog()
PRODUCT_BY_ID = {product["entity_id"]: product for product in CATALOG}
EMBEDDING_MODEL = _run_quietly(SentenceTransformer, MODEL_NAME)
FAISS_INDEX = _build_faiss_index(CATALOG)
TOKENIZED_CORPUS, BM25_INDEX = _build_bm25_index(CATALOG)


if __name__ == "__main__":
    test_queries = (
        "Java developer mid level stakeholder communication",
        "contact centre entry level English speaking",
        "graduate numerical reasoning personality",
        "safety critical industrial plant operator",
    )

    for query_index, test_query in enumerate(test_queries):
        print(test_query)
        for result in hybrid_search(test_query, top_k=5):
            print(
                f'{result["name"]} | {result["score"]} | '
                f'{result["role"]} | {result["product_family"]}'
            )
        if query_index != len(test_queries) - 1:
            print("---")
