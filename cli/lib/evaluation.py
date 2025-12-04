from .hybrid_search import HybridSearch
from .search_utils import (
    load_golden_dataset,
    load_movies,
)
from .semantic_search import SemanticSearch


def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k if k > 0 else 0.0


def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    """
    Recall@k = (# of relevant docs retrieved in top k) / (total # of relevant docs)

    If there are no relevant docs in the golden dataset, recall is defined as 0.0.
    """
    if not relevant_docs:
        return 0.0

    top_k = retrieved_docs[:k]
    relevant_retrieved = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_retrieved += 1

    return relevant_retrieved / len(relevant_docs)


def f1_score(precision: float, recall: float) -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall)

    If both precision and recall are 0, F1 is defined as 0.0.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_command(limit: int = 5) -> dict:
    movies = load_movies()
    golden_data = load_golden_dataset()
    test_cases = golden_data["test_cases"]

    # Ensure embeddings are warm/ready (even though HybridSearch handles its own setup)
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    results_by_query = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])

        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)

        retrieved_docs: list[str] = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)
        f1 = f1_score(precision, recall)

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs),
        }

    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
    }
