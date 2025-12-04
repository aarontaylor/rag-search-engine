import os
from typing import List

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import format_search_result


def _normalize_list(scores: List[float]) -> List[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        # All identical → treat all as 1.0
        return [1.0 for _ in scores]

    rng = max_score - min_score
    return [(s - min_score) / rng for s in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    """Weighted hybrid score: alpha * bm25 + (1 - alpha) * semantic."""
    return alpha * bm25_score + (1.0 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given rank and k."""
    return 1.0 / (k + rank)


class HybridSearch:
    """
    Hybrid search engine combining:
    - Keyword BM25 search (via InvertedIndex)
    - Semantic chunked search (via ChunkedSemanticSearch)
    """

    def __init__(self, documents: list[dict]):
        self.documents = documents
        # Map from doc_id to full movie object
        self.doc_map = {doc["id"]: doc for doc in documents}

        # Initialize semantic search (chunk-based)
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # Initialize keyword BM25 search
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        """Internal convenience method for invoking BM25 keyword search."""
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def _extract_bm25_doc_and_score(self, result):
        """
        Handle different possible BM25 result shapes.

        We expect something like either:
        - (movie_dict, score)
        - (score, movie_dict)

        We'll return (doc_id, score).
        """
        # If it's already a dict (future-proofing)
        if isinstance(result, dict):
            doc_id = result.get("id")
            score = result.get("score", 0.0)
            return doc_id, float(score)

        # If it's a tuple, figure out which side is the doc and which is the score.
        if isinstance(result, tuple) and len(result) == 2:
            a, b = result

            # (movie_dict, score)
            if isinstance(a, dict) and isinstance(b, (int, float)):
                doc = a
                score = float(b)
            # (score, movie_dict)
            elif isinstance(b, dict) and isinstance(a, (int, float)):
                doc = b
                score = float(a)
            else:
                # Unexpected shape, just bail out
                return None, 0.0

            doc_id = doc.get("id")
            return doc_id, score

        # Unknown shape
        return None, 0.0

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        """
        Hybrid search using weighted combination of normalized BM25 and semantic scores.

        Steps:
        - Run BM25 search and semantic chunked search (both with 500 * limit)
        - Normalize both sets of scores to [0, 1]
        - Combine with hybrid_score()
        - Return top `limit` results as formatted dicts
        """
        # Get plenty of candidates from each method
        bm25_limit = limit * 500
        semantic_limit = limit * 500

        bm25_raw_results = self._bm25_search(query, bm25_limit)
        semantic_results = self.semantic_search.search_chunks(query, semantic_limit)

        # --- Process BM25 results ---
        bm25_scores_by_id: dict = {}
        bm25_scores_list: list[float] = []
        bm25_ids_ordered: list = []

        for r in bm25_raw_results:
            doc_id, score = self._extract_bm25_doc_and_score(r)
            if doc_id is None:
                continue
            bm25_scores_by_id[doc_id] = score
            bm25_ids_ordered.append(doc_id)
            bm25_scores_list.append(score)

        bm25_norm_list = _normalize_list(bm25_scores_list)
        bm25_norm_by_id: dict = {}
        for doc_id, norm_score in zip(bm25_ids_ordered, bm25_norm_list):
            bm25_norm_by_id[doc_id] = norm_score

        # --- Process semantic results ---
        semantic_scores_by_id: dict = {}
        semantic_scores_list: list[float] = []
        semantic_ids_ordered: list = []

        for r in semantic_results:
            doc_id = r["id"]
            score = float(r["score"])
            semantic_scores_by_id[doc_id] = score
            semantic_ids_ordered.append(doc_id)
            semantic_scores_list.append(score)

        semantic_norm_list = _normalize_list(semantic_scores_list)
        semantic_norm_by_id: dict = {}
        for doc_id, norm_score in zip(semantic_ids_ordered, semantic_norm_list):
            semantic_norm_by_id[doc_id] = norm_score

        # --- Combine scores by doc_id ---
        all_doc_ids = set(bm25_norm_by_id.keys()) | set(semantic_norm_by_id.keys())

        combined_results: list[dict] = []
        for doc_id in all_doc_ids:
            bm25_norm = bm25_norm_by_id.get(doc_id, 0.0)
            semantic_norm = semantic_norm_by_id.get(doc_id, 0.0)

            h_score = hybrid_score(bm25_norm, semantic_norm, alpha)

            doc = self.doc_map.get(doc_id)
            if not doc:
                continue

            description = doc.get("description", "")
            short_desc = description[:100]

            result = format_search_result(
                doc_id=doc_id,
                title=doc.get("title", ""),
                document=short_desc,
                score=h_score,
                bm25=bm25_norm,
                semantic=semantic_norm,
            )
            combined_results.append(result)

        # Sort by hybrid score (stored in "score") descending
        combined_results.sort(key=lambda r: r["score"], reverse=True)

        # Only return top `limit`
        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 5) -> list[dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).

        Steps:
        - Run BM25 and semantic searches (500 * limit)
        - Use ranks (positions) instead of scores
        - For each doc, sum 1/(k+rank) from both lists
        - Return sorted by RRF score
        """
        bm25_limit = limit * 500
        semantic_limit = limit * 500

        bm25_raw_results = self._bm25_search(query, bm25_limit)
        semantic_results = self.semantic_search.search_chunks(query, semantic_limit)

        # doc_id -> info: {bm25_rank, semantic_rank, rrf_score}
        doc_info: dict[str, dict] = {}

        # BM25 ranks (1-based)
        for idx, r in enumerate(bm25_raw_results, start=1):
            doc_id, _ = self._extract_bm25_doc_and_score(r)
            if doc_id is None:
                continue

            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }

            info = doc_info[doc_id]
            # if we've seen it before, keep the best (lowest) rank
            if info["bm25_rank"] is None or idx < info["bm25_rank"]:
                info["bm25_rank"] = idx

            info["rrf_score"] += rrf_score(idx, k)

        # Semantic ranks (1-based)
        for idx, r in enumerate(semantic_results, start=1):
            doc_id = r["id"]

            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }

            info = doc_info[doc_id]
            if info["semantic_rank"] is None or idx < info["semantic_rank"]:
                info["semantic_rank"] = idx

            info["rrf_score"] += rrf_score(idx, k)

        # Build result list
        results: list[dict] = []
        for doc_id, info in doc_info.items():
            doc = self.doc_map.get(doc_id)
            if not doc:
                continue

            desc = doc.get("description", "")
            short_desc = desc[:100]

            result = format_search_result(
                doc_id=doc_id,
                title=doc.get("title", ""),
                document=short_desc,
                score=info["rrf_score"],
                bm25_rank=info["bm25_rank"],
                semantic_rank=info["semantic_rank"],
            )
            results.append(result)

        # Sort by RRF score descending
        results.sort(key=lambda r: r["score"], reverse=True)

        return results[:limit]
