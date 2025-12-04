import os
import pickle
import string
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    BM25_K1,
    BM25_B,
)


class InvertedIndex:
    def __init__(self) -> None:
        # token -> set(doc_id)
        self.index: dict[str, set[int]] = defaultdict(set)
        # doc_id -> movie dict
        self.docmap: dict[int, dict] = {}
        # doc_id -> Counter(token -> count)
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        # doc_id -> document length (# of tokens)
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    # ---------- Index building ----------

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            text = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, text)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    # ---------- Index helpers ----------

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        # Save doc length
        self.doc_lengths[doc_id] = len(tokens)

        # Update TF and inverted index
        tf_counter = self.term_frequencies[doc_id]
        for token in tokens:
            tf_counter[token] += 1
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        return sorted(list(self.index.get(term, set())))

    # ---------- TF / IDF / BM25 ----------

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("get_tf expects exactly one token")
        token = tokens[0]

        counter = self.term_frequencies.get(doc_id)
        if counter is None:
            return 0
        return counter.get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("get_bm25_idf expects exactly one token")

        token = tokens[0]
        df = len(self.get_documents(token))
        N = len(self.docmap)

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str,
                     k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0

        doc_len = self.doc_lengths.get(doc_id, 0)
        avg_len = self.__get_avg_doc_length()

        if avg_len <= 0:
            length_norm = 1.0
        else:
            length_norm = 1 - b + b * (doc_len / avg_len)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    # ---------- BM25 score ----------

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        tokens = tokenize_text(query)
        scores: dict[int, float] = {}

        # Score every document
        for doc_id in self.docmap:
            total = 0.0
            for tok in tokens:
                total += self.bm25(doc_id, tok)
            scores[doc_id] = total

        # Sort by score desc
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top N: list of tuples (doc, score)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            results.append((self.docmap[doc_id], score))
        return results


# ---------- Commands for CLI ----------

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()

    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for tok in query_tokens:
        for doc_id in idx.get_documents(tok):
            if doc_id not in seen:
                seen.add(doc_id)
                results.append(idx.docmap[doc_id])
                if len(results) >= limit:
                    return results

    return results


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


# ---------- Tokenization ----------

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()

    stop_words = load_stopwords()
    tokens = [t for t in tokens if t and t not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens
