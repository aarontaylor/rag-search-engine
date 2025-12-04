import os
import pickle
import string

from nltk.stem import PorterStemmer

from .search_utils import load_movies, load_stopwords, PROJECT_ROOT

# Paths for cache files
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")

# Normalization helpers
_TRANSLATOR = str.maketrans("", "", string.punctuation)
STOPWORDS = set(load_stopwords())
STEMMER = PorterStemmer()


def process_text_to_tokens(text: str) -> list[str]:
    """
    Full pipeline:
    - lowercase
    - remove punctuation
    - split on whitespace
    - drop empty tokens
    - remove stopwords
    - stem tokens
    """
    normalized = text.lower().translate(_TRANSLATOR)
    tokens = [t for t in normalized.split() if t]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return [STEMMER.stem(t) for t in tokens]


class InvertedIndex:
    def __init__(self) -> None:
        # token -> set(doc_id)
        self.index: dict[str, set[int]] = {}
        # doc_id -> full movie dict
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize the text and add mappings token -> doc_id."""
        tokens = process_text_to_tokens(text)
        for tok in tokens:
            if tok not in self.index:
                self.index[tok] = set()
            self.index[tok].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        """
        Return sorted list of doc IDs for a given *processed* token.
        (Assumes the caller already ran process_text_to_tokens on the query.)
        """
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self) -> None:
        """Build index + docmap from all movies."""
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        """Persist index and docmap
