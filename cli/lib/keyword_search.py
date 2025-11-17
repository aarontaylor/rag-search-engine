import os
import pickle
import string

from nltk.stem import PorterStemmer

from .search_utils import load_movies, load_stopwords, PROJECT_ROOT

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")

_TRANSLATOR = str.maketrans("", "", string.punctuation)
STOPWORDS = set(load_stopwords())
STEMMER = PorterStemmer()


def _normalize(text: str) -> str:
    return text.lower().translate(_TRANSLATOR)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.split() if t]


def _remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def _stem(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(t) for t in tokens]


class InvertedIndex:
    def __init__(self) -> None:
        # token -> set(doc_id)
        self.index: dict[str, set[int]] = {}
        # doc_id -> full movie dict
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize the text and add mappings token -> doc_id."""
        tokens = _stem(_remove_stopwords(_tokenize(_normalize(text))))
        for tok in tokens:
            if tok not in self.index:
                self.index[tok] = set()
            self.index[tok].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        """Return sorted list of doc IDs for a given token (already stemmed in our use)."""
        tok = _normalize(term)
        doc_ids = self.index.get(tok, set())
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
        """Persist index and docmap to disk under cache/."""
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        """Load index and docmap from disk. Raises if files don't exist."""
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)
