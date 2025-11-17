import os
import pickle
import string

from .search_utils import load_movies, PROJECT_ROOT


CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")

# Reuse same normalization style: lowercase + remove punctuation
_TRANSLATOR = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    return text.lower().translate(_TRANSLATOR)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.split() if t]
    

class InvertedIndex:
    def __init__(self) -> None:
        # token -> set(doc_id)
        self.index: dict[str, set[int]] = {}
        # doc_id -> full movie dict
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize the text and add mappings token -> doc_id."""
        tokens = _tokenize(_normalize(text))
        for tok in tokens:
            if tok not in self.index:
                self.index[tok] = set()
            self.index[tok].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        """Return sorted list of doc IDs for a given token."""
        tok = _normalize(term)
        doc_ids = self.index.get(tok, set())
        return sorted(doc_ids)

    def build(self) -> None:
        """Build index + docmap from all movies."""
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]  # movie IDs are ints
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
