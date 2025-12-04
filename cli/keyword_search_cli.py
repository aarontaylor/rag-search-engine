#!/usr/bin/env python3

import argparse
import math

from lib.keyword_search import (
    build_command,
    search_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search_command,
)
from lib.search_utils import BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build
    subparsers.add_parser("build", help="Build the inverted index")

    # search
    search_parser = subparsers.add_parser("search", help="Keyword search (simple)")
    search_parser.add_argument("query", type=str, help="Search query")

    # idf
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Search term")

    # tf
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)

    # tfidf
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)

    # bm25idf
    bm25idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score"
    )
    bm25idf_parser.add_argument("term", type=str)

    # bm25tf
    bm25tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score"
    )
    bm25tf_parser.add_argument("doc_id", type=int)
    bm25tf_parser.add_argument("term", type=str)
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1)
    bm25tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B)

    # bm25search
    bm25search_parser = subparsers.add_parser(
        "bm25search",
        help="Search movies using full BM25",
    )
    bm25search_parser.add_argument("query", type=str)
    bm25search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")

        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")

        case "idf":
            from lib.keyword_search import InvertedIndex, tokenize_text

            idx = InvertedIndex()
            idx.load()
            tokens = tokenize_text(args.term)
            if len(tokens) > 1:
                raise ValueError("idf expects one token")
            token = tokens[0]
            df = len(idx.get_documents(token))
            doc_count = len(idx.docmap)
            idf = math.log((doc_count + 1) / (df + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tf":
            idx = InvertedIndex()
            idx.load()
            print(idx.get_tf(args.doc_id, args.term))

        case "tfidf":
            from lib.keyword_search import InvertedIndex, tokenize_text

            idx = InvertedIndex()
            idx.load()

            tf = idx.get_tf(args.doc_id, args.term)
            tokens = tokenize_text(args.term)
            token = tokens[0] if tokens else None
            df = len(idx.get_documents(token)) if token else 0
            doc_count = len(idx.docmap)
            idf = math.log((doc_count + 1) / (df + 1))
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf * idf:.2f}")

        case "bm25idf":
            score = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {score:.2f}")

        case "bm25tf":
            score = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {score:.2f}")

        case "bm25search":
            print("BM25 Searching for:", args.query)
            results = bm25_search_command(args.query, args.limit)
            for i, (doc, score) in enumerate(results, 1):
                print(f"{i}. ({doc['id']}) {doc['title']} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
