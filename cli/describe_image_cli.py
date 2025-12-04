#!/usr/bin/env python3

import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal query rewriting using Gemini (image + text)"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file to analyze",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()

    # Load API key from environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY environment variable is not set")

    # Determine MIME type of the image, defaulting to image/jpeg
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    # Read image bytes
    try:
        with open(args.image, "rb") as f:
            img = f.read()
    except FileNotFoundError:
        raise SystemExit(f"Image file not found: {args.image}")

    # Set up Gemini client
    client = genai.Client(api_key=api_key)

    # System prompt describing the task
    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve "
        "search results from a movie database. Make sure to:\n"
        "- Synthesize visual and textual information\n"
        "- Focus on movie-specific details (actors, scenes, style, etc.)\n"
        "- Return only the rewritten query, without any additional commentary"
    )

    # Build multimodal parts: system prompt, image, and text query
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    # Call Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=parts,
    )

    rewritten = (response.text or "").strip()

    print(f"Rewritten query: {rewritten}")
    if getattr(response, "usage_metadata", None) is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
