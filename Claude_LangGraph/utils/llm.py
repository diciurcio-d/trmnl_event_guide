"""Shared LLM utilities."""

import os
from google import genai

_client = None


def get_gemini_model():
    """Get Gemini client (singleton to avoid repeated setup)."""
    global _client

    if _client is not None:
        return _client

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Google API key not found. Set GOOGLE_API_KEY environment variable."
        )

    _client = genai.Client(api_key=api_key)
    return _client


def generate_content(prompt: str) -> str:
    """Generate content using Gemini."""
    client = get_gemini_model()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )
    return response.text
