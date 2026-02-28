"""Event follow-up info: fetch event page and answer questions via LLM or web grounding."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_settings():
    """Load settings module directly to avoid circular imports."""
    settings_path = Path(__file__).parent.parent / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


_settings = _load_settings()
_GEMINI_MODEL = getattr(_settings, "GEMINI_MODEL", "gemini-2.5-flash-lite")
_WEB_GROUNDING_MODEL = "gemini-2.5-flash-lite"


def answer_event_question(
    event_url: str,
    event_name: str,
    question: str,
    cloudflare_protected: bool = False,
) -> dict:
    """
    Answer a question about a specific event.

    Strategy:
    1. Fetch the event page and try to answer from its content.
    2. If the page has no useful info (or fetch fails), fall back to
       Gemini web-grounded search.

    Returns:
        {
            "answer": str,
            "source": "url" | "web_grounding" | "error",
            "sources": list[str],
        }
    """
    try:
        page_text = _fetch_event_page(event_url, cloudflare_protected)

        if page_text:
            answer = _answer_from_page(page_text, event_name, question)
            if answer is not None:
                return {"answer": answer, "source": "url", "sources": []}

        # Page fetch failed or LLM said INSUFFICIENT — use web grounding
        return _web_grounding_answer(event_name, question)

    except Exception as exc:
        print(f"[event_info] Unhandled error: {exc}", flush=True)
        return {
            "answer": "Sorry, I couldn't find more info about this event.",
            "source": "error",
            "sources": [],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_event_page(event_url: str, cloudflare_protected: bool) -> str:
    """Fetch and clean the event page. Returns plain text, or empty string on failure."""
    if not event_url:
        return ""

    try:
        if cloudflare_protected:
            from curl_cffi import requests as cf_requests
            timeout = int(getattr(_settings, "EVENT_FETCHER_HTML_TIMEOUT_SEC", 15))
            resp = cf_requests.get(event_url, impersonate="chrome124", timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            raw_html = resp.text
        else:
            from venue_scout.event_fetcher import _fetch_raw_html, _fetch_with_playwright, _strip_html_for_llm
            try:
                raw_html = _fetch_raw_html(event_url)
            except Exception:
                raw_html, _ = _fetch_with_playwright(event_url)

        from venue_scout.event_fetcher import _strip_html_for_llm
        return _strip_html_for_llm(raw_html)[:8000]

    except Exception as exc:
        print(f"[event_info] Page fetch failed for {event_url}: {exc}", flush=True)
        return ""


def _answer_from_page(page_text: str, event_name: str, question: str) -> str | None:
    """
    Ask the LLM to answer the question from page content.
    Returns the answer string, or None if the content is INSUFFICIENT.
    """
    from utils.llm import generate_content

    prompt = (
        f"Answer this question about an event using only the page content below.\n"
        f"Be concise (2-4 sentences). If the content doesn't contain enough information to "
        f"answer, respond with exactly the word: INSUFFICIENT\n\n"
        f"Event: {event_name}\n"
        f"Question: {question}\n\n"
        f"Page content:\n{page_text}"
    )

    try:
        response_text = generate_content(prompt, model_name=_GEMINI_MODEL, timeout_sec=15)
        if response_text.strip() == "INSUFFICIENT":
            return None
        return response_text.strip()
    except Exception as exc:
        print(f"[event_info] LLM page-answer failed: {exc}", flush=True)
        return None


def _web_grounding_answer(event_name: str, question: str) -> dict:
    """Use Gemini web grounding to answer the question about the event."""
    try:
        from utils.llm import get_gemini_model
        from google.genai import types

        client = get_gemini_model(timeout_sec=20)
        prompt = (
            f"Find information about the event '{event_name}' in New York City and "
            f"answer this question: {question}"
        )
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
        )
        response = client.models.generate_content(
            model=_WEB_GROUNDING_MODEL,
            contents=prompt,
            config=config,
        )

        # Extract source URLs from grounding metadata
        sources: list[str] = []
        try:
            chunks = (
                response.candidates[0]
                .grounding_metadata
                .grounding_chunks
            )
            for chunk in chunks or []:
                url = getattr(getattr(chunk, "web", None), "uri", None)
                if url:
                    sources.append(url)
        except Exception:
            pass

        answer = (response.text or "").strip()
        if not answer:
            answer = "I couldn't find more information about this event."

        return {"answer": answer, "source": "web_grounding", "sources": sources}

    except Exception as exc:
        print(f"[event_info] Web grounding failed: {exc}", flush=True)
        return {
            "answer": "Sorry, I couldn't find more info about this event.",
            "source": "error",
            "sources": [],
        }
