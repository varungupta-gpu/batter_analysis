"""Gemini API helper used by the segment and player LLM runners."""

import json
import time
from typing import Any, Dict, Optional

import requests

DEFAULT_MODEL = "gemini-3.7-flash"
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def build_gemini_request(
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
    }
    if top_p is not None:
        generation_config["topP"] = top_p
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }


def call_gemini(
    prompt: str,
    api_key: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
    top_p: Optional[float] = None,
    max_attempts: int = 6,
    timeout: int = 180,
) -> str:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    payload = build_gemini_request(
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
    )
    url = API_URL_TEMPLATE.format(model=model)
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        print(f"\nCalling Gemini model {model} (attempt {attempt}/{max_attempts})...", flush=True)
        try:
            response = requests.post(
                url,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response_data = response.json()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.ok:
                text = _extract_text(response_data)
                if text:
                    print("Gemini response received.", flush=True)
                    return text
                last_error = f"Empty Gemini response: {json.dumps(response_data)[:500]}"
            else:
                error = response_data.get("error", {})
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{error.get('status', 'UNKNOWN')} - {error.get('message', 'No message')}"
                )
                if response.status_code not in {429, 500, 503}:
                    break

        if attempt < max_attempts:
            print("Gemini returned an error. Retrying soon...", flush=True)
            time.sleep(min(2 ** (attempt - 1), 20))

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts. Last error: {last_error}")


def _extract_text(response_data: Dict[str, Any]) -> str:
    candidates = response_data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(text.strip() for text in texts if text and text.strip()).strip()
