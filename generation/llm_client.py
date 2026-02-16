import logging
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)
_CLIENT_CACHE: Dict[tuple, OpenAI] = {}

# Optional default pricing (USD per 1M tokens).
# Override via config["pricing"] for exact models/rates in your environment.
DEFAULT_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}


def _estimate_tokens_from_text(text: str) -> int:
    # Rough estimate when token usage is unavailable.
    return max(1, len(text) // 4)


def _compute_estimated_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Dict[str, Dict[str, float]],
) -> float:
    model_rates = pricing.get(model)
    if not model_rates:
        return 0.0

    input_rate = model_rates.get("input", 0.0)
    output_rate = model_rates.get("output", 0.0)

    input_cost = (prompt_tokens / 1_000_000) * input_rate
    output_cost = (completion_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


def _get_client(config: Dict[str, Any]) -> OpenAI:
    api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Provide config['api_key'] or env var.")

    base_url = config.get("base_url")
    timeout = config.get("timeout")
    cache_key = (api_key, base_url, timeout)
    cached = _CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if base_url and timeout:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    elif base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif timeout:
        client = OpenAI(api_key=api_key, timeout=timeout)
    else:
        client = OpenAI(api_key=api_key)

    _CLIENT_CACHE[cache_key] = client
    return client


def generate_completion(messages: List[Dict[str, str]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a chat completion via OpenAI with retries and usage/cost logging.

    Args:
        messages: OpenAI chat messages.
        config: Runtime configuration dictionary. Supported keys:
            - api_key (str, optional)
            - model (str, default: "gpt-4o-mini")
            - temperature (float, default: 0.7)
            - max_tokens (int, default: 500)
            - timeout (int | float, optional)
            - base_url (str, optional)
            - retries (int, default: 3)
            - retry_backoff_seconds (float, default: 1.0)
            - pricing (dict, optional):
              {"model_name": {"input": usd_per_1m, "output": usd_per_1m}}

    Returns:
        Dict with completion text and metadata.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of role/content dicts")

    model = config.get("model", "gpt-4o-mini")
    temperature = float(config.get("temperature", 0.7))
    max_tokens = int(config.get("max_tokens", 500))
    retries = int(config.get("retries", 3))
    backoff = float(config.get("retry_backoff_seconds", 1.0))
    pricing = config.get("pricing") or DEFAULT_PRICING_PER_1M

    if retries < 1:
        retries = 1

    prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
    prompt_len_chars = len(prompt_text)

    client = _get_client(config)
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = (response.choices[0].message.content or "").strip()
            completion_len_chars = len(content)

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)

            if prompt_tokens is None:
                prompt_tokens = _estimate_tokens_from_text(prompt_text)
            if completion_tokens is None:
                completion_tokens = _estimate_tokens_from_text(content)

            estimated_cost_usd = _compute_estimated_cost(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                pricing=pricing,
            )

            logger.info(
                "llm.generate_completion model=%s prompt_chars=%d completion_chars=%d "
                "prompt_tokens=%d completion_tokens=%d estimated_cost_usd=%.6f",
                model,
                prompt_len_chars,
                completion_len_chars,
                prompt_tokens,
                completion_tokens,
                estimated_cost_usd,
            )

            return {
                "content": content,
                "model": model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "length": {
                    "prompt_chars": prompt_len_chars,
                    "completion_chars": completion_len_chars,
                },
                "estimated_cost_usd": estimated_cost_usd,
                "attempts": attempt,
            }
        except Exception as exc:
            last_error = exc
            logger.warning(
                "llm.generate_completion attempt=%d/%d failed: %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(backoff * attempt)

    logger.error("llm.generate_completion failed after %d attempts", retries)
    return {
        "content": "",
        "model": model,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "length": {"prompt_chars": prompt_len_chars, "completion_chars": 0},
        "estimated_cost_usd": 0.0,
        "attempts": retries,
        "error": str(last_error) if last_error else "Unknown error",
    }
