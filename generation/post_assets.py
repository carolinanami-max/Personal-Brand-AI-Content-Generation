import base64
import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from generation.llm_client import generate_completion


def _build_hashtag_prompt(post: str, topic: str, business_objective: str) -> str:
    return (
        "Create 8 LinkedIn hashtags for this post.\n"
        "Rules:\n"
        "- Return JSON only.\n"
        '- Schema: {"hashtags": ["#tag1", "#tag2"]}\n'
        "- Hashtags must be relevant to SMEs, AI implementation, and business outcomes.\n"
        "- No generic spam tags like #success or #motivation.\n\n"
        f"Topic: {topic}\n"
        f"Business objective: {business_objective}\n\n"
        f"Post:\n{post}\n"
    )


def _parse_hashtags(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return text
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            return text

    tags = payload.get("hashtags", [])
    if not isinstance(tags, list):
        return ""
    cleaned = []
    for tag in tags:
        tag_text = str(tag).strip()
        if not tag_text:
            continue
        if not tag_text.startswith("#"):
            tag_text = f"#{tag_text.lstrip('#')}"
        cleaned.append(tag_text.replace(" ", ""))
    return " ".join(cleaned[:10])


def generate_hashtags(post: str, topic: str, business_objective: str, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": "You create specific, SME-relevant LinkedIn hashtags. Return JSON only.",
        },
        {
            "role": "user",
            "content": _build_hashtag_prompt(post=post, topic=topic, business_objective=business_objective),
        },
    ]
    llm_result = generate_completion(messages=messages, config=config)
    hashtags = _parse_hashtags(llm_result.get("content", ""))
    metadata = {
        "llm": {
            "model": llm_result.get("model"),
            "attempts": llm_result.get("attempts"),
            "usage": llm_result.get("usage", {}),
            "length": llm_result.get("length", {}),
            "estimated_cost_usd": llm_result.get("estimated_cost_usd", 0.0),
            "error": llm_result.get("error"),
        }
    }
    return hashtags, metadata


def generate_post_image(post: str, topic: str, config: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, {"error": "OPENAI_API_KEY not set"}

    image_model = config.get("image_model", "gpt-image-1")
    image_size = config.get("image_size", "1024x1024")
    client = OpenAI(api_key=api_key, timeout=float(config.get("timeout", 60)))

    prompt = (
        "Create a premium, professional LinkedIn cover-style image for a business audience.\n"
        "Art direction:\n"
        "- Executive editorial style, modern and minimal.\n"
        "- Clean composition with a clear focal point and strong visual hierarchy.\n"
        "- Sophisticated corporate palette (deep blue, slate, white, subtle accent), high contrast.\n"
        "- Subtle data/operations motifs (workflow lines, dashboards, process blocks) without clutter.\n"
        "- Plenty of negative space suitable for social media crops.\n"
        "- No logos, no brand marks, no watermarks, no UI screenshots.\n"
        "- Avoid busy backgrounds, cartoon style, and stock-photo look.\n"
        "- Do not render readable text inside the image.\n"
        "Output quality:\n"
        "- Crisp, polished, presentation-ready, suitable for LinkedIn post visuals.\n"
        f"Topic: {topic}\n"
        f"Post context: {post}\n"
    )

    try:
        response = client.images.generate(
            model=image_model,
            prompt=prompt,
            size=image_size,
        )
        data = response.data[0]
        b64_data = getattr(data, "b64_json", None)
        if not b64_data:
            return None, {"error": "Image response missing b64_json", "model": image_model}

        image_bytes = base64.b64decode(b64_data)
        with tempfile.NamedTemporaryFile(prefix="pbcg_", suffix=".png", delete=False) as temp_file:
            temp_file.write(image_bytes)
            image_path = temp_file.name

        return image_path, {"model": image_model, "size": image_size, "path": image_path}
    except Exception as exc:
        return None, {"error": str(exc), "model": image_model, "size": image_size}
