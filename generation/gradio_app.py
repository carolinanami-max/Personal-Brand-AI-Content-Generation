import json
import os
import re
import socket
import sys
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from generation.generate_post import OPENAI_MODEL_OPTIONS, generate_post
from generation.llm_client import generate_completion
from generation.brand_checker import check_brand_consistency
from generation.refiner import refine_post
from generation.post_assets import generate_hashtags, generate_post_image
from generation.feedback_loop import build_feedback_guidance, save_feedback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SOFIE_PHOTO_PATH = ASSETS_DIR / "sopie_bennett.png"
PROFILE_IMAGE_PATH = SOFIE_PHOTO_PATH

COHERE_MODEL_OPTIONS = [
    "command-a-03-2025",
    "command-r7b-12-2024",
    "command-r-plus-08-2024",
    "command-r-08-2024",
]

DASHBOARD_CSS = """
#app-shell {gap: 18px;}
#sidebar {
  background: #ffffff;
  border: 1px solid #dfe5f0;
  border-radius: 14px;
  min-height: 92vh;
  padding: 14px 10px;
}
#main-area {
  background: #f5f7fc;
  border: 1px solid #dfe5f0;
  border-radius: 14px;
  padding: 16px;
}
.logo-wrap {
  font-size: 30px;
  font-weight: 700;
  color: #1f4fa8;
  margin: 4px 4px 16px 4px;
}
.logo-wrap span {
  color: #3ea2d8;
}
.nav-menu {
  margin-top: 8px;
  padding: 0 2px;
}
.nav-item {
  padding: 10px 12px;
  border-radius: 10px;
  color: #36517f;
  margin-bottom: 6px;
  font-size: 18px;
}
.nav-item.active {
  background: #e8f0ff;
  color: #245ec7;
  font-weight: 600;
}
.nav-btn button {
  width: 100%;
  justify-content: flex-start;
  padding: 10px 12px;
  border-radius: 10px;
  color: #36517f;
  margin-bottom: 6px;
  font-size: 18px;
  border: none;
  background: transparent;
}
.nav-btn.active button {
  background: #e8f0ff;
  color: #245ec7;
  font-weight: 600;
}
.pillars-pre {
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}
.user-tile {
  margin-top: 26px;
  border-top: 1px solid #e3e8f3;
  padding-top: 12px;
  color: #41577f;
}
.user-row {
  display: flex;
  align-items: center;
  gap: 10px;
}
.profile-sm {
  width: 46px !important;
  height: 46px !important;
  min-width: 46px;
  min-height: 46px;
  aspect-ratio: 1 / 1;
  border-radius: 50%;
  overflow: hidden;
  border: 2px solid #e3e8f3;
}
.profile-sm > div {
  width: 100% !important;
  height: 100% !important;
}
.profile-sm img {
  width: 100%;
  height: 100%;
  aspect-ratio: 1 / 1;
  border-radius: 50%;
  object-fit: cover;
  object-position: center top;
}
.topbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.icon-chip {
  font-size: 18px;
  color: #4e6891;
}
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #ffffff;
  border: 1px solid #dfe5f0;
  border-radius: 12px;
  padding: 10px 14px;
}
.welcome {
  margin: 14px 0 6px 2px;
}
.welcome h2 {
  margin: 0 0 3px 0;
  color: #243a61;
}
.welcome p {
  margin: 0;
  color: #516c95;
}
.panel-card {
  background: #ffffff;
  border: 1px solid #dfe5f0;
  border-radius: 12px;
  padding: 12px;
}
.panel-title {
  font-size: 25px;
  font-weight: 600;
  color: #2b446f;
  margin-bottom: 8px;
}
.chip-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.chip {
  background: #edf3ff;
  color: #36517f;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 14px;
}
.feedback-icons button {
  min-width: 64px;
}
#generate-btn button {
  background: linear-gradient(90deg, #2d63c9, #347fda);
  border: none;
}
@media (max-width: 1024px) {
  #sidebar {min-height: auto;}
}
"""

# Load env vars from common local files so OPENAI_API_KEY is available in UI runs.
load_dotenv(PROJECT_ROOT / ".ENV")
load_dotenv(PROJECT_ROOT / ".env")


def _build_config(
    model: str,
    custom_model: Optional[str],
    cohere_model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    timeout: float,
) -> Dict[str, Any]:
    selected_model = (custom_model or model or "").strip()
    config: Dict[str, Any] = {
        "model": selected_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retries": retries,
        "timeout": timeout,
        "cohere_model": cohere_model,
    }
    return config


def _load_prompt_file(filename: str) -> str:
    prompt_path = PROJECT_ROOT / "prompts" / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _extract_json_payload(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        raise ValueError("Empty response content.")

    # Strip fenced code blocks if present.
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)

    start = content.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response.")

    decoder = json.JSONDecoder()
    parse_errors: List[str] = []

    for candidate in (
        content[start:],
        re.sub(r",\s*([}\]])", r"\1", content[start:]),
    ):
        try:
            payload, _ = decoder.raw_decode(candidate)
            if not isinstance(payload, dict):
                raise ValueError("Top-level JSON payload must be an object.")
            return payload
        except Exception as exc:
            parse_errors.append(str(exc))

    # Salvage attempt: if the model appended trailing broken text, try parsing up to
    # the last plausible object terminator.
    core = content[start:]
    closing_positions = [idx for idx, ch in enumerate(core) if ch == "}"]
    for idx in reversed(closing_positions[-60:]):
        candidate = core[: idx + 1]
        try:
            payload, _ = decoder.raw_decode(candidate)
            if not isinstance(payload, dict):
                continue
            return payload
        except Exception as exc:
            parse_errors.append(str(exc))

    raise ValueError(f"Invalid JSON payload from model. Parse errors: {' | '.join(parse_errors)}")


def _build_pillars_markdown(payload: Dict[str, Any]) -> str:
    pillars = payload.get("pillars", [])
    if not isinstance(pillars, list) or not pillars:
        return "No content pillars available."

    lines: List[str] = []
    def _priority_value(pillar: Dict[str, Any]) -> int:
        try:
            return int(pillar.get("priority", 999))
        except (TypeError, ValueError):
            return 999

    sorted_pillars = sorted(
        [pillar for pillar in pillars if isinstance(pillar, dict)],
        key=_priority_value,
    )
    for pillar in sorted_pillars:
        name = str(pillar.get("name", "Untitled")).strip()
        description = str(pillar.get("description", "")).strip()
        pain_points = pillar.get("sme_pain_points", [])
        angles = pillar.get("example_angles", [])
        post_types = pillar.get("recommended_post_types", [])
        priority = pillar.get("priority", "-")

        lines.append(f"### {priority}. {name}")
        lines.append(description or "No description.")
        if isinstance(pain_points, list) and pain_points:
            lines.append("SME pain points:")
            lines.extend([f"- {str(item)}" for item in pain_points[:3]])
        if isinstance(angles, list) and angles:
            lines.append("Example angles:")
            lines.extend([f"- {str(item)}" for item in angles[:3]])
        if isinstance(post_types, list) and post_types:
            lines.append(f"Recommended post types: {', '.join(str(item) for item in post_types)}")
        lines.append("")

    return "\n".join(lines).strip()


def _pillars_to_topic_options(payload: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
    pillars = payload.get("pillars", [])
    if not isinstance(pillars, list):
        return [], None

    options: List[str] = []
    for pillar in pillars:
        if not isinstance(pillar, dict):
            continue
        name = str(pillar.get("name", "")).strip()
        angles = pillar.get("example_angles", [])
        if isinstance(angles, list) and angles:
            for angle in angles[:2]:
                angle_text = str(angle).strip()
                if angle_text:
                    options.append(angle_text)
        elif name:
            options.append(name)

    # Preserve order and remove duplicates.
    deduped: List[str] = []
    seen = set()
    for option in options:
        if option in seen:
            continue
        seen.add(option)
        deduped.append(option)
    default_value = deduped[0] if deduped else None
    return deduped, default_value


def generate_content_pillars(
    target_persona: str,
    model: str,
    custom_model: Optional[str],
    temperature: float,
    max_tokens: int,
    retries: int,
    timeout: float,
) -> Tuple[Dict[str, Any], str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        message = "OPENAI_API_KEY not found in environment."
        return {"error": message}, message, gr.update()

    template = _load_prompt_file("pillar_generation_prompt.txt")
    user_prompt = template
    base_max_tokens = max(1200, int(max_tokens))
    config = {
        "model": (custom_model or model or "").strip(),
        "temperature": temperature,
        "max_tokens": base_max_tokens,
        "retries": retries,
        "timeout": timeout,
        "response_format": {"type": "json_object"},
    }

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate reusable SME content pillars. "
                    "Return strict JSON only matching the requested schema."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        def _request_pillars(request_config: Dict[str, Any]) -> Dict[str, Any]:
            result = generate_completion(messages=messages, config=request_config)
            content = (result.get("content") or "").strip()
            if not content:
                llm_error = (result.get("error") or "").strip()
                if llm_error:
                    raise ValueError(f"Model returned empty content. LLM error: {llm_error}")
                raise ValueError("Model returned empty content.")
            return _extract_json_payload(content)

        request_variants: List[Dict[str, Any]] = [
            dict(config),
            # Some model/provider combinations reject response_format=json_object.
            {k: v for k, v in config.items() if k != "response_format"},
            # Last retry with more output budget in case of truncation.
            {
                **{k: v for k, v in config.items() if k != "response_format"},
                "max_tokens": max(1800, base_max_tokens),
                "temperature": min(float(temperature), 0.5),
            },
        ]

        last_exc: Optional[Exception] = None
        payload: Optional[Dict[str, Any]] = None
        for variant in request_variants:
            try:
                payload = _request_pillars(variant)
                break
            except Exception as exc:
                last_exc = exc

        if payload is None:
            raise ValueError(str(last_exc) if last_exc else "Unable to generate pillars.")

        payload["user_id"] = payload.get("user_id") or ""
        payload["version"] = payload.get("version") or "v1"
        payload["created_at"] = payload.get("created_at") or datetime.now(timezone.utc).isoformat()

        pillars = payload.get("pillars", [])
        if not isinstance(pillars, list):
            raise ValueError("Invalid pillars payload: 'pillars' must be a list.")
        payload["pillars"] = pillars[:6]

        pillar_md = _build_pillars_markdown(payload)
        topic_options, default_topic = _pillars_to_topic_options(payload)
        topic_update = gr.update(choices=topic_options, value=default_topic)
        return payload, pillar_md, topic_update
    except Exception as exc:
        message = f"Failed to generate pillars: {exc}"
        return {"error": message}, message, gr.update()


def show_dashboard_view() -> Tuple[Any, Any]:
    return gr.update(visible=True), gr.update(visible=False)


def show_content_pillars_view() -> Tuple[Any, Any]:
    return gr.update(visible=False), gr.update(visible=True)


def run_generation(
    topic: str,
    post_type: str,
    target_persona: str,
    model: str,
    custom_model: Optional[str],
    cohere_model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    timeout: float,
) -> Tuple[str, str, str, Optional[str], Dict[str, Any], Any]:
    steps: List[str] = []
    topic = (topic or "").strip()
    target_persona = (target_persona or "").strip()

    if not topic:
        return "Validation failed: Topic is required.", "", "", None, {}, gr.update(visible=False)
    if not target_persona:
        return "Validation failed: Target persona is required.", "", "", None, {}, gr.update(visible=False)
    if not (custom_model or model):
        return "Validation failed: Model selection is required.", "", "", None, {}, gr.update(visible=False)

    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_key:
        return "Validation failed: OPENAI_API_KEY not found in environment.", "", "", None, {}, gr.update(visible=False)
    has_cohere_key = bool(os.getenv("COHERE_API_KEY"))
    if not has_cohere_key:
        return "Validation failed: COHERE_API_KEY not found in environment.", "", "", None, {}, gr.update(visible=False)

    try:
        steps.append("1. Inputs validated - topic, objective, and keys are present.")
        config = _build_config(
            model=model,
            custom_model=custom_model,
            cohere_model=cohere_model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
        )
        feedback_guidance, feedback_meta = build_feedback_guidance(
            post_type=post_type,
            target_persona=target_persona,
        )
        config["feedback_guidance"] = feedback_guidance
        steps.append(
            "2. Loaded feedback memory - "
            f"accepted: {feedback_meta.get('accepted_count', 0)}, "
            f"rejected: {feedback_meta.get('rejected_count', 0)}."
        )
        steps.append("3. Generating candidate drafts - creates multiple angle variations.")
        draft_post, metadata = generate_post(
            topic=topic,
            post_type=post_type,
            business_objective=target_persona,
            config=config,
        )
        selected_angle = (
            metadata.get("candidate_generation", {}).get("selected_angle")
            if isinstance(metadata, dict)
            else None
        )
        if selected_angle:
            steps.append(f"4. Cohere selected best angle - picked: {selected_angle}.")
        else:
            steps.append("4. Cohere selected best angle - ranked drafts by quality.")

        steps.append("5. First refinement pass - removes vague language and improves specificity.")
        refined_post, refinement_metadata = refine_post(
            draft_post=draft_post,
            topic=topic,
            post_type=post_type,
            business_objective=target_persona,
            config=config,
        )
        if not refined_post:
            refined_post = draft_post

        steps.append("6. Initial brand check - scores tone, SME relevance, clarity, and differentiation.")
        initial_brand_result, initial_brand_metadata = check_brand_consistency(
            post=refined_post,
            config=config,
        )

        steps.append("7. Feedback-driven refinement - applies brand checker suggestions.")
        feedback_refined_post, feedback_refinement_metadata = refine_post(
            draft_post=refined_post,
            topic=topic,
            post_type=post_type,
            business_objective=target_persona,
            config=config,
            brand_feedback_summary=initial_brand_result.get("feedback_summary", ""),
            brand_score=int(initial_brand_result.get("score", 0)),
        )
        final_post = feedback_refined_post or refined_post

        steps.append("8. Final brand check - verifies improvements after feedback.")
        steps.append("9. Generating hashtags for publishing.")
        steps.append("10. Generating supporting image.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_brand = executor.submit(check_brand_consistency, post=final_post, config=config)
            future_hashtags = executor.submit(
                generate_hashtags,
                post=final_post,
                topic=topic,
                business_objective=target_persona,
                config=config,
            )
            future_image = executor.submit(
                generate_post_image,
                post=final_post,
                topic=topic,
                config=config,
            )

            final_brand_result, final_brand_metadata = future_brand.result()
            hashtags, hashtags_metadata = future_hashtags.result()
            image_path, image_metadata = future_image.result()

        metadata["refinement"] = {
            "initial": refinement_metadata,
            "feedback_driven": feedback_refinement_metadata,
        }
        metadata["brand_check"] = {
            "initial": {
                "result": initial_brand_result,
                "metadata": initial_brand_metadata,
            },
            "final": {
                "result": final_brand_result,
                "metadata": final_brand_metadata,
            },
        }
        final_score = final_brand_result.get("score", 0)
        steps.append(f"11. Final post ready - final brand score: {final_score}/100.")
        if not image_path:
            steps.append("Image generation failed - see logs/metadata.")
        metadata["post_assets"] = {
            "hashtags": hashtags_metadata,
            "image": image_metadata,
        }

        feedback_payload = {
            "topic": topic,
            "post_type": post_type,
            "target_persona": target_persona,
            "final_post": final_post,
            "hashtags": hashtags,
            "brand_score": int(final_score or 0),
        }

        return (
            "\n".join(steps),
            final_post,
            hashtags,
            image_path,
            feedback_payload,
            gr.update(visible=True),
        )
    except Exception as exc:
        steps.append(f"Failed: {exc}")
        return "\n".join(steps), "", "", None, {}, gr.update(visible=False)


def _persist_feedback(
    decision: str,
    notes: str,
    payload: Dict[str, Any],
) -> bool:
    decision_norm = (decision or "").strip().lower()
    if decision_norm not in {"accept", "reject"}:
        return False
    if not payload or not payload.get("final_post"):
        return False

    record = dict(payload)
    record["decision"] = decision_norm
    record["notes"] = (notes or "").strip()
    save_feedback(record)
    return True


def submit_accept_feedback(payload: Dict[str, Any]) -> Tuple[Any, Any]:
    _persist_feedback(decision="accept", notes="", payload=payload)
    return gr.update(visible=False), gr.update(value="")


def open_reject_details() -> Any:
    return gr.update(visible=True)


def cancel_reject_details() -> Tuple[Any, Any]:
    return gr.update(visible=False), gr.update(value="")


def submit_reject_feedback(notes: str, payload: Dict[str, Any]) -> Tuple[Any, Any]:
    _persist_feedback(decision="reject", notes=notes, payload=payload)
    return gr.update(visible=False), gr.update(value="")


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AI Brand Builder", css=DASHBOARD_CSS, theme=gr.themes.Soft()) as demo:
        with gr.Row(elem_id="app-shell"):
            with gr.Column(scale=1, elem_id="sidebar"):
                gr.Markdown('<div class="logo-wrap">AI Brand Builder</div>')
                with gr.Column(elem_classes=["nav-menu"]):
                    dashboard_nav_btn = gr.Button("🏠 Dashboard", elem_classes=["nav-btn", "active"])
                    pillars_nav_btn = gr.Button("🧩 Content Pillars", elem_classes=["nav-btn"])
                    gr.Markdown(
                        """
                        <div class="nav-item">🗓️ Content Planner</div>
                        <div class="nav-item">✍️ Generate Posts</div>
                        <div class="nav-item">📊 Analytics</div>
                        <div class="nav-item">⚙️ Settings</div>
                        """
                    )
                with gr.Row(elem_classes=["user-tile", "user-row"]):
                    gr.Image(
                        value=str(PROFILE_IMAGE_PATH),
                        show_label=False,
                        container=False,
                        interactive=False,
                        width=46,
                        height=46,
                        elem_classes=["profile-sm"],
                    )
                    gr.Markdown("**Sofie Bennet**  \nAI Consultant")

            with gr.Column(scale=4, elem_id="main-area"):
                with gr.Row(elem_classes=["topbar"]):
                    gr.Markdown("**Dashboard**")
                    with gr.Row(elem_classes=["topbar-right"]):
                        gr.Image(
                            value=str(PROFILE_IMAGE_PATH),
                            show_label=False,
                            container=False,
                            interactive=False,
                            width=46,
                            height=46,
                            elem_classes=["profile-sm"],
                        )
                        gr.Markdown('<span class="icon-chip">🔔</span> <span class="icon-chip">☰</span>')
                with gr.Column(visible=True) as dashboard_view:
                    gr.Markdown(
                        """
                        <div class="welcome">
                          <h2>Welcome back, Sofie!</h2>
                          <p>Let's create your next LinkedIn post.</p>
                        </div>
                        """
                    )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                """
                                <div class="panel-card">
                                  <div class="panel-title">Brand Profile Overview</div>
                                  <div class="chip-row">
                                    <span class="chip">AI Automation for SMEs</span>
                                    <span class="chip">German Market</span>
                                    <span class="chip">Practical Insights</span>
                                    <span class="chip">Case Studies</span>
                                  </div>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-card"]):
                                gr.Markdown("### Generate a New Post")
                                topic = gr.Dropdown(
                                    label="Select Content Pillar / Topic",
                                    choices=[],
                                    allow_custom_value=True,
                                    value=None,
                                    info="Auto-filled from Content Pillars. You can also type a custom topic.",
                                )
                                post_type = gr.Dropdown(
                                    label="Choose Post Type",
                                    choices=["thought_leadership", "educational", "trend_commentary"],
                                    value="thought_leadership",
                                )
                                target_persona = gr.Textbox(
                                    label="Target Persona",
                                    placeholder="e.g., SME founder with limited technical background",
                                )

                                with gr.Accordion("Advanced Generation Settings", open=False):
                                    model = gr.Dropdown(
                                        label="Model",
                                        choices=OPENAI_MODEL_OPTIONS,
                                        value=OPENAI_MODEL_OPTIONS[0],
                                    )
                                    custom_model = gr.Textbox(
                                        label="Custom Model (optional override)",
                                        placeholder="e.g., gpt-5-mini",
                                    )
                                    cohere_model = gr.Dropdown(
                                        label="Cohere Evaluator Model",
                                        choices=COHERE_MODEL_OPTIONS,
                                        value=COHERE_MODEL_OPTIONS[0],
                                    )
                                    temperature = gr.Slider(
                                        label="Temperature", minimum=0.0, maximum=1.5, step=0.1, value=0.7
                                    )
                                    max_tokens = gr.Slider(
                                        label="Max Tokens", minimum=100, maximum=2000, step=50, value=500
                                    )
                                    retries = gr.Slider(label="Retries", minimum=1, maximum=6, step=1, value=3)
                                    timeout = gr.Slider(
                                        label="Timeout (seconds)", minimum=10, maximum=180, step=5, value=60
                                    )
                                generate_btn = gr.Button("Generate Post", variant="primary", elem_id="generate-btn")
                                steps_output = gr.Textbox(
                                    label="Generation Steps",
                                    lines=10,
                                )

                    with gr.Group(elem_classes=["panel-card"]):
                        final_post_output = gr.Textbox(
                            label="Final Post",
                            lines=18,
                        )
                        hashtags_output = gr.Textbox(
                            label="Hashtags",
                            lines=2,
                        )
                        image_output = gr.Image(
                            label="Generated Image",
                            type="filepath",
                        )
                        feedback_state = gr.State({})
                        with gr.Column(visible=False) as feedback_controls:
                            gr.Markdown("### Feedback")
                            with gr.Row(elem_classes=["feedback-icons"]):
                                accept_btn = gr.Button("✅", variant="secondary")
                                reject_btn = gr.Button("❌", variant="secondary")
                            with gr.Column(visible=False) as reject_panel:
                                reject_notes = gr.Textbox(
                                    label="Why was this rejected?",
                                    placeholder="Add details to improve future generations.",
                                    lines=3,
                                )
                                with gr.Row():
                                    reject_submit_btn = gr.Button("Submit Rejection", variant="stop")
                                    reject_cancel_btn = gr.Button("Cancel")

                with gr.Column(visible=False) as pillars_view:
                    gr.Markdown(
                        """
                        <div class="welcome">
                          <h2>Content Pillars</h2>
                          <p>Reusable SME themes generated from your brand profile.</p>
                        </div>
                        """
                    )
                    with gr.Group(elem_classes=["panel-card"]):
                        refresh_pillars_btn = gr.Button("Regenerate Pillars")
                        pillars_markdown = gr.Markdown("Click **Content Pillars** to generate your strategic pillars.")
                        pillars_json = gr.JSON(
                            label="Pillars JSON",
                            value={
                                "user_id": "",
                                "created_at": "",
                                "version": "v1",
                                "pillars": [],
                            },
                        )

        generate_btn.click(
            fn=run_generation,
            inputs=[
                topic,
                post_type,
                target_persona,
                model,
                custom_model,
                cohere_model,
                temperature,
                max_tokens,
                retries,
                timeout,
            ],
            outputs=[
                steps_output,
                final_post_output,
                hashtags_output,
                image_output,
                feedback_state,
                feedback_controls,
            ],
        )

        dashboard_nav_btn.click(
            fn=show_dashboard_view,
            inputs=[],
            outputs=[dashboard_view, pillars_view],
        )

        pillars_nav_btn.click(
            fn=show_content_pillars_view,
            inputs=[],
            outputs=[dashboard_view, pillars_view],
        ).then(
            fn=generate_content_pillars,
            inputs=[target_persona, model, custom_model, temperature, max_tokens, retries, timeout],
            outputs=[pillars_json, pillars_markdown, topic],
        )

        refresh_pillars_btn.click(
            fn=generate_content_pillars,
            inputs=[target_persona, model, custom_model, temperature, max_tokens, retries, timeout],
            outputs=[pillars_json, pillars_markdown, topic],
        )

        accept_btn.click(
            fn=submit_accept_feedback,
            inputs=[feedback_state],
            outputs=[reject_panel, reject_notes],
        )

        reject_btn.click(
            fn=open_reject_details,
            inputs=[],
            outputs=[reject_panel],
        )

        reject_cancel_btn.click(
            fn=cancel_reject_details,
            inputs=[],
            outputs=[reject_panel, reject_notes],
        )

        reject_submit_btn.click(
            fn=submit_reject_feedback,
            inputs=[reject_notes, feedback_state],
            outputs=[reject_panel, reject_notes],
        )

    return demo


def main() -> None:
    demo = build_interface()
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=preferred_port,
            allowed_paths=[str(ASSETS_DIR)],
        )
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            fallback_port = sock.getsockname()[1]
        demo.launch(
            server_name="127.0.0.1",
            server_port=fallback_port,
            allowed_paths=[str(ASSETS_DIR)],
        )


if __name__ == "__main__":
    main()
