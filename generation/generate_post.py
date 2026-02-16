import argparse
import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  # Add this line

# Set up logger first
logger = logging.getLogger(__name__)

# Add path before local imports
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now import local modules
from generation.llm_client import generate_completion
from generation.cohere_evaluator import evaluate_candidates_with_cohere
from src.document_processor import DocumentProcessor

# Initialize document processor for RAG
doc_processor = DocumentProcessor()
doc_processor.load_all()  # Load knowledge base once
logger.info("📚 RAG knowledge base loaded")

OPENAI_MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
]

TEMPLATE_MAP = {
    "thought_leadership": "thought_leadership.txt",
    "educational": "educational.txt",
    "trend_commentary": "trend_commentary.txt",
}

ANGLE_STRATEGIES = [
    ("contrarian", "Take a clear contrarian stance and challenge a common belief in the first 2 lines."),
    (
        "operational_lesson",
        "Focus on an operational lesson from execution friction (ownership, process, constraints, tradeoffs).",
    ),
    ("case_first", "Start with a concrete case/example first, then extract the insight and implication."),
]

STATIC_EXAMPLES = [
    {
        "topic": "Why SME teams fail at AI adoption after pilot success",
        "post_type": "thought_leadership",
        "business_objective": "Build authority with SME founders and generate inbound consulting leads",
    },
    {
        "topic": "How to design a 30-day AI workflow rollout for a 20-person services company",
        "post_type": "educational",
        "business_objective": "Educate SME operators with practical implementation guidance",
    },
    {
        "topic": "Shift from AI experimentation to AI operations in small businesses",
        "post_type": "trend_commentary",
        "business_objective": "Position as a strategic advisor on SME AI execution",
    },
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_prompt_file(filename: str) -> str:
    prompt_path = _project_root() / "prompts" / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _build_user_prompt(
    template_text: str,
    topic: str,
    business_objective: str,
    angle_instruction: str = "",
    feedback_guidance: str = "",
) -> str:
    # GET RAG CONTEXT FROM KNOWLEDGE BASE
    context = doc_processor.search(topic)
    
    # Format template with context
    base = template_text.format(
        topic=topic,
        audience="SME decision makers",
        goal=business_objective,
        brand_context=context,
        market_context="N/A",
    )
    sections = [base]
    if angle_instruction:
        sections.append(f"Additional angle instruction:\n- {angle_instruction}")
    if (feedback_guidance or "").strip():
        sections.append(f"Feedback memory from previous outputs:\n{feedback_guidance.strip()}")
    return "\n\n".join(sections)


def _generate_candidate_drafts(
    system_prompt: str,
    template_text: str,
    topic: str,
    post_type: str,
    business_objective: str,
    config: Dict[str, Any],
    feedback_guidance: str = "",
) -> List[Dict[str, Any]]:
    candidates_by_index: Dict[int, Dict[str, Any]] = {}

    def _generate_for_angle(index: int, angle_name: str, angle_instruction: str) -> Tuple[int, Dict[str, Any]]:
        user_prompt = _build_user_prompt(
            template_text=template_text,
            topic=topic,
            business_objective=business_objective,
            angle_instruction=angle_instruction,
            feedback_guidance=feedback_guidance,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        llm_result = generate_completion(messages=messages, config=config)
        text = (llm_result.get("content") or "").strip()
        candidate = {
            "post_type": post_type,
            "angle": angle_name,
            "text": text,
            "llm": {
                "model": llm_result.get("model"),
                "attempts": llm_result.get("attempts"),
                "usage": llm_result.get("usage", {}),
                "length": llm_result.get("length", {}),
                "estimated_cost_usd": llm_result.get("estimated_cost_usd", 0.0),
                "error": llm_result.get("error"),
            },
        }
        return index, candidate

    max_workers = min(len(ANGLE_STRATEGIES), int(config.get("parallel_workers", 3)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_generate_for_angle, idx, angle_name, angle_instruction)
            for idx, (angle_name, angle_instruction) in enumerate(ANGLE_STRATEGIES)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, candidate = future.result()
            except Exception:
                continue
            if (candidate.get("text") or "").strip():
                candidates_by_index[idx] = candidate

    ordered_indices = sorted(candidates_by_index.keys())
    return [candidates_by_index[idx] for idx in ordered_indices]


def _build_mock_post(topic: str, post_type: str, business_objective: str) -> str:
    if post_type == "thought_leadership":
        return (
            f"Most SME teams do not fail at AI because of tools.\n\n"
            f"They fail because ownership is unclear after the pilot.\n\n"
            f"In one client project on {topic}, the pilot worked, but no one owned handoff to operations. "
            f"Within two weeks, usage dropped.\n\n"
            f"The business implication is simple: if adoption accountability is missing, ROI disappears.\n\n"
            f"Practical takeaway: assign one workflow owner before rollout day.\n\n"
            f"CTA: Are you assigning AI ownership at the team level or only at leadership level?"
        )
    if post_type == "educational":
        return (
            f"If AI rollout feels messy, the problem is usually sequence.\n\n"
            f"Concept: operational adoption beats technical adoption.\n\n"
            f"Step 1: choose one workflow tied to a KPI.\n"
            f"Step 2: define owner, trigger, and review cadence.\n"
            f"Step 3: measure outcome weekly and tighten process.\n\n"
            f"Example workflow for {topic}: intake -> draft -> review -> client-ready output with clear handoffs.\n\n"
            f"SME application: start with one team and one measurable outcome to keep resource load realistic.\n\n"
            f"CTA: Want a one-page rollout checklist I use with SME clients?"
        )
    return (
        f"Trend summary: SMEs are moving from AI experimentation to operational integration.\n\n"
        f"What changed is not model quality alone. It is pressure for measurable business outcomes.\n\n"
        f"For SMEs, this matters because ad-hoc prompting does not scale under limited team capacity.\n\n"
        f"Tactical implication: standardize one high-frequency workflow first, then expand.\n\n"
        f"Example: a small services team used a fixed discovery-call summary workflow and cut prep time by 35%.\n\n"
        f"CTA: Which workflow in your business is ready for standardization this quarter?"
    )


def generate_post(
    topic: str,
    post_type: str,
    business_objective: str,
    config: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a LinkedIn post via OpenAI using system + template prompt assembly.

    Returns:
        (final_post, metadata)
    """
    normalized_type = post_type.strip().lower()
    if normalized_type not in TEMPLATE_MAP:
        raise ValueError(
            f"Unsupported post_type '{post_type}'. "
            f"Use one of: {', '.join(sorted(TEMPLATE_MAP.keys()))}"
        )

    system_prompt = _load_prompt_file("system_prompt.txt")
    template_text = _load_prompt_file(TEMPLATE_MAP[normalized_type])
    feedback_guidance = str(config.get("feedback_guidance", "") or "")
    candidates = _generate_candidate_drafts(
        system_prompt=system_prompt,
        template_text=template_text,
        topic=topic,
        post_type=normalized_type,
        business_objective=business_objective,
        config=config,
        feedback_guidance=feedback_guidance,
    )
    if not candidates:
        raise RuntimeError("Failed to generate candidate drafts.")

    best_index, evaluator_metadata = evaluate_candidates_with_cohere(
        topic=topic,
        post_type=normalized_type,
        business_objective=business_objective,
        candidates=candidates,
        config=config,
    )
    selected = candidates[best_index]
    final_post = selected["text"]

    metadata = {
        "topic": topic,
        "post_type": normalized_type,
        "business_objective": business_objective,
        "rag_context_used": True,  # Added to metadata
        "prompt_files": {
            "system": "prompts/system_prompt.txt",
            "template": f"prompts/{TEMPLATE_MAP[normalized_type]}",
        },
        "candidate_generation": {
            "count": len(candidates),
            "angles": [candidate["angle"] for candidate in candidates],
            "selected_index": best_index,
            "selected_angle": selected["angle"],
            "evaluator": evaluator_metadata,
            "feedback_guidance_used": bool(feedback_guidance.strip()),
        },
        "candidates": candidates,
        "llm": {
            "model": selected["llm"].get("model"),
            "attempts": selected["llm"].get("attempts"),
            "usage": selected["llm"].get("usage", {}),
            "length": selected["llm"].get("length", {}),
            "estimated_cost_usd": selected["llm"].get("estimated_cost_usd", 0.0),
            "error": selected["llm"].get("error"),
        },
    }
    return final_post, metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a LinkedIn post with OpenAI")
    parser.add_argument("--topic", required=False, help="Post topic")
    parser.add_argument(
        "--post-type",
        required=False,
        choices=sorted(TEMPLATE_MAP.keys()),
        help="Type of post template to use",
    )
    parser.add_argument("--business-objective", required=False, help="Business objective for this post")
    parser.add_argument(
        "--model",
        default=OPENAI_MODEL_OPTIONS[0],
        choices=OPENAI_MODEL_OPTIONS,
        help="OpenAI model",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max output tokens")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts")
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0, help="Retry backoff")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Print metadata JSON only",
    )
    parser.add_argument(
        "--run-static-examples",
        action="store_true",
        help="Run built-in static examples (also used automatically when no required inputs are provided)",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    config: Dict[str, Any] = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "retries": args.retries,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "timeout": args.timeout,
    }
    if args.api_key:
        config["api_key"] = args.api_key

    use_static_examples = args.run_static_examples or not (
        args.topic and args.post_type and args.business_objective
    )
    has_api_key = bool(config.get("api_key") or os.getenv("OPENAI_API_KEY"))

    if use_static_examples:
        for idx, example in enumerate(STATIC_EXAMPLES, start=1):
            if has_api_key:
                post, metadata = generate_post(
                    topic=example["topic"],
                    post_type=example["post_type"],
                    business_objective=example["business_objective"],
                    config=config,
                )
            else:
                post = _build_mock_post(
                    topic=example["topic"],
                    post_type=example["post_type"],
                    business_objective=example["business_objective"],
                )
                metadata = {
                    "topic": example["topic"],
                    "post_type": example["post_type"],
                    "business_objective": example["business_objective"],
                    "rag_context_used": False,
                    "prompt_files": {
                        "system": "prompts/system_prompt.txt",
                        "template": f"prompts/{TEMPLATE_MAP[example['post_type']]}",
                    },
                    "llm": {
                        "model": "mock-static-example",
                        "attempts": 0,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                        "length": {"prompt_chars": 0, "completion_chars": len(post)},
                        "estimated_cost_usd": 0.0,
                        "error": None,
                    },
                }
            if args.metadata_only:
                print(json.dumps(metadata, indent=2))
            else:
                print(f"=== STATIC EXAMPLE {idx} ({example['post_type']}) ===")
                print(post)
                print("\n--- METADATA ---")
                print(json.dumps(metadata, indent=2))
                print()
        return

    if not has_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Provide --api-key or export OPENAI_API_KEY.")

    post, metadata = generate_post(
        topic=args.topic,
        post_type=args.post_type,
        business_objective=args.business_objective,
        config=config,
    )

    if args.metadata_only:
        print(json.dumps(metadata, indent=2))
        return

    print(post)
    print("\n--- METADATA ---")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()