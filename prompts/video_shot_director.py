"""
Prompt helpers for video shot direction.
"""

SHOT_COUNTS = ["3 shots", "4 shots", "5 shots", "6 shots", "8 shots", "10 shots"]
VIDEO_MODELS = ["LTXV 2.3", "Wan 2.2", "general video model"]
SEQUENCE_STYLE = [
    "single scene coverage", "cinematic trailer beat", "music video sequence",
    "product showcase", "character action beat", "emotional dialogue beat",
    "establishing-to-detail sequence", "loopable sequence",
]

SYSTEM_PROMPT = """You are a video director and text-to-video prompt planner.

Convert the provided concept or scene prompt into a shot-by-shot video prompt sequence.

Controls:
- Target model: {target_model}
- Shot count: {shot_count}
- Sequence style: {sequence_style}
- Total duration: {total_duration}
- Custom guidance: {custom_system_prompt}

Respond in this EXACT JSON format:
{{
    "sequence_prompt": "A complete prompt describing the full video sequence with shot progression and continuity.",
    "shot_list": "Numbered shots with duration, framing, camera movement, subject action, lighting, and transition notes.",
    "per_shot_prompts": "Separate compact prompt for each shot, numbered clearly.",
    "continuity_notes": "What must remain consistent across shots: identity, wardrobe, lighting, geography, props, style.",
    "negative_prompt": "Video artifacts and sequence problems to avoid.",
    "model_notes": "Practical notes for the selected video model."
}}

Rules:
- Preserve character and setting continuity across every shot.
- Keep each shot simple enough for text-to-video generation.
- Prefer coherent camera language over editing jargon.
- Output ONLY valid JSON."""


def build_system_prompt(
    target_model: str, shot_count: str, sequence_style: str,
    total_duration: str, custom_system_prompt: str,
) -> str:
    return SYSTEM_PROMPT.format(
        target_model=target_model,
        shot_count=shot_count,
        sequence_style=sequence_style,
        total_duration=total_duration or "infer from shot count",
        custom_system_prompt=custom_system_prompt.strip() or "No additional guidance.",
    )


def build_user_prompt(scene_prompt: str) -> str:
    return f"""Create a shot-by-shot video prompt sequence from this source:

{scene_prompt}

Output valid JSON only."""
