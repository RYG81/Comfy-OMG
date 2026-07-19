"""
Prompt templates for turning image analysis into model-specific edit instructions.
"""

IMAGE_EDIT_MODELS = (
    "FLUX 2",
    "Nano Banana",
    "ChatGPT Image",
    "Qwen-Image-Edit-2511",
    "FLUX Klein 9B",
    "General Image Edit",
)

MODEL_EDIT_GUIDANCE = {
    "FLUX 2": (
        "Use one dense, visually concrete paragraph. Describe the requested result "
        "as a coherent final image, integrate the edit naturally, and explicitly "
        "preserve unchanged identity, composition, camera, lighting, and style."
    ),
    "Nano Banana": (
        "Use direct natural editing language. Clearly identify what changes, what "
        "it becomes, where the change occurs, and what must remain exactly unchanged. "
        "Keep the instruction unambiguous and avoid prompt-tag lists."
    ),
    "ChatGPT Image": (
        "Use clear declarative natural language describing the finished image. "
        "Resolve spatial relationships, materials, visible text, lighting behavior, "
        "and preservation constraints explicitly."
    ),
    "Qwen-Image-Edit-2511": (
        "Use a precise localized edit instruction with concrete target attributes "
        "and spatial references. Separate the intended visual change from preservation "
        "requirements through fluent sentences, not labels."
    ),
    "FLUX Klein 9B": (
        "Use a compact but information-dense paragraph. Prioritize the requested "
        "change, subject identity, essential scene geometry, camera, and lighting; "
        "avoid repetition and decorative quality-tag spam."
    ),
    "General Image Edit": (
        "Use one clear natural-language paragraph describing the desired final image, "
        "the requested changes, and the important source details that must be preserved."
    ),
}

SYSTEM_PROMPT = """You are an expert prompt engineer for instruction-based AI image editing.

You receive a structured analysis of a source image and a user's requested changes. Produce an edit prompt for {target_model}.

Model-specific guidance:
{model_guidance}

Rules:
- Apply every requested change accurately.
- Preserve every analyzed detail the user did not ask to change, especially subject identity, facial traits, body proportions, clothing, pose, location, object layout, camera angle, framing, lens perspective, depth of field, lighting direction, shadow behavior, palette, art style, mood, and technical finish.
- When a requested change conflicts with the source analysis, the requested change wins only for that specific attribute.
- Resolve references such as "her clothing", "the background", or "the left object" from the analysis where possible.
- Do not invent unrelated changes.
- The edit_prompt must describe the intended final image, not discuss your reasoning.
- The edit_prompt must be one dense, fluent natural-language paragraph.
- Never put field labels or headings such as "Subject:", "Change:", "Camera:", "Lighting:", or "Preserve:" inside edit_prompt.
- Do not use markdown or bullet points inside edit_prompt.

Return ONLY valid JSON in this exact structure:
{{
    "edit_prompt": "One dense natural-language paragraph ready for the selected image-edit model",
    "preservation_notes": "Concise details that must remain unchanged",
    "change_summary": "Concise summary of the requested visual changes"
}}

All values must be strings."""


def build_system_prompt(target_model: str) -> str:
    guidance = MODEL_EDIT_GUIDANCE.get(
        target_model, MODEL_EDIT_GUIDANCE["General Image Edit"]
    )
    return SYSTEM_PROMPT.format(
        target_model=target_model,
        model_guidance=guidance,
    )


def build_user_prompt(analyzed_data: str, requested_changes: str, target_model: str) -> str:
    return (
        f"Target image-edit model: {target_model}\n\n"
        "SOURCE IMAGE ANALYSIS:\n"
        f"{analyzed_data.strip()}\n\n"
        "REQUESTED CHANGES:\n"
        f"{requested_changes.strip()}\n\n"
        "Create the final edit instruction now. Preserve all source attributes not "
        "explicitly changed. Return valid JSON only."
    )
