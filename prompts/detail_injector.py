"""
System prompts for the Detail Injector node.
Add specific types of details to existing prompts.
"""

DETAIL_TYPE = [
    "texture", "material", "pattern", "wear_and_tear", "lighting_details",
    "reflections", "shadows", "micro_details", "macro_details", "atmospheric",
    "color_variations", "surface_quality", "aging_effects", "environmental",
    "fabric_details", "skin_details", "hair_details", "eye_details",
    "architectural_details", "nature_details", "mechanical_details", "magical_effects",
]

DETAIL_INTENSITY = [
    "subtle - barely noticeable enhancement",
    "light - gentle addition of details",
    "moderate - noticeable but balanced",
    "heavy - significant detail emphasis",
    "extreme - hyper-detailed focus",
]

SYSTEM_PROMPT = """You are an expert at enhancing image generation prompts with specific, targeted details.

Your task: Inject {detail_type} details into the provided prompt.
Detail Intensity: {intensity}

Types of details you might add:
- Texture: Surface texture descriptions (rough, smooth, bumpy, silky, etc.)
- Material: Specific material callouts (leather grain, metal sheen, fabric weave)
- Pattern: Patterns and designs (floral, geometric, organic, etc.)
- Wear and Tear: Signs of use, age, damage (scratches, fading, patina)
- Lighting Details: How light interacts (highlights, subsurface scattering)
- Reflections: Reflective surfaces and their behavior
- Micro Details: Tiny details (pores, threads, dust particles)
- Atmospheric: Environmental particles and effects

Respond in this EXACT JSON format:
{{
    "enhanced_prompt": "The original prompt with injected details, naturally integrated",
    "added_details": "List of specific details that were added",
    "detail_placement": "Where in the prompt details were injected",
    "detail_tags": "Comma-separated detail tags for tag-based models",
    "quality_boost": "Additional quality tags that complement these details",
    "negative_additions": "Negatives to prevent conflicting with added details"
}}

RULES:
- Integrate details naturally into the existing prompt
- Don't change the core subject or composition
- Match the style and tone of the original prompt
- Add details that enhance without overwhelming
- Output ONLY valid JSON"""

def build_system_prompt(detail_types: list[str], intensity: str = "moderate") -> str:
    detail_str = ", ".join(detail_types) if detail_types else "general"
    return SYSTEM_PROMPT.format(
        detail_type=detail_str,
        intensity=intensity,
    )

def build_user_prompt(prompt: str) -> str:
    return f"""Inject details into this prompt:

"{prompt}"

Add the specified type of details at the chosen intensity. Output valid JSON only."""
