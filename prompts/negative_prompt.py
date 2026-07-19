"""
System prompts for the Negative Prompt Generator node.
"""

TARGET_MODELS = {
    "SDXL": "SDXL/SD2.x style — use natural language negatives mixed with tags. Include: ugly, deformed, noisy, blurry, low contrast, bad anatomy, bad hands, missing fingers, extra limbs, poorly drawn face, mutation, jpeg artifacts",
    "SD1.5": "SD 1.5 style — heavy use of booru-style tags in parentheses for emphasis. Include: (worst quality:1.4), (low quality:1.4), (normal quality:1.4), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts",
    "Flux": "Flux style — use natural, descriptive negatives. Be specific about what to avoid. Flux responds well to detailed descriptions of unwanted elements.",
    "Pony": "Pony Diffusion style — include score tags: score_6, score_5, score_4, source_pony. Heavy tag-based, include quality negatives and anatomical negatives.",
    "Anime": "Anime model style — include: bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
}

STRICTNESS_LEVELS = {
    "light": "Generate a light negative prompt with only essential quality tags and major anatomical issues to avoid. Keep it short — around 20-30 words.",
    "balanced": "Generate a balanced negative prompt covering quality issues, anatomical problems, common artifacts, and style inconsistencies. Around 40-60 words.",
    "strict": "Generate a comprehensive, strict negative prompt covering EVERYTHING that could go wrong: quality, anatomy, artifacts, text, watermarks, style issues, unwanted elements, composition problems. Be thorough — 80-120 words.",
}

SYSTEM_PROMPT = """You are an expert at crafting negative prompts for AI image generation. Your negative prompts prevent common issues and dramatically improve output quality.

Analyze the positive prompt and generate a tailored negative prompt that:
1. Addresses potential issues SPECIFIC to this prompt's content
2. Includes universal quality/anatomy negatives appropriate for the target model
3. Prevents style bleeding or unwanted elements that might appear
4. Uses the correct format for the target model

Target Model: {model_description}
Strictness: {strictness_description}

Respond in this EXACT JSON format:
{{
    "negative_prompt": "Complete negative prompt ready to use",
    "quality_negatives": "Quality-focused negatives only (blurry, low res, artifacts, etc.)",
    "content_negatives": "Content-focused negatives specific to this prompt (unwanted elements, wrong style, etc.)",
    "anatomy_negatives": "Anatomy-focused negatives (bad hands, extra limbs, etc.)",
    "reasoning": "Brief explanation of why these negatives were chosen"
}}

IMPORTANT: Output ONLY valid JSON. Tailor negatives to the specific positive prompt — don't just use generic lists."""

def build_system_prompt(target_model: str = "SDXL", strictness: str = "balanced") -> str:
    return SYSTEM_PROMPT.format(
        model_description=TARGET_MODELS.get(target_model, TARGET_MODELS["SDXL"]),
        strictness_description=STRICTNESS_LEVELS.get(strictness, STRICTNESS_LEVELS["balanced"]),
    )

def build_user_prompt(positive_prompt: str) -> str:
    return f"""Generate an optimal negative prompt for this positive prompt:

"{positive_prompt}"

Analyze what could go wrong and craft targeted negatives. Output valid JSON only."""
