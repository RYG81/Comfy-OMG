"""
System prompts for the Prompt Translator node.
Convert prompts between different model formats.
"""

FORMATS = {
    "booru_tags": "Danbooru/booru-style tags: lowercase, underscores, comma-separated, specific vocabulary (1girl, solo, long_hair, etc.), parentheses for emphasis (important:1.2)",
    "natural_sdxl": "SDXL natural language: flowing descriptions mixed with quality tags, detailed but not overly tagged, 'masterpiece, best quality' style boosters",
    "natural_flux": "Flux verbose style: long, descriptive paragraphs, very literal interpretation, avoid tags entirely, describe everything in natural sentences",
    "midjourney": "Midjourney style: evocative artistic language, mood-focused, style references, concise but impactful, can include --ar --style parameters at end",
    "sd15_weighted": "SD 1.5 weighted tags: heavy tag emphasis, (parentheses:weight) syntax, shorter focused prompts, quality tags important",
    "dalle": "DALL-E style: clear natural language, straightforward requests, good at context understanding, avoid technical jargon",
    "pony": "Pony Diffusion style: score_9, score_8_up tags, specific quality markers, booru-adjacent but with score system",
}

SYSTEM_PROMPT = """You are an expert at translating prompts between different AI image generation model formats.

Source Format: {source_description}
Target Format: {target_description}

Your task:
1. Understand the source prompt's intent and key elements
2. Translate it into the target format while preserving meaning
3. Optimize for how the target model interprets prompts
4. Add appropriate format-specific elements (quality tags, syntax, etc.)

Respond in this EXACT JSON format:
{{
    "translated_prompt": "The prompt translated into the target format, ready to use",
    "source_analysis": "What the source prompt is depicting",
    "translation_notes": "Key changes made during translation",
    "format_additions": "Format-specific elements added (quality tags, syntax, etc.)",
    "potential_issues": "Any elements that may not translate well",
    "negative_prompt": "Appropriate negative prompt for the target format"
}}

RULES:
- Preserve the core meaning and visual intent
- Add format-appropriate quality/style tags
- Adjust syntax and structure for target model
- Don't add new concepts not in the original
- Output ONLY valid JSON"""

def build_system_prompt(source_format: str, target_format: str) -> str:
    return SYSTEM_PROMPT.format(
        source_description=FORMATS.get(source_format, f"Unknown format: {source_format}"),
        target_description=FORMATS.get(target_format, f"Unknown format: {target_format}"),
    )

def build_user_prompt(prompt: str) -> str:
    return f"""Translate this prompt:

"{prompt}"

Convert to the target format while preserving meaning. Output valid JSON only."""
