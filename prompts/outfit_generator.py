"""
System prompts for the Outfit Generator node.
"""

OUTFIT_TYPES = {
    "casual": "Everyday casual wear — comfortable, relaxed, modern streetwear or athleisure",
    "formal": "Formal attire — business suits, evening gowns, professional wear",
    "fantasy": "Fantasy costumes — medieval, elvish, magical robes, armor, cloaks",
    "sci-fi": "Science fiction attire — futuristic bodysuits, space suits, cybernetic enhancements",
    "historical": "Historical costumes — period-accurate clothing from specific eras",
    "uniform": "Uniforms — military, school, work uniforms, service industry",
    "costume": "Costumes and cosplay — character costumes, themed outfits",
    "streetwear": "Urban streetwear — trendy, hypebeast, urban fashion",
    "haute_couture": "High fashion — avant-garde, runway, designer pieces",
}

SEASONS = {
    "spring": "Light layers, pastels, transitional pieces, floral patterns",
    "summer": "Light fabrics, bright colors, minimal layers, breathable materials",
    "fall": "Warm layers, earth tones, cozy textures, transitional outerwear",
    "winter": "Heavy layers, dark colors, warm materials, coats and boots",
    "any": "Season-neutral, versatile pieces",
}

SYSTEM_PROMPT = """You are a fashion designer and costume creator. You design detailed, visually striking outfits that work perfectly for AI image generation.

Character: {character_description}
Outfit Type: {outfit_type_description}
Season: {season_description}
Color Scheme Preference: {color_scheme}
{reference_instruction}

Design a complete outfit from head to toe. Be specific about:
- Materials and textures (silk, leather, denim, etc.)
- Exact colors and patterns
- How pieces fit (loose, fitted, oversized)
- Small details that add character

Respond in this EXACT JSON format:
{{
    "outfit_prompt": "Complete outfit description as a single prompt — integrate all pieces naturally for image generation",
    "top_description": "Upper body clothing — shirts, jackets, coats, etc. with full detail",
    "bottom_description": "Lower body clothing — pants, skirts, shorts with full detail",
    "accessories": "All accessories — jewelry, bags, belts, hats, scarves, watches, glasses",
    "footwear": "Shoes, boots, sandals — style, color, heel height, condition",
    "outfit_tags": "Comma-separated tags for the complete look",
    "style_notes": "Overall style direction and mood of the outfit",
    "color_palette": "Main colors used in the outfit"
}}

Output ONLY valid JSON. Be creative but coherent — all pieces should work together."""

def build_system_prompt(
    character_description: str,
    outfit_type: str = "casual",
    season: str = "any",
    color_scheme: str = "",
    has_reference: bool = False
) -> str:
    reference_instruction = "A reference image has been provided. Take inspiration from its style and elements while creating something appropriate for the character." if has_reference else ""
    return SYSTEM_PROMPT.format(
        character_description=character_description,
        outfit_type_description=OUTFIT_TYPES.get(outfit_type, OUTFIT_TYPES["casual"]),
        season_description=SEASONS.get(season, SEASONS["any"]),
        color_scheme=color_scheme if color_scheme.strip() else "Designer's choice — pick colors that suit the character and style",
        reference_instruction=reference_instruction,
    )

def build_user_prompt(has_reference: bool = False) -> str:
    if has_reference:
        return "Using the reference image for inspiration, design a complete outfit for this character. Output valid JSON only."
    return "Design a complete, detailed outfit for this character. Output valid JSON only."
