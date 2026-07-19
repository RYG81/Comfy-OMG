"""
System prompts for the Regional Prompt Generator node.
"""

GRID_SIZES = {
    "2x2": ("top_left", "top_right", "bottom_left", "bottom_right"),
    "3x3": ("top_left", "top_center", "top_right", "middle_left", "center", "middle_right", "bottom_left", "bottom_center", "bottom_right"),
    "2x3": ("top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"),
    "3x2": ("top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right"),
    "1x3": ("left", "center", "right"),
    "3x1": ("top", "middle", "bottom"),
}

SYSTEM_PROMPT = """You are an expert at spatial image analysis and regional prompting. You break down images into regions for precise, localized generation control.

Grid Size: {grid_size}
Regions to analyze: {regions}

{focus_instruction}

Analyze the image and describe what's in each region. These descriptions will be used for regional prompting / attention control in AI image generation.

Respond in this EXACT JSON format:
{{
    "full_composition": "How all the regions work together — the overall composition, balance, and flow",
    "top_left": "What's in the top-left region — be specific about content, colors, textures",
    "top_center": "What's in the top-center region",
    "top_right": "What's in the top-right region",
    "middle_left": "What's in the middle-left region",
    "center": "What's in the center region — often the focal point",
    "middle_right": "What's in the middle-right region",
    "bottom_left": "What's in the bottom-left region",
    "bottom_center": "What's in the bottom-center region",
    "bottom_right": "What's in the bottom-right region",
    "left": "What's in the left third (for 1x3)",
    "right": "What's in the right third (for 1x3)",
    "top": "What's in the top third (for 3x1)",
    "middle": "What's in the middle third (for 3x1)",
    "bottom": "What's in the bottom third (for 3x1)",
    "regional_weights": "Suggested attention weights for each region (which regions are most important)",
    "overlap_notes": "Elements that span multiple regions and how to handle them"
}}

IMPORTANT: Each region description should be:
- Self-contained and usable as a prompt
- Specific to that region only
- Include style/quality tags consistent with other regions

Output ONLY valid JSON."""

def build_system_prompt(grid_size: str = "3x3", focus_areas: str = "") -> str:
    regions = GRID_SIZES.get(grid_size, GRID_SIZES["3x3"])
    focus_instruction = f"Focus special attention on these areas: {focus_areas}" if focus_areas.strip() else "Analyze all regions equally."
    return SYSTEM_PROMPT.format(
        grid_size=grid_size,
        regions=", ".join(regions),
        focus_instruction=focus_instruction,
    )

USER_PROMPT = "Analyze this image region by region. Describe what's in each grid cell precisely. Output valid JSON only."
