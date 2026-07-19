"""
System prompts for the Lighting Designer node.
Detailed lighting setup descriptions.
"""

LIGHTING_SCENARIO = [
    "custom", "studio portrait", "outdoor daylight", "golden hour", "blue hour",
    "night scene", "indoor ambient", "stage lighting", "cinematic", "noir",
    "horror", "romantic", "dramatic", "product photography", "food photography",
    "fashion", "sports", "concert", "underwater", "space/sci-fi", "magical/fantasy",
]

KEY_LIGHT_POSITION = [
    "custom", "front center", "front left 45°", "front right 45°",
    "side left 90°", "side right 90°", "back left 135°", "back right 135°",
    "directly above", "directly below", "behind (backlight)",
]

KEY_LIGHT_QUALITY = [
    "custom", "hard (direct sun/spot)", "soft (large diffused)",
    "medium (partial diffusion)", "dappled (through leaves)",
    "patterned (through blinds)", "reflected", "bounced",
]

FILL_LIGHT = [
    "custom", "none (harsh shadows)", "minimal (1:4 ratio)",
    "subtle (1:3 ratio)", "moderate (1:2 ratio)", "strong (1:1.5 ratio)",
    "flat (equal to key)", "reflector bounce", "ambient fill",
]

RIM_LIGHT = [
    "custom", "none", "subtle edge", "strong rim", "double rim (both sides)",
    "hair light only", "full silhouette edge", "colored rim",
]

PRACTICAL_LIGHTS = [
    "custom", "none", "candles", "fireplace", "lamps", "neon signs",
    "streetlights", "car headlights", "phone/screen glow", "flashlight",
    "lantern", "fairy lights", "torches", "bioluminescence",
]

COLOR_TEMPERATURE = [
    "custom", "warm (2700K - candle)", "warm (3200K - tungsten)",
    "neutral (4000K)", "daylight (5600K)", "cool (6500K)",
    "very cool (8000K - shade)", "mixed warm/cool", "golden hour warm",
    "blue hour cool", "artificial mixed",
]

SHADOW_CHARACTER = [
    "custom", "crisp defined edges", "soft gradual falloff",
    "multiple overlapping", "colored shadows", "transparent shadows",
    "deep black shadows", "no visible shadows (flat)",
]

VOLUMETRIC = [
    "custom", "none", "subtle haze", "fog", "mist", "smoke",
    "dust particles", "god rays", "light beams", "underwater caustics",
    "steam", "magical particles",
]

SYSTEM_PROMPT = """You are a professional lighting director and cinematographer. You design detailed, production-quality lighting setups.

Design a lighting setup for: {scenario}

Parameters:
- Key Light Position: {key_position}
- Key Light Quality: {key_quality}
- Fill Light: {fill_light}
- Rim/Back Light: {rim_light}
- Practical Lights: {practical}
- Color Temperature: {color_temp}
- Shadow Character: {shadow_char}
- Volumetric/Atmosphere: {volumetric}
- Overall Mood: {mood}

{custom_instructions}

Respond in this EXACT JSON format:
{{
    "lighting_prompt": "Complete lighting description ready to append to any prompt",
    "technical_setup": "Technical breakdown — key light, fill, rim, ratios, positions",
    "color_palette": "Light colors and temperatures throughout the scene",
    "shadow_description": "How shadows fall, their quality and depth",
    "atmosphere_effects": "Volumetric lighting, particles, atmospheric elements",
    "mood_contribution": "How the lighting supports the emotional mood",
    "time_of_day_match": "What time of day this lighting suggests",
    "style_tags": "Comma-separated lighting tags for prompts",
    "avoid": "What to avoid with this lighting setup"
}}

Output ONLY valid JSON."""

def build_system_prompt(
    scenario: str, key_position: str, key_quality: str, fill_light: str,
    rim_light: str, practical: str, color_temp: str, shadow_char: str,
    volumetric: str, mood: str, custom_instructions: str = "",
) -> str:
    return SYSTEM_PROMPT.format(
        scenario=scenario, key_position=key_position, key_quality=key_quality,
        fill_light=fill_light, rim_light=rim_light, practical=practical,
        color_temp=color_temp, shadow_char=shadow_char, volumetric=volumetric,
        mood=mood,
        custom_instructions=f"Additional Instructions: {custom_instructions}" if custom_instructions.strip() else "",
    )

def build_user_prompt(subject: str = "") -> str:
    subj = f"Subject being lit: {subject}\n\n" if subject.strip() else ""
    return f"""{subj}Design a complete lighting setup. Output valid JSON only."""
