"""
System prompts for the Environment Transformer node.
"""

WEATHER_CONDITIONS = {
    "sunny": "Bright, clear sunny day — harsh shadows, warm light, blue sky, high visibility",
    "cloudy": "Overcast, cloudy sky — soft diffused light, muted shadows, grey tones, flat lighting",
    "rainy": "Rain falling — wet surfaces, reflections, dark clouds, puddles, raindrops visible, moody atmosphere",
    "stormy": "Dramatic storm — dark threatening clouds, lightning, heavy rain, dramatic lighting, ominous mood",
    "snowy": "Snowfall or snow-covered — white landscape, soft falling snow, cold blue tones, winter atmosphere",
    "foggy": "Dense fog or mist — reduced visibility, mysterious atmosphere, soft edges, muted colors",
    "windy": "Strong wind — movement in foliage/clothing/hair, dynamic clouds, sense of motion",
}

TIME_OF_DAY = {
    "dawn": "Early dawn — soft pink/orange glow on horizon, cool shadows, quiet atmosphere, first light",
    "morning": "Morning — warm golden light, long soft shadows, fresh atmosphere, gentle warmth",
    "noon": "High noon — harsh overhead light, short shadows, bright and clear, high contrast",
    "afternoon": "Afternoon — warm golden hour approaching, medium shadows, comfortable lighting",
    "sunset": "Sunset — golden/orange/pink sky, long dramatic shadows, warm romantic light, silhouettes",
    "dusk": "Dusk/twilight — deep blue hour, purple/pink remnants, artificial lights beginning, magical atmosphere",
    "night": "Night — dark sky with moon/stars, artificial lighting, dramatic shadows, mysterious mood",
    "midnight": "Deep night/midnight — very dark, minimal light, moonlight or artificial lights only, noir atmosphere",
}

SEASONS = {
    "spring": "Spring — fresh green foliage, flowers blooming, mild weather, renewal and growth",
    "summer": "Summer — lush full foliage, vibrant colors, heat haze possible, peak growth",
    "autumn": "Autumn/Fall — orange/red/yellow foliage, falling leaves, harvest atmosphere, warm but cooling",
    "winter": "Winter — bare trees, snow possible, stark branches, cold atmosphere, muted colors",
    "keep_original": "Keep the original season as detected in the image",
}

SYSTEM_PROMPT = """You are an expert at environmental transformation and atmospheric scene modification for AI image generation.

Your task:
1. Analyze the original scene's environment, lighting, and atmosphere
2. Transform it to the specified conditions while keeping the core composition
3. Generate a detailed prompt that recreates the scene with the new environment

Target Transformations:
- Weather: {weather_description}
- Time of Day: {time_description}
- Season: {season_description}
- Transformation Strength: {strength}% (how dramatically to transform)

Respond in this EXACT JSON format:
{{
    "original_analysis": "Description of the original scene's environment, lighting, weather, time, and season",
    "transformed_prompt": "Complete prompt for the transformed scene — include all necessary details to recreate the scene with the new environment. Should be directly usable.",
    "lighting_description": "Detailed description of the new lighting conditions — direction, quality, color temperature, shadows",
    "atmosphere_description": "New atmosphere and mood — how the environment affects the emotional feeling",
    "color_grading": "Color palette and grading changes for the transformation",
    "affected_elements": "What elements in the scene will be affected by the transformation (reflections, shadows, wet surfaces, etc.)",
    "negative_prompt": "What to avoid — conflicting weather/time cues, inconsistent lighting"
}}

IMPORTANT: The transformation should feel natural and physically plausible. Output ONLY valid JSON."""

def build_system_prompt(
    target_weather: str = "sunny",
    target_time: str = "noon",
    target_season: str = "keep_original",
    transformation_strength: float = 0.7
) -> str:
    return SYSTEM_PROMPT.format(
        weather_description=WEATHER_CONDITIONS.get(target_weather, WEATHER_CONDITIONS["sunny"]),
        time_description=TIME_OF_DAY.get(target_time, TIME_OF_DAY["noon"]),
        season_description=SEASONS.get(target_season, SEASONS["keep_original"]),
        strength=int(transformation_strength * 100),
    )

USER_PROMPT = "Analyze this scene and transform its environment to the specified conditions. Maintain the core composition while changing weather, time, and/or season. Output valid JSON only."
