"""
System prompts for the Background Generator node.
Generate detailed backgrounds/environments separately from subjects.
"""

BACKGROUND_TYPE = [
    "custom", "interior", "exterior", "natural landscape", "urban", "fantasy",
    "sci-fi", "abstract", "gradient", "studio", "historical", "underwater",
    "aerial", "space", "surreal", "minimalist", "detailed", "apocalyptic",
]

ARCHITECTURE_STYLE = [
    "custom", "modern", "contemporary", "minimalist", "industrial", "rustic",
    "victorian", "art deco", "art nouveau", "gothic", "baroque", "renaissance",
    "medieval", "ancient greek", "ancient roman", "ancient egyptian", "japanese",
    "chinese", "middle eastern", "futuristic", "cyberpunk", "steampunk", "organic",
]

NATURE_TYPE = [
    "custom", "forest", "jungle", "meadow", "field", "mountain", "cliff",
    "beach", "ocean", "lake", "river", "waterfall", "desert", "canyon",
    "tundra", "arctic", "volcanic", "cave", "garden", "orchard", "vineyard",
]

URBAN_TYPE = [
    "custom", "downtown", "suburbs", "slums", "rooftop", "alley", "street",
    "plaza", "market", "industrial district", "port", "train station",
    "airport", "highway", "bridge", "tunnel", "parking garage", "mall",
]

ROOM_TYPE = [
    "custom", "living room", "bedroom", "bathroom", "kitchen", "dining room",
    "office", "study", "library", "studio", "workshop", "garage", "attic",
    "basement", "hallway", "staircase", "balcony", "porch", "rooftop terrace",
    "throne room", "ballroom", "dungeon", "laboratory", "spaceship interior",
]

DEPTH_COMPLEXITY = [
    "custom", "simple - flat background", "shallow - single layer",
    "medium - foreground and background", "deep - multiple layers",
    "complex - intricate depth", "infinite - endless vista",
]

POPULATION = [
    "custom", "empty - no one", "sparse - few distant figures",
    "moderate - some activity", "busy - crowded", "packed - dense crowd",
]

SYSTEM_PROMPT = """You are an expert environment designer and background artist. You create detailed, immersive background descriptions optimized for AI image generation.

Design a background with these parameters:
- Type: {bg_type}
- Architecture Style: {architecture}
- Nature Elements: {nature}
- Urban Elements: {urban}
- Room Type (if interior): {room}
- Depth/Complexity: {depth}
- Population: {population}
- Time of Day: {time_of_day}
- Weather: {weather}
- Season: {season}
- Mood: {mood}
- Color Palette: {colors}

{custom_instructions}

Respond in this EXACT JSON format:
{{
    "background_prompt": "Complete background description ready to use, with no subject/character references",
    "foreground_elements": "Elements in the foreground that frame the scene",
    "midground_elements": "Main environmental elements in the middle distance",
    "background_elements": "Distant elements and sky/horizon",
    "atmospheric_effects": "Fog, haze, particles, lighting effects",
    "color_description": "Color palette and lighting colors",
    "depth_cues": "Elements that create sense of depth",
    "mood_elements": "What creates the emotional atmosphere",
    "integration_tips": "How to integrate a subject into this background",
    "negative_prompt": "What to avoid for this background"
}}

IMPORTANT:
- NO characters or subjects in the background description
- Focus purely on environment and setting
- Include enough detail for the AI to render convincingly
- Output ONLY valid JSON"""

def build_system_prompt(**kwargs) -> str:
    return SYSTEM_PROMPT.format(
        bg_type=kwargs.get("bg_type", "custom"),
        architecture=kwargs.get("architecture", "custom"),
        nature=kwargs.get("nature", "custom"),
        urban=kwargs.get("urban", "custom"),
        room=kwargs.get("room", "custom"),
        depth=kwargs.get("depth", "medium - foreground and background"),
        population=kwargs.get("population", "empty - no one"),
        time_of_day=kwargs.get("time_of_day", "custom"),
        weather=kwargs.get("weather", "custom"),
        season=kwargs.get("season", "custom"),
        mood=kwargs.get("mood", "custom"),
        colors=kwargs.get("colors", "custom"),
        custom_instructions=kwargs.get("custom_instructions", ""),
    )

def build_user_prompt(description: str = "") -> str:
    if description.strip():
        return f"Create this background environment:\n\n{description}\n\nOutput valid JSON only."
    return "Create a detailed background environment based on the specified parameters. Output valid JSON only."
