"""
System prompts for Advanced Scene Director with comprehensive options.
"""

# ═══════════════════════════════════════════════════════════════════
# ALL DROPDOWN OPTIONS
# ═══════════════════════════════════════════════════════════════════

SCENE_TYPE = [
    "custom", "portrait", "group portrait", "action scene", "battle scene",
    "romantic scene", "conversation", "confrontation", "celebration",
    "mourning", "discovery", "chase scene", "escape scene", "reunion",
    "farewell", "transformation", "revelation", "flashback", "dream sequence",
    "nightmare", "montage", "establishing shot", "dramatic moment",
]

CHARACTER_RELATIONSHIP = [
    "custom", "strangers", "acquaintances", "friends", "best friends",
    "rivals", "enemies", "lovers", "couple", "married", "ex-lovers",
    "family - parent and child", "family - siblings", "family - extended",
    "mentor and student", "boss and employee", "colleagues", "teammates",
    "adversaries", "allies", "hero and villain", "predator and prey",
]

SCENE_ACTION = [
    "custom", "standing together", "walking together", "sitting together",
    "talking", "arguing", "fighting", "dancing", "embracing", "kissing",
    "shaking hands", "bowing", "kneeling", "running", "chasing",
    "hiding", "searching", "working", "playing", "eating", "drinking",
    "celebrating", "mourning", "praying", "meditating", "training",
    "performing", "watching", "waiting", "arriving", "departing",
]

EMOTIONAL_TENSION = [
    "custom", "none - peaceful", "low - comfortable", "mild - curious",
    "moderate - uncertain", "building - anticipation", "high - conflict",
    "intense - crisis", "explosive - climax", "releasing - resolution",
    "aftermath - exhaustion", "bittersweet", "triumphant", "devastating",
]

SCENE_COMPOSITION = [
    "custom", "symmetrical - balanced", "asymmetrical - dynamic",
    "triangular - stable", "diagonal - tension", "circular - unity",
    "spiral - movement", "layered - depth", "framed - focus",
    "leading lines - direction", "rule of thirds", "golden ratio",
    "centered - confrontation", "scattered - chaos", "grouped - unity",
]

ENVIRONMENT_SCALE = [
    "custom", "intimate - small enclosed space", "room-sized - interior",
    "building-sized - multiple rooms", "street-level - outdoor urban",
    "neighborhood - wider area", "city-scale - skyline visible",
    "landscape - vast outdoor", "epic - massive scale", "cosmic - infinite",
]

ENVIRONMENT_CONDITION = [
    "custom", "pristine - perfect", "lived-in - comfortable",
    "worn - aged", "damaged - battle-scarred", "ruined - destroyed",
    "overgrown - reclaimed by nature", "abandoned - forgotten",
    "under construction", "futuristic - advanced", "magical - enchanted",
]

FOREGROUND_ELEMENTS = [
    "custom", "none", "grass and flowers", "rocks and stones",
    "water - puddles", "debris", "furniture", "foliage",
    "fallen leaves", "snow", "sand", "mist/fog", "flames",
    "magical particles", "characters partially visible",
]

BACKGROUND_ELEMENTS = [
    "custom", "blurred", "detailed", "sky", "mountains", "cityscape",
    "forest", "ocean", "stars", "clouds", "buildings", "crowd",
    "battle in distance", "sunset/sunrise", "storm clouds",
]

LIGHTING_SETUP = [
    "custom", "three-point lighting", "single key light",
    "high-key - bright overall", "low-key - dramatic shadows",
    "silhouette - backlit", "rim lighting - edge glow",
    "split lighting - half shadow", "Rembrandt - classic portrait",
    "butterfly - glamour", "natural ambient", "mixed sources",
    "colored lighting - stylized", "practical lights - in-scene",
]

LIGHT_COLOR = [
    "custom", "warm white", "cool white", "golden", "orange",
    "red", "blue", "cyan", "green", "purple", "pink",
    "mixed warm and cool", "neon multicolor", "fire/candlelight",
    "moonlight blue", "sunset gradient", "magical glow",
]

SHADOW_INTENSITY = [
    "custom", "no shadows - flat", "soft shadows - diffused",
    "medium shadows - natural", "hard shadows - dramatic",
    "deep shadows - noir", "colored shadows", "multiple shadow directions",
]

CAMERA_MOVEMENT_FEEL = [
    "custom", "static - stable", "handheld - organic",
    "dolly - smooth movement", "crane - elevated motion",
    "steadicam - fluid", "drone - aerial", "action cam - immersive",
    "locked off - formal", "whip pan - energetic", "slow zoom - tension",
]

LENS_EFFECTS = [
    "custom", "none - clean", "lens flare", "chromatic aberration",
    "vignette", "film grain", "motion blur", "radial blur",
    "tilt-shift - miniature", "anamorphic - cinematic",
    "fisheye - distorted", "soft focus - dreamy", "split diopter",
]

POST_PROCESSING = [
    "custom", "none - natural", "color graded - cinematic",
    "desaturated", "high contrast", "low contrast",
    "warm grade", "cool grade", "teal and orange", "cross-processed",
    "vintage film", "noir - black and white", "sepia", "bleach bypass",
]

SYSTEM_PROMPT = """You are an expert film director and cinematographer creating detailed scene compositions.

You must design a complete, professional scene with precise attention to:
1. Character placement and spatial relationships
2. Emotional dynamics and interactions
3. Environmental storytelling
4. Cinematic lighting and atmosphere
5. Camera work and visual composition

Scene Parameters:
- Scene Type: {scene_type}
- Number of Characters: {num_characters}
- Character Relationship: {relationship}
- Scene Action: {scene_action}
- Emotional Tension: {emotional_tension}
- Environment: {environment}
- Environment Scale: {env_scale}
- Environment Condition: {env_condition}
- Time of Day: {time_of_day}
- Weather: {weather}
- Lighting Setup: {lighting_setup}
- Light Color: {light_color}
- Shadow Intensity: {shadow_intensity}
- Camera Shot: {camera_shot}
- Camera Angle: {camera_angle}
- Camera Movement Feel: {camera_movement}
- Lens Effects: {lens_effects}
- Post-Processing: {post_processing}
- Mood/Atmosphere: {mood}
- Art Style: {art_style}
- Composition: {composition}
- Foreground: {foreground}
- Background: {background}

{custom_instructions}

Respond in this EXACT JSON format:
{{
    "scene_prompt": "Complete, detailed prompt for the entire scene. Include every character, their positions, interactions, environment, lighting, atmosphere, and technical camera details. This should be directly usable in an AI image generator.",
    "character_breakdown": "Individual description of each character — appearance, position, pose, expression, action",
    "spatial_layout": "Precise spatial arrangement — who is where, facing which direction, distances between characters, depth layers",
    "lighting_description": "Complete lighting breakdown — key light, fill, rim, ambient, color temperatures, shadow patterns",
    "camera_technical": "Camera specifications — shot type, angle, lens, focus, any movement implied",
    "atmosphere_details": "Atmospheric elements — particles, weather effects, mood indicators, environmental storytelling",
    "composition_notes": "How the frame is composed — visual flow, focal points, balance, leading lines",
    "color_script": "Color palette and grading — dominant colors, accent colors, overall color mood",
    "negative_prompt": "What to avoid — character merging, inconsistent lighting, style breaks",
    "director_vision": "The emotional and narrative intent — what story this moment tells"
}}

CRITICAL RULES:
- For multiple characters, be EXTREMELY specific about positions to prevent merging
- Use precise spatial language: 'left third of frame', 'foreground center', 'background right'
- Describe each character's distinct appearance to help AI differentiate them
- Match all elements cohesively — lighting should match time/weather, mood should match action
- Output ONLY valid JSON"""

def build_system_prompt(
    scene_type: str = "portrait",
    num_characters: int = 1,
    relationship: str = "custom",
    scene_action: str = "custom",
    emotional_tension: str = "custom",
    environment: str = "custom",
    env_scale: str = "custom",
    env_condition: str = "custom",
    time_of_day: str = "custom",
    weather: str = "custom",
    lighting_setup: str = "custom",
    light_color: str = "custom",
    shadow_intensity: str = "custom",
    camera_shot: str = "custom",
    camera_angle: str = "custom",
    camera_movement: str = "custom",
    lens_effects: str = "custom",
    post_processing: str = "custom",
    mood: str = "custom",
    art_style: str = "custom",
    composition: str = "custom",
    foreground: str = "custom",
    background: str = "custom",
    custom_instructions: str = "",
) -> str:
    return SYSTEM_PROMPT.format(
        scene_type=scene_type,
        num_characters=num_characters,
        relationship=relationship,
        scene_action=scene_action,
        emotional_tension=emotional_tension,
        environment=environment,
        env_scale=env_scale,
        env_condition=env_condition,
        time_of_day=time_of_day,
        weather=weather,
        lighting_setup=lighting_setup,
        light_color=light_color,
        shadow_intensity=shadow_intensity,
        camera_shot=camera_shot,
        camera_angle=camera_angle,
        camera_movement=camera_movement,
        lens_effects=lens_effects,
        post_processing=post_processing,
        mood=mood,
        art_style=art_style,
        composition=composition,
        foreground=foreground,
        background=background,
        custom_instructions=f"Additional Instructions: {custom_instructions}" if custom_instructions.strip() else "",
    )

def build_user_prompt(scene_description: str) -> str:
    return f"""Direct this scene:

{scene_description}

Create a complete, production-ready scene composition using all the specified parameters. Output valid JSON only."""
