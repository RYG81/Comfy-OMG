"""
System prompts and dropdown options for the Scene Director node.
"""

SUBJECT_TYPES = [
    "custom", "single character", "two characters", "small group", "large crowd",
    "hero character", "villain character", "couple", "family", "team",
    "animal subject", "creature", "robot/android", "vehicle", "object focus",
    "landscape/no characters",
]

SUBJECT_FOCUS = [
    "custom", "clear main subject", "ensemble balance", "hero dominates frame",
    "subject partially hidden", "silhouette focus", "face and expression focus",
    "body language focus", "hands/action focus", "environment dwarfs subject",
]

POSE_DIRECTION = [
    "custom", "standing relaxed", "standing heroic", "walking", "running",
    "kneeling", "crouching", "sitting", "leaning", "reaching out",
    "looking over shoulder", "face to face", "back to back", "embracing",
    "fighting stance", "mid-action leap", "falling", "floating",
    "hands on hips", "arms crossed", "pointing", "holding object",
]

BLOCKING = [
    "custom", "center stage", "left third", "right third", "foreground center",
    "midground center", "background center", "triangular blocking",
    "diagonal blocking", "symmetrical blocking", "asymmetrical blocking",
    "characters separated by distance", "characters overlapping carefully",
    "one subject foreground, one background", "crowd frames main subject",
]

MOODS = [
    "custom", "dramatic", "peaceful", "action", "romantic", "mysterious",
    "comedic", "epic", "melancholic", "tense", "hopeful", "ominous",
    "triumphant", "intimate", "surreal", "dreamlike", "noir", "chaotic",
]

SETTINGS = [
    "custom", "studio", "bedroom", "living room", "office", "restaurant",
    "street", "alley", "rooftop", "garden", "forest", "beach", "desert",
    "mountain", "cave", "ruins", "castle", "temple", "spaceship interior",
    "cyberpunk city", "fantasy realm", "battlefield", "dreamscape",
]

TIME_OF_DAY = [
    "custom", "dawn", "morning", "noon", "afternoon", "golden hour",
    "sunset", "dusk", "blue hour", "night", "midnight", "timeless",
]

WEATHER = [
    "custom", "clear", "sunny", "overcast", "light rain", "heavy rain",
    "thunderstorm", "snow", "blizzard", "fog", "mist", "windy",
    "dust storm", "aurora", "meteor shower",
]

LIGHTING_SETUPS = [
    "custom", "natural ambient", "three-point lighting", "single key light",
    "softbox studio", "high-key bright", "low-key dramatic", "chiaroscuro",
    "Rembrandt lighting", "split lighting", "silhouette/backlit",
    "rim lighting", "practical lights in scene", "candlelight/firelight",
    "neon lighting", "moonlight", "cinematic lighting", "volumetric god rays",
    "magical glow",
]

LIGHT_COLOR = [
    "custom", "neutral white", "warm gold", "cool blue", "moonlight blue",
    "fire orange", "red", "cyan", "green", "purple", "pink",
    "mixed warm and cool", "neon multicolor", "sunset gradient",
]

SHADOW_STYLE = [
    "custom", "flat minimal shadows", "soft diffused shadows",
    "natural medium shadows", "hard graphic shadows", "deep noir shadows",
    "long sunset shadows", "colored shadows", "multiple shadow directions",
]

CAMERA_SHOTS = [
    "custom", "extreme close-up", "close-up", "medium close-up",
    "medium shot", "full shot", "wide shot", "extreme wide shot",
    "establishing shot", "portrait", "two-shot", "group shot",
    "over-the-shoulder", "point-of-view",
]

CAMERA_ANGLES = [
    "custom", "eye level", "low angle", "high angle", "bird's eye view",
    "worm's eye view", "dutch angle", "overhead", "front view",
    "side profile", "three-quarter view", "back view", "dynamic angle",
]

LENS_CHOICES = [
    "custom", "wide-angle lens", "ultra-wide lens", "standard 50mm lens",
    "portrait 85mm lens", "telephoto lens", "macro lens", "anamorphic lens",
    "fisheye lens", "tilt-shift lens",
]

DEPTH_OF_FIELD = [
    "custom", "deep focus", "shallow depth of field", "soft bokeh",
    "strong bokeh", "selective focus", "rack-focus inspired",
    "foreground blur", "background blur",
]

COMPOSITION_STYLES = [
    "custom", "rule of thirds", "centered", "symmetrical", "asymmetrical",
    "diagonal tension", "leading lines", "frame within frame",
    "negative space", "layered depth", "triangular composition",
    "golden ratio", "crowded chaotic frame", "minimal clean frame",
]

ART_STYLES = [
    "custom", "photorealistic", "cinematic photo", "anime", "manga",
    "digital painting", "concept art", "oil painting", "watercolor",
    "comic book", "3D render", "dark fantasy", "sci-fi", "cyberpunk",
    "studio ghibli inspired", "noir", "surrealism",
]

COLOR_GRADES = [
    "custom", "natural color", "warm grade", "cool grade", "teal and orange",
    "desaturated", "high contrast", "low contrast", "black and white",
    "sepia", "vintage film", "bleach bypass", "neon saturated",
]

DIRECTOR_INTENT = [
    "custom", "emphasize emotional intimacy", "emphasize power imbalance",
    "emphasize mystery", "emphasize speed and danger", "emphasize isolation",
    "emphasize wonder", "emphasize scale", "emphasize comedy",
    "emphasize tragedy", "emphasize heroic resolve", "emphasize unease",
]

OUTPUT_PROFILES = [
    "balanced image prompt",
    "FLUX natural language",
    "SDXL tag-dense",
    "Qwen image detailed natural",
    "video keyframe friendly",
]

ASPECT_RATIOS = [
    "custom", "1:1 square", "16:9 landscape", "9:16 portrait",
    "4:3 classic", "3:4 portrait", "21:9 cinematic wide",
    "2:3 poster", "3:2 photo",
]

PROMPT_LENGTH = [
    "compact", "standard", "rich",
]


SYSTEM_PROMPT = """You are a production-level film director, cinematographer, blocking artist, and AI image prompt engineer.

Your job is to turn the user's scene idea into a director-controlled image prompt. Follow the selected controls exactly when they are provided, and resolve conflicts with coherent cinematic judgment.

Director controls:
- Output profile: {output_profile}
- Aspect ratio / framing target: {aspect_ratio}
- Prompt length: {prompt_length}
- Number of characters: {num_characters}
- Subject type: {subject_type}
- Subject focus: {subject_focus}
- Pose / performance: {pose_direction}
- Blocking: {blocking}
- Setting: {setting}
- Time of day: {time_of_day}
- Weather: {weather}
- Mood: {mood}
- Director intent: {director_intent}
- Lighting setup: {lighting_setup}
- Light color: {light_color}
- Shadow style: {shadow_style}
- Camera shot: {camera_shot}
- Camera angle: {camera_angle}
- Lens: {lens}
- Depth of field: {depth_of_field}
- Composition: {composition}
- Art style: {art_style}
- Color grade: {color_grade}
- Additional details: {additional_details}

Custom system guidance from the user:
{custom_system_prompt}

Respond in this EXACT JSON format:
{{
    "scene_prompt": "A complete, model-ready prompt in the selected output profile. Include subject identity, pose, blocking, setting, action, mood, lighting, camera, lens, composition, color, aspect framing, and style.",
    "composition_guide": "Specific frame layout using foreground/midground/background, left/center/right, scale, facing directions, and focal hierarchy.",
    "lighting_setup": "Production lighting notes: key/fill/rim/ambient/practical lights, direction, color, shadow behavior, atmosphere.",
    "character_descriptions": "Distinct character-by-character descriptions covering appearance, wardrobe, pose, expression, action, and position.",
    "negative_prompt": "Things to avoid for this scene, including composition errors, character merging, bad anatomy, inconsistent lighting, unwanted style drift.",
    "director_notes": "Narrative intent, emotional subtext, performance direction, and why the selected camera/lighting choices support the scene."
}}

Rules:
- Treat the custom system guidance as high-priority creative direction unless it asks for invalid JSON.
- For multi-character scenes, prevent character merging with explicit spatial separation and unique descriptors.
- If a dropdown says custom but no custom text is provided, infer a sensible choice from the scene.
- Keep the final scene_prompt directly usable in an image-generation prompt box.
- Match the output profile:
  - FLUX natural language: use clear complete sentences, not tag soup.
  - SDXL tag-dense: use concise comma-separated visual tags and useful quality/style descriptors.
  - Qwen image detailed natural: use a rich natural visual brief with explicit text/material/layout constraints when relevant.
  - video keyframe friendly: write a still-frame prompt with continuity locks for identity, wardrobe, lighting, and environment.
- Match prompt length: compact = one tight paragraph, standard = detailed but controlled, rich = dense production detail without contradiction.
- Avoid contradictory camera language, impossible lighting, and overloaded actions.
- Output ONLY valid JSON."""


def build_system_prompt(**kwargs) -> str:
    def clean(value: object, fallback: str = "infer from the scene") -> str:
        text = str(value or "").strip()
        if not text or text == "custom":
            return fallback
        return text

    defaults = {
        "output_profile": "balanced image prompt",
        "aspect_ratio": "infer from the scene",
        "prompt_length": "standard",
        "num_characters": 1,
        "subject_type": "infer from the scene",
        "subject_focus": "infer from the scene",
        "pose_direction": "infer from the scene",
        "blocking": "infer from the scene",
        "setting": "infer from the scene",
        "time_of_day": "infer from the scene",
        "weather": "infer from the scene",
        "mood": "infer from the scene",
        "director_intent": "infer from the scene",
        "lighting_setup": "infer from the scene",
        "light_color": "infer from the scene",
        "shadow_style": "infer from the scene",
        "camera_shot": "infer from the scene",
        "camera_angle": "infer from the scene",
        "lens": "infer from the scene",
        "depth_of_field": "infer from the scene",
        "composition": "infer from the scene",
        "art_style": "infer from the scene",
        "color_grade": "infer from the scene",
        "additional_details": "none",
        "custom_system_prompt": "No additional custom system guidance.",
    }
    values = {key: clean(value) for key, value in kwargs.items()}
    for key, value in defaults.items():
        values.setdefault(key, value)
    values["custom_system_prompt"] = (
        str(kwargs.get("custom_system_prompt", "")).strip()
        or "No additional custom system guidance."
    )
    values["num_characters"] = kwargs.get("num_characters", 1)
    return SYSTEM_PROMPT.format(**values)


def build_director_brief(**kwargs) -> str:
    labels = {
        "num_characters": "Number of characters",
        "subject_type": "Subject type",
        "subject_focus": "Subject focus",
        "pose_direction": "Pose / performance",
        "blocking": "Blocking",
        "setting": "Setting",
        "time_of_day": "Time of day",
        "weather": "Weather",
        "mood": "Mood",
        "director_intent": "Director intent",
        "lighting_setup": "Lighting setup",
        "light_color": "Light color",
        "shadow_style": "Shadow style",
        "camera_shot": "Camera shot",
        "camera_angle": "Camera angle",
        "lens": "Lens",
        "depth_of_field": "Depth of field",
        "composition": "Composition",
        "art_style": "Art style",
        "color_grade": "Color grade",
        "additional_details": "Additional details",
        "custom_system_prompt": "Custom system guidance",
        "output_profile": "Output profile",
        "aspect_ratio": "Aspect ratio / framing target",
        "prompt_length": "Prompt length",
    }
    lines = []
    for key, label in labels.items():
        value = str(kwargs.get(key, "") or "").strip()
        if value and value != "custom":
            lines.append(f"- {label}: {value}")
    return "\n".join(lines) if lines else "- No explicit director controls provided"


def build_user_prompt(scene_description: str, director_brief: str = "") -> str:
    return f"""Direct this scene with full cinematic control:

{scene_description}

Selected director controls that MUST affect the output:
{director_brief}

Create a production-ready scene composition. Output valid JSON only."""
