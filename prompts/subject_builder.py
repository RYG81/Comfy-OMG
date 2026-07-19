"""
System prompts for the Subject Builder node.
Build detailed character/subject descriptions.
"""

SPECIES = [
    "custom", "human", "elf", "dwarf", "orc", "goblin", "fairy", "angel", "demon",
    "vampire", "werewolf", "ghost", "zombie", "skeleton", "mermaid", "centaur",
    "dragon", "phoenix", "unicorn", "griffin", "cat", "dog", "wolf", "fox", "lion",
    "tiger", "bear", "horse", "rabbit", "mouse", "bird", "owl", "eagle", "snake",
    "robot", "android", "cyborg", "AI hologram", "alien", "slime", "elemental",
]

GENDER_PRESENTATION = [
    "custom", "feminine", "masculine", "androgynous", "non-binary presenting",
    "ambiguous", "varies", "not applicable",
]

BUILD = [
    "custom", "petite", "slim", "lean", "average", "athletic", "fit", "toned",
    "muscular", "bodybuilder", "stocky", "heavyset", "curvy", "voluptuous",
    "tall and lanky", "short and compact", "imposing", "delicate", "graceful",
]

SKIN = [
    "custom", "fair", "light", "medium", "olive", "tan", "brown", "dark brown",
    "ebony", "porcelain", "freckled", "scarred", "tattooed", "scaled", "furry",
    "metallic", "translucent", "glowing", "stone-textured", "bark-like",
]

DISTINGUISHING_FEATURES = [
    "custom", "none", "scar across face", "scar across eye", "missing eye",
    "heterochromia", "pointed ears", "horns", "fangs", "wings", "tail",
    "unusual eye color", "glowing eyes", "tattoos", "piercings", "birthmark",
    "prosthetic limb", "mechanical parts", "unusual hair", "halo", "aura",
]

PERSONALITY_VIBE = [
    "custom", "confident", "shy", "mysterious", "friendly", "intimidating",
    "wise", "playful", "serious", "melancholic", "energetic", "calm",
    "mischievous", "noble", "villainous", "heroic", "innocent", "jaded",
    "romantic", "scholarly", "wild", "refined", "battle-hardened", "gentle",
]

SYSTEM_PROMPT = """You are an expert character designer. You create detailed, vivid character descriptions optimized for AI image generation.

Build a character based on these specifications:
- Species/Type: {species}
- Gender Presentation: {gender}
- Age Appearance: {age}
- Build/Body Type: {build}
- Skin/Surface: {skin}
- Hair Style: {hair_style}
- Hair Color: {hair_color}
- Eye Description: {eyes}
- Facial Features: {facial}
- Distinguishing Features: {distinguishing}
- Personality Vibe: {personality}

{custom_details}

Create a complete character description that an AI image generator can visualize consistently.

Respond in this EXACT JSON format:
{{
    "full_description": "Complete character description in natural language, ready to use as a prompt",
    "appearance_tags": "Comma-separated appearance tags (booru-style)",
    "physical_summary": "Brief physical description summary",
    "face_description": "Detailed face and expression description",
    "body_description": "Body type and posture description",
    "clothing_suggestion": "Suggested default clothing that fits the character",
    "personality_expression": "How their personality shows in their appearance/expression",
    "consistency_anchors": "Key visual elements that should ALWAYS be included to maintain consistency",
    "negative_prompt": "What to avoid when generating this character"
}}

IMPORTANT: Create a memorable, distinctive character that will be easy to generate consistently. Output ONLY valid JSON."""

def build_system_prompt(
    species: str, gender: str, age: str, build: str, skin: str,
    hair_style: str, hair_color: str, eyes: str, facial: str,
    distinguishing: str, personality: str, custom_details: str = "",
) -> str:
    return SYSTEM_PROMPT.format(
        species=species, gender=gender, age=age, build=build, skin=skin,
        hair_style=hair_style, hair_color=hair_color, eyes=eyes, facial=facial,
        distinguishing=distinguishing, personality=personality,
        custom_details=f"Additional Details: {custom_details}" if custom_details.strip() else "",
    )

def build_user_prompt(character_name: str = "") -> str:
    name_str = f"Character Name: {character_name}\n\n" if character_name.strip() else ""
    return f"""{name_str}Create a detailed, consistent character description. Output valid JSON only."""
