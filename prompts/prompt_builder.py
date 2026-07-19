"""
System prompts for the Prompt Builder node.
Comprehensive element-based prompt construction.
"""

# ═══════════════════════════════════════════════════════════════════
# DROPDOWN OPTIONS FOR ALL CATEGORIES
# ═══════════════════════════════════════════════════════════════════

SUBJECT_TYPE = [
    "custom", "woman", "man", "girl", "boy", "elderly woman", "elderly man",
    "androgynous person", "couple", "group of people", "family",
    "cat", "dog", "wolf", "fox", "lion", "tiger", "bear", "horse", "dragon",
    "bird", "eagle", "owl", "phoenix", "butterfly", "fish", "whale", "dolphin",
    "robot", "android", "cyborg", "mech", "AI entity",
    "elf", "dwarf", "orc", "fairy", "angel", "demon", "vampire", "werewolf",
    "witch", "wizard", "knight", "samurai", "ninja", "pirate",
    "car", "motorcycle", "spaceship", "airplane", "train", "ship",
    "building", "castle", "temple", "house", "skyscraper", "ruins",
    "landscape", "cityscape", "seascape", "mountain", "forest", "desert",
    "abstract shape", "geometric form", "organic form",
    "food", "flower", "tree", "crystal", "gem", "artifact",
]

AGE_APPEARANCE = [
    "custom", "baby", "toddler", "child", "preteen", "teenager",
    "young adult (20s)", "adult (30s)", "middle-aged (40s-50s)",
    "mature (60s)", "elderly (70s+)", "ageless", "ancient",
]

BODY_TYPE = [
    "custom", "petite", "slim", "average", "athletic", "muscular", "bodybuilder",
    "curvy", "plus-size", "tall and lanky", "short and stocky",
    "graceful", "imposing", "delicate", "robust",
]

ETHNICITY_APPEARANCE = [
    "custom", "not specified", "East Asian", "South Asian", "Southeast Asian",
    "Middle Eastern", "African", "European", "Latin American", "Native American",
    "Pacific Islander", "Mixed heritage", "Fantasy race", "Alien",
]

HAIR_STYLE = [
    "custom", "bald", "buzzcut", "short cropped", "pixie cut", "bob",
    "shoulder length", "long straight", "long wavy", "long curly",
    "braided", "twin braids", "ponytail", "high ponytail", "twin tails",
    "bun", "messy bun", "updo", "dreadlocks", "afro", "mohawk",
    "undercut", "slicked back", "windswept", "wet hair", "bedhead",
]

HAIR_COLOR = [
    "custom", "black", "dark brown", "brown", "light brown", "auburn",
    "red", "ginger", "strawberry blonde", "blonde", "platinum blonde",
    "white", "silver", "gray", "blue", "light blue", "purple", "pink",
    "green", "teal", "orange", "rainbow", "ombre", "highlights",
]

FACIAL_FEATURES = [
    "custom", "sharp features", "soft features", "angular jaw", "round face",
    "high cheekbones", "freckles", "beauty mark", "scars", "tattoos",
    "piercing eyes", "gentle eyes", "heterochromia", "glasses", "sunglasses",
    "beard", "stubble", "mustache", "clean shaven", "makeup", "no makeup",
]

EXPRESSION = [
    "custom", "neutral", "slight smile", "big smile", "laughing", "smirk",
    "serious", "stern", "angry", "furious", "sad", "crying", "melancholic",
    "surprised", "shocked", "scared", "terrified", "disgusted",
    "confused", "thoughtful", "dreamy", "sleepy", "exhausted",
    "confident", "shy", "embarrassed", "blushing", "seductive", "playful",
    "determined", "hopeful", "proud", "guilty", "jealous", "loving",
]

POSE = [
    "custom", "standing straight", "standing relaxed", "standing contrapposto",
    "walking", "running", "jumping", "flying", "floating", "falling",
    "sitting", "sitting cross-legged", "sitting on chair", "sitting casually",
    "kneeling", "crouching", "lying down", "lying on side", "lying face down",
    "leaning", "leaning against wall", "arms crossed", "hands on hips",
    "hands in pockets", "arms raised", "reaching out", "pointing",
    "fighting stance", "action pose", "dynamic pose", "dramatic pose",
    "dancing", "stretching", "yoga pose", "meditation pose",
    "hugging", "holding hands", "back to back", "face to face",
    "looking over shoulder", "looking up", "looking down", "profile view",
]

CLOTHING_STYLE = [
    "custom", "nude", "casual", "formal", "business", "streetwear",
    "athletic", "swimwear", "lingerie", "sleepwear", "costume",
    "historical", "victorian", "medieval", "ancient", "futuristic",
    "military", "uniform", "school uniform", "maid outfit", "nurse outfit",
    "fantasy armor", "sci-fi suit", "superhero suit", "traditional",
    "kimono", "hanbok", "sari", "dress", "gown", "wedding dress",
    "punk", "goth", "bohemian", "preppy", "hipster", "minimalist",
]

CLOTHING_ITEMS = [
    "custom", "t-shirt", "blouse", "shirt", "sweater", "hoodie", "jacket",
    "coat", "blazer", "vest", "tank top", "crop top", "cardigan",
    "jeans", "pants", "shorts", "skirt", "mini skirt", "long skirt",
    "dress", "maxi dress", "cocktail dress", "sundress",
    "suit", "tuxedo", "uniform", "armor", "robes", "cloak", "cape",
]

ACCESSORIES = [
    "custom", "none", "necklace", "choker", "pendant", "earrings", "rings",
    "bracelet", "watch", "glasses", "sunglasses", "hat", "cap", "beanie",
    "headband", "hair clips", "tiara", "crown", "veil", "mask",
    "scarf", "tie", "bowtie", "belt", "gloves", "bag", "backpack",
    "umbrella", "cane", "sword", "shield", "wand", "staff", "wings",
]

ENVIRONMENT = [
    "custom", "studio", "white background", "gradient background", "simple background",
    "bedroom", "living room", "kitchen", "bathroom", "office", "classroom",
    "restaurant", "cafe", "bar", "club", "library", "museum", "gallery",
    "street", "alley", "rooftop", "balcony", "garden", "park", "plaza",
    "beach", "ocean", "underwater", "lake", "river", "waterfall", "pool",
    "forest", "jungle", "meadow", "field", "mountain", "cliff", "cave",
    "desert", "canyon", "volcano", "glacier", "arctic", "swamp",
    "city", "downtown", "suburbs", "village", "ruins", "abandoned building",
    "castle", "palace", "temple", "church", "shrine", "dungeon",
    "spaceship interior", "space station", "alien planet", "moon surface",
    "cyberpunk city", "fantasy realm", "dreamscape", "void", "heaven", "hell",
]

TIME_OF_DAY = [
    "custom", "dawn", "early morning", "morning", "late morning",
    "noon", "early afternoon", "afternoon", "late afternoon",
    "golden hour", "sunset", "dusk", "twilight", "blue hour",
    "evening", "night", "midnight", "late night", "timeless",
]

WEATHER = [
    "custom", "clear", "sunny", "partly cloudy", "overcast", "cloudy",
    "light rain", "rain", "heavy rain", "thunderstorm", "drizzle",
    "snow", "heavy snow", "blizzard", "sleet", "hail",
    "fog", "mist", "haze", "smog", "sandstorm", "dust storm",
    "windy", "stormy", "hurricane", "tornado",
    "rainbow", "aurora", "meteor shower", "eclipse",
]

SEASON = [
    "custom", "spring", "early summer", "summer", "late summer",
    "early autumn", "autumn", "late autumn", "early winter", "winter", "late winter",
]

LIGHTING_TYPE = [
    "custom", "natural light", "sunlight", "daylight", "moonlight", "starlight",
    "golden hour light", "blue hour light", "overcast light", "diffused light",
    "artificial light", "studio lighting", "ring light", "softbox",
    "neon lights", "fluorescent", "incandescent", "candlelight", "firelight",
    "spotlight", "backlight", "rim light", "fill light", "ambient light",
    "dramatic lighting", "cinematic lighting", "noir lighting", "chiaroscuro",
    "bioluminescence", "magical glow", "holographic", "volumetric light",
]

LIGHTING_DIRECTION = [
    "custom", "front lighting", "side lighting", "back lighting", "top lighting",
    "bottom lighting (uplighting)", "45-degree lighting", "Rembrandt lighting",
    "split lighting", "loop lighting", "butterfly lighting", "broad lighting",
    "short lighting", "rim lighting", "silhouette", "contre-jour",
]

LIGHTING_MOOD = [
    "custom", "bright and airy", "soft and dreamy", "warm and cozy", "cool and crisp",
    "dramatic and moody", "dark and mysterious", "ethereal and glowing",
    "harsh and stark", "romantic and intimate", "energetic and vibrant",
    "melancholic and somber", "peaceful and serene", "tense and suspenseful",
]

CAMERA_SHOT = [
    "custom", "extreme close-up (ECU)", "close-up (CU)", "medium close-up (MCU)",
    "medium shot (MS)", "medium full shot (MFS)", "full shot (FS)",
    "wide shot (WS)", "extreme wide shot (EWS)", "establishing shot",
    "portrait", "headshot", "bust shot", "waist-up", "knee shot", "full body",
    "two-shot", "group shot", "over-the-shoulder (OTS)", "point-of-view (POV)",
]

CAMERA_ANGLE = [
    "custom", "eye level", "low angle", "high angle", "bird's eye view",
    "worm's eye view", "dutch angle (tilted)", "overhead shot", "top-down",
    "front view", "side view (profile)", "three-quarter view", "back view",
    "dynamic angle", "dramatic angle", "canted frame",
]

CAMERA_LENS = [
    "custom", "wide-angle lens", "ultra-wide lens", "fisheye lens",
    "standard lens (50mm)", "portrait lens (85mm)", "telephoto lens",
    "macro lens", "tilt-shift lens", "anamorphic lens",
]

DEPTH_OF_FIELD = [
    "custom", "deep focus (everything sharp)", "shallow depth of field",
    "bokeh background", "soft bokeh", "harsh bokeh", "swirly bokeh",
    "tilt-shift effect", "selective focus", "rack focus",
]

ART_STYLE = [
    "custom", "photorealistic", "hyperrealistic", "realistic", "semi-realistic",
    "anime", "manga", "cartoon", "comic book", "graphic novel",
    "disney", "pixar", "dreamworks", "studio ghibli", "makoto shinkai",
    "oil painting", "watercolor", "acrylic", "gouache", "pastel",
    "pencil sketch", "charcoal", "ink drawing", "pen and ink",
    "digital painting", "digital art", "concept art", "matte painting",
    "impressionism", "expressionism", "surrealism", "art nouveau", "art deco",
    "pop art", "minimalist", "abstract", "cubism", "baroque", "renaissance",
    "cyberpunk", "steampunk", "dieselpunk", "solarpunk", "fantasy art",
    "dark fantasy", "sci-fi art", "retro", "vintage", "vaporwave", "synthwave",
    "pixel art", "low poly", "3D render", "unreal engine", "octane render",
]

ART_MEDIUM = [
    "custom", "photograph", "digital art", "oil on canvas", "watercolor on paper",
    "acrylic painting", "pencil drawing", "charcoal drawing", "ink illustration",
    "3D render", "CGI", "vector art", "mixed media", "collage",
    "spray paint", "airbrush", "marker", "crayon", "chalk",
    "sculpture", "clay model", "bronze statue", "marble sculpture",
    "stained glass", "mosaic", "tapestry", "embroidery",
]

COLOR_PALETTE = [
    "custom", "vibrant colors", "muted colors", "pastel colors", "neon colors",
    "warm colors", "cool colors", "neutral colors", "earth tones",
    "monochromatic", "black and white", "sepia", "duotone", "triadic colors",
    "complementary colors", "analogous colors", "split-complementary",
    "high contrast", "low contrast", "desaturated", "highly saturated",
    "dark palette", "light palette", "golden palette", "blue palette",
    "red palette", "green palette", "purple palette", "rainbow",
]

MOOD_ATMOSPHERE = [
    "custom", "happy", "joyful", "peaceful", "serene", "calm", "relaxing",
    "romantic", "dreamy", "nostalgic", "melancholic", "sad", "somber",
    "mysterious", "suspenseful", "tense", "dramatic", "intense",
    "dark", "gloomy", "ominous", "creepy", "horrifying", "nightmarish",
    "energetic", "dynamic", "action-packed", "chaotic", "explosive",
    "magical", "ethereal", "otherworldly", "surreal", "psychedelic",
    "epic", "grand", "majestic", "heroic", "triumphant",
    "intimate", "cozy", "warm", "inviting", "whimsical", "playful",
]

QUALITY_TAGS = [
    "custom", "masterpiece", "best quality", "high quality", "ultra detailed",
    "highly detailed", "intricate details", "sharp focus", "8k", "4k",
    "professional", "award-winning", "trending on artstation", "featured",
    "studio quality", "cinematic", "photographic", "raw photo",
]

SPECIAL_EFFECTS = [
    "custom", "none", "particles", "sparkles", "glitter", "dust motes",
    "lens flare", "light rays", "god rays", "volumetric lighting",
    "motion blur", "radial blur", "chromatic aberration", "film grain",
    "smoke", "fog", "mist", "steam", "fire", "flames", "sparks",
    "water droplets", "rain drops", "snow falling", "leaves falling",
    "petals falling", "confetti", "bubbles", "lightning", "electricity",
    "magic effects", "glowing", "aura", "energy", "portal", "hologram",
]

COMPOSITION = [
    "custom", "centered", "rule of thirds", "golden ratio", "symmetrical",
    "asymmetrical", "diagonal", "leading lines", "frame within frame",
    "negative space", "fill the frame", "layered", "depth", "foreground interest",
    "background blur", "minimalist", "busy", "balanced", "dynamic",
]

SYSTEM_PROMPT = """You are an expert prompt engineer. You take structured elements and combine them into a cohesive, natural-sounding image generation prompt.

You have been given specific values for various image elements. Your job is to:
1. Combine all provided elements into a single, well-structured prompt
2. Make it flow naturally — not just a list of keywords
3. Add necessary connecting words and descriptions
4. Order elements by importance (subject first, then details, then environment, then technical)
5. Include all specified quality and style tags appropriately
6. Handle "custom" values by incorporating the custom text provided

Respond in this EXACT JSON format:
{{
    "constructed_prompt": "The complete, ready-to-use prompt combining all elements naturally",
    "prompt_tags": "The same prompt but in comma-separated tag format for models that prefer tags",
    "negative_prompt": "Appropriate negative prompt based on the elements chosen",
    "summary": "Brief summary of what the image should depict",
    "element_count": "Number of elements incorporated"
}}

IMPORTANT RULES:
- Make the prompt feel natural and descriptive, not like a list
- Put the most important elements (subject, action) first
- Group related elements together
- Ensure style consistency throughout
- Output ONLY valid JSON"""

def build_user_prompt(elements: dict) -> str:
    """Build the user prompt from all provided elements."""
    prompt_parts = []
    
    for category, value in elements.items():
        if value and value != "custom" and value != "none" and value != "not specified":
            prompt_parts.append(f"• {category}: {value}")
    
    elements_text = "\n".join(prompt_parts)
    
    return f"""Combine these image elements into a cohesive prompt:

{elements_text}

Create a natural, well-structured prompt. Output valid JSON only."""
