"""
System prompts for the Creature Creator node.
Design fantasy, sci-fi, or hybrid creatures.
"""

BASE_CREATURE = [
    "custom", "original design", "dragon", "phoenix", "griffin", "unicorn",
    "pegasus", "chimera", "hydra", "basilisk", "kraken", "leviathan",
    "werewolf", "vampire", "demon", "angel", "fairy", "elemental",
    "golem", "treant", "slime", "mimic", "beholder", "mind flayer",
    "xenomorph-like", "kaiju", "mech-creature", "bio-mechanical",
]

SIZE_SCALE = [
    "tiny - insect sized", "small - cat sized", "medium - dog sized",
    "large - horse sized", "huge - elephant sized", "massive - building sized",
    "colossal - kaiju sized", "cosmic - planet sized",
]

BODY_PLAN = [
    "custom", "bipedal", "quadruped", "hexapod", "octopod", "serpentine",
    "avian", "aquatic", "amorphous", "centauroid", "arachnid", "insectoid",
    "tentacled", "floating", "multiple bodies", "shapeshifter",
]

SURFACE_COVERING = [
    "custom", "scales", "fur", "feathers", "skin", "chitin", "bark",
    "crystal", "metal", "stone", "slime", "fire", "shadow", "light",
    "smoke", "water", "ice", "electricity", "void",
]

SPECIAL_FEATURES = [
    "custom", "wings", "multiple heads", "extra limbs", "tail",
    "horns", "antlers", "tusks", "fangs", "claws", "spines",
    "bioluminescence", "camouflage", "elemental aura", "third eye",
    "tentacles", "mandibles", "poison glands", "breath weapon",
]

TEMPERAMENT = [
    "custom", "aggressive", "passive", "territorial", "curious",
    "protective", "playful", "mysterious", "malevolent", "benevolent",
    "ancient and wise", "primal and instinctive", "intelligent and calculating",
]

SYSTEM_PROMPT = """You are an expert creature designer for fantasy, sci-fi, and imaginative media.

Design a creature with these specifications:
- Base/Inspiration: {base}
- Size: {size}
- Body Plan: {body_plan}
- Surface Covering: {surface}
- Special Features: {features}
- Temperament: {temperament}
- Habitat: {habitat}
- Color Scheme: {colors}
- Style: {art_style}

{custom_details}

Respond in this EXACT JSON format:
{{
    "creature_prompt": "Complete creature description ready for image generation",
    "anatomy_description": "Detailed body structure and proportions",
    "surface_details": "Skin/scales/fur texture and patterns",
    "feature_details": "Special features and how they look",
    "coloration": "Color scheme and markings",
    "eye_description": "Eye appearance (count, color, glow, etc.)",
    "movement_impression": "How the creature appears to move/behave",
    "environmental_context": "Natural habitat hints in the design",
    "size_reference": "How to convey scale in the image",
    "style_tags": "Tags for generating this creature",
    "negative_prompt": "What to avoid when generating"
}}

Create a unique, visually interesting creature. Output ONLY valid JSON."""

def build_system_prompt(**kwargs) -> str:
    return SYSTEM_PROMPT.format(
        base=kwargs.get("base", "original design"),
        size=kwargs.get("size", "large - horse sized"),
        body_plan=kwargs.get("body_plan", "custom"),
        surface=kwargs.get("surface", "custom"),
        features=kwargs.get("features", "custom"),
        temperament=kwargs.get("temperament", "custom"),
        habitat=kwargs.get("habitat", "custom"),
        colors=kwargs.get("colors", "custom"),
        art_style=kwargs.get("art_style", "custom"),
        custom_details=f"Additional Details: {kwargs.get('custom_details', '')}" if kwargs.get('custom_details') else "",
    )

def build_user_prompt(creature_concept: str = "") -> str:
    if creature_concept.strip():
        return f"Create this creature:\n\n{creature_concept}\n\nOutput valid JSON only."
    return "Design a unique creature based on the specified parameters. Output valid JSON only."
