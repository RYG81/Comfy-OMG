"""
System prompts for the Texture & Material Designer node.
Create detailed material and texture descriptions.
"""

MATERIAL_CATEGORY = [
    "custom", "metal", "wood", "stone", "fabric", "leather", "glass",
    "plastic", "ceramic", "organic", "liquid", "crystal", "ice",
    "fire", "smoke", "energy", "magical", "alien", "bio-mechanical",
]

METAL_TYPE = [
    "custom", "gold", "silver", "bronze", "copper", "iron", "steel",
    "chrome", "titanium", "brass", "platinum", "rust", "patina",
    "brushed", "polished", "hammered", "damascus", "mithril", "adamantine",
]

FABRIC_TYPE = [
    "custom", "silk", "satin", "velvet", "cotton", "linen", "wool",
    "denim", "leather", "suede", "lace", "chiffon", "tulle", "brocade",
    "tweed", "corduroy", "mesh", "latex", "vinyl", "fur",
]

STONE_TYPE = [
    "custom", "marble", "granite", "slate", "sandstone", "limestone",
    "obsidian", "jade", "onyx", "quartz", "crystal", "geode",
    "concrete", "brick", "cobblestone", "ancient weathered", "mossy",
]

WOOD_TYPE = [
    "custom", "oak", "mahogany", "pine", "cedar", "birch", "walnut",
    "cherry", "bamboo", "driftwood", "petrified", "burnt", "weathered",
    "polished", "carved", "inlaid", "bark", "roots",
]

SURFACE_FINISH = [
    "custom", "matte", "satin", "glossy", "mirror polish", "brushed",
    "textured", "rough", "smooth", "bumpy", "pitted", "scratched",
    "worn", "pristine", "weathered", "aged", "patinated",
]

CONDITION = [
    "custom", "pristine - brand new", "good - light use",
    "worn - visible wear", "aged - significant weathering",
    "damaged - cracks/chips", "ruined - heavily deteriorated",
    "ancient - archaeological", "magical - enhanced/glowing",
]

SYSTEM_PROMPT = """You are an expert texture and material artist. You create detailed, photorealistic material descriptions for AI image generation.

Design a material/texture with these specifications:
- Category: {category}
- Specific Type: {specific_type}
- Surface Finish: {finish}
- Condition: {condition}
- Color: {color}
- Pattern: {pattern}
- Lighting Response: {lighting}
- Special Properties: {special}

{custom_details}

Respond in this EXACT JSON format:
{{
    "material_prompt": "Complete material description for image generation",
    "surface_texture": "Detailed surface texture description",
    "color_description": "Exact colors and color variations",
    "reflectivity": "How light reflects off the surface",
    "transparency": "Any transparency or translucency",
    "pattern_details": "Patterns, grain, or texture patterns",
    "wear_details": "Signs of age, use, or weathering",
    "tactile_impression": "How it would feel (conveyed visually)",
    "close_up_details": "What you'd see in extreme close-up",
    "material_tags": "Tags for material rendering",
    "lighting_tips": "Best lighting to showcase this material",
    "negative_prompt": "What to avoid"
}}

Create a convincing, detailed material description. Output ONLY valid JSON."""

def build_system_prompt(**kwargs) -> str:
    return SYSTEM_PROMPT.format(
        category=kwargs.get("category", "custom"),
        specific_type=kwargs.get("specific_type", "custom"),
        finish=kwargs.get("finish", "custom"),
        condition=kwargs.get("condition", "good - light use"),
        color=kwargs.get("color", "custom"),
        pattern=kwargs.get("pattern", "custom"),
        lighting=kwargs.get("lighting", "custom"),
        special=kwargs.get("special", "none"),
        custom_details=f"Additional Details: {kwargs.get('custom_details', '')}" if kwargs.get('custom_details') else "",
    )

def build_user_prompt(material_description: str = "") -> str:
    if material_description.strip():
        return f"Create this material:\n\n{material_description}\n\nOutput valid JSON only."
    return "Design a detailed material/texture based on the specified parameters. Output valid JSON only."
