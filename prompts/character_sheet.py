"""
System prompts for the Character Sheet Creator node.
"""

STYLES = [
    "Anime/Manga",
    "Semi-Realistic",
    "Realistic Illustration",
    "Real Photo / Cosplay Reference",
    "Studio Portrait Photography",
    "Fashion Editorial Photo",
    "Cinematic Character Photo",
    "Disney/Pixar",
    "Comic Book (Western)",
    "Graphic Novel",
    "Chibi/Super-Deformed",
    "Game Character Concept Art",
    "Animation Model Sheet",
    "3D Character Render",
    "Clay / Maquette Render",
    "Watercolor Illustration",
    "Oil Painting",
    "Pixel Art",
    "Art Nouveau",
    "Dark Fantasy",
    "Cyberpunk",
    "Studio Ghibli",
]

SHEET_FORMATS = [
    "production turnaround sheet",
    "animation model sheet",
    "game character reference sheet",
    "photoreal casting reference sheet",
    "cosplay construction reference sheet",
    "fashion character lineup sheet",
    "face and expression reference sheet",
    "full master sheet",
]

POSE_STANDARDS = [
    "relaxed neutral stance",
    "strict A-pose",
    "strict T-pose",
    "natural contrapposto",
    "hands visible at sides",
    "front-facing fashion stance",
    "hero stance",
    "character acting pose",
]

BACKGROUND_STYLES = [
    "clean white background",
    "light gray studio background",
    "transparent-background style",
    "turnaround grid on white",
    "blueprint/model-sheet layout",
    "photo studio cyclorama",
    "minimal contact shadows",
]

CONSISTENCY_LEVELS = [
    "balanced",
    "strict - exact outfit and proportions",
    "very strict - production model sheet",
    "photo continuity - same actor, wardrobe, lighting",
]

MODEL_TARGETS = [
    "general image models",
    "SDXL",
    "Flux",
    "Pony/Anime",
    "Midjourney",
    "realistic photo models",
    "3D/render models",
]

EXPRESSION_SETS = [
    "core emotions",
    "animation acting set",
    "subtle realistic expressions",
    "anime expression set",
    "dramatic cinematic expressions",
]

SYSTEM_PROMPT = """You are a senior character designer, animation model-sheet artist, costume continuity supervisor, and AI prompt engineer.

Your task is to create a TRUE character reference sheet prompt set, not generic character prompts. First lock a consistent design bible in your reasoning, then write every output so it repeats the same identity anchors, proportions, wardrobe, colors, materials, face, hair, and distinctive marks.

Character sheet controls:
- Visual style: {style}
- Sheet format: {sheet_format}
- Pose standard: {pose_standard}
- Background/layout: {background_style}
- Consistency level: {consistency_level}
- Target generator: {model_target}
- Number of expression variations: {num_expressions}
- Expression set: {expression_set}
- Include callouts: {include_callouts}
- Custom system guidance: {custom_system_prompt}

Respond in this EXACT JSON format:
{{
    "character_summary": "A locked design bible: identity, age range, body proportions, face shape, hair, eyes, skin/surface, wardrobe, color palette, materials, accessories, silhouette, and 5-8 consistency anchors that must repeat in every view.",
    "full_body_front": "Single-view prompt for a full-body front reference. Use the chosen style, pose standard, clean sheet background, exact wardrobe/materials/colors, visible silhouette, readable shoes/hands, neutral expression, no perspective distortion.",
    "full_body_side": "Single-view prompt for a full-body side/profile reference. Preserve exact proportions, outfit, hairstyle volume, accessories, colors, and silhouette. Mention side-view construction details.",
    "full_body_back": "Single-view prompt for a full-body back reference. Preserve exact design, show rear hair shape, back of outfit, seams, accessories, footwear, and silhouette details.",
    "full_body_3quarter": "Single-view prompt for a full-body 3/4 reference. Preserve the exact same character, with a readable pose and clear front/side design information.",
    "headshot_front": "Head-and-shoulders front reference prompt. Preserve face shape, eye color, hairstyle, hairline/bangs, ears, markings, makeup/facial hair, expression baseline, and lighting/style.",
    "headshot_3quarter": "Head-and-shoulders 3/4 reference prompt. Preserve exact face identity, hair volume, profile cues, and distinguishing marks.",
    "headshot_profile": "Head-and-shoulders side profile reference prompt. Preserve nose, chin, forehead, ear, neck, hair silhouette, and profile identity markers.",
    "expression_sheet": "A grid prompt with exactly {num_expressions} labeled expressions from the chosen expression set. Same face, same hairstyle, same lighting, same angle family, clear labels if the target model supports text; otherwise visual separation without text.",
    "full_body_grid": "A single prompt for a clean turnaround sheet with front, side, back, and 3/4 full-body views of the exact same character. White/model-sheet background, consistent scale, aligned feet, same outfit and proportions, no extra characters.",
    "face_grid": "A single prompt for a face reference grid with front, 3/4, profile, and expression variants. Same identity, same hairstyle, same lighting, consistent facial proportions.",
    "master_sheet": "A single complete character sheet prompt containing full-body turnaround, face close-ups, expression row, material/color callouts, accessory callouts, silhouette notes, and consistency anchors, arranged as a professional reference sheet."
}}

Critical rules:
- Every output must describe the SAME character, not variations.
- Repeat the key identity anchors in every prompt.
- For real-photo styles, phrase prompts as studio reference photography, same actor/model, same wardrobe, consistent lens and lighting, clean casting/costume reference sheet.
- For illustrated styles, phrase prompts as production model sheet / character design reference sheet.
- Avoid dynamic perspective, cropped limbs, changing outfits, changing age, changing facial structure, changing hair length/color, and different art styles between panels.
- Keep grid prompts as single-image sheet layouts with multiple panels, not multiple separate generations.
- Output ONLY valid JSON."""


def build_system_prompt(
    style: str = "Anime/Manga",
    num_expressions: int = 4,
    sheet_format: str = "production turnaround sheet",
    pose_standard: str = "relaxed neutral stance",
    background_style: str = "clean white background",
    consistency_level: str = "strict - exact outfit and proportions",
    model_target: str = "general image models",
    expression_set: str = "core emotions",
    include_callouts: bool = True,
    custom_system_prompt: str = "",
) -> str:
    return SYSTEM_PROMPT.format(
        style=style,
        num_expressions=num_expressions,
        sheet_format=sheet_format,
        pose_standard=pose_standard,
        background_style=background_style,
        consistency_level=consistency_level,
        model_target=model_target,
        expression_set=expression_set,
        include_callouts="yes" if include_callouts else "no",
        custom_system_prompt=custom_system_prompt.strip() or "No additional custom guidance.",
    )


def build_user_prompt(character_description: str) -> str:
    return f"""Create a complete production character reference sheet prompt set for this character:

{character_description}

Lock the character design first, then generate all view prompts with strict visual consistency. Output valid JSON only."""
