"""
Prompts for character continuity and sheet utility nodes.
"""

ANCHOR_MODES = [
    "strict identity lock",
    "animation model sheet",
    "real photo identity",
    "costume continuity",
    "stylized character",
]

OUTFIT_THEMES = [
    "casual", "formal", "streetwear", "fantasy armor", "sci-fi suit",
    "historical", "school/uniform", "royal", "adventure", "winter",
    "summer", "ceremonial", "battle damaged", "fashion editorial",
]

POSE_SETS = [
    "basic turnaround poses", "action poses", "combat poses", "acting poses",
    "fashion poses", "casual everyday poses", "hero poses", "villain poses",
    "dance poses", "sports poses", "realistic photo poses",
]

EXPRESSION_SETS = [
    "core emotions", "subtle realistic", "anime expressions",
    "animation acting", "dramatic cinematic", "comedy expressions",
    "villain expressions", "romantic expressions",
]

SHEET_STYLES = [
    "match source style", "anime model sheet", "real photo reference",
    "concept art sheet", "3D render sheet", "fashion design sheet",
    "clean production sheet",
]


ANCHOR_SYSTEM_PROMPT = """You are a character continuity supervisor. Extract stable identity anchors from the provided character description, prompt, sheet text, or image-analysis text.

Mode: {mode}

Respond in this EXACT JSON format:
{{
    "identity_anchors": "The non-negotiable identity details: age range, face shape, eye/hair/skin, body proportions, silhouette, marks, species/type, wardrobe anchors.",
    "face_anchors": "Face-specific anchors that should remain stable across views and generations.",
    "body_anchors": "Body/proportion/posture anchors.",
    "wardrobe_anchors": "Clothing, materials, accessories, colors, footwear, props.",
    "style_anchors": "Art/photo/render style anchors that should remain stable.",
    "negative_prompt": "Things that would break continuity or identity.",
    "compact_anchor_prompt": "A compact reusable prompt fragment containing the most important anchors."
}}

Rules:
- Preserve details, do not invent unless required to fill gaps.
- If details conflict, state the most stable interpretation.
- Output ONLY valid JSON."""


def build_anchor_system_prompt(mode: str) -> str:
    return ANCHOR_SYSTEM_PROMPT.format(mode=mode)


def build_anchor_user_prompt(source_text: str) -> str:
    return f"""Extract character consistency anchors from this source:

{source_text}

Output valid JSON only."""


OUTFIT_SHEET_SYSTEM_PROMPT = """You are a costume designer and character continuity artist.

Create a same-character outfit sheet. The character identity must stay unchanged; only outfits change.

Controls:
- Outfit theme: {outfit_theme}
- Number of outfits: {outfit_count}
- Sheet style: {sheet_style}
- Keep palette: {keep_palette}
- Include construction notes: {include_construction_notes}
- Custom guidance: {custom_system_prompt}

Respond in this EXACT JSON format:
{{
    "outfit_sheet_prompt": "A single prompt for a multi-panel outfit sheet showing the same character in {outfit_count} outfits, consistent identity, aligned full-body panels, clean background.",
    "outfit_breakdown": "Numbered outfit descriptions, each preserving identity while changing clothing.",
    "material_color_notes": "Fabric, material, palette, accessory, and footwear notes.",
    "continuity_anchors": "Identity details that must remain identical across all outfits.",
    "negative_prompt": "Avoid identity drift, changed body/face/hair, inconsistent scale, extra people, messy layout."
}}

Output ONLY valid JSON."""


def build_outfit_sheet_system_prompt(
    outfit_theme: str, outfit_count: int, sheet_style: str,
    keep_palette: bool, include_construction_notes: bool, custom_system_prompt: str,
) -> str:
    return OUTFIT_SHEET_SYSTEM_PROMPT.format(
        outfit_theme=outfit_theme,
        outfit_count=outfit_count,
        sheet_style=sheet_style,
        keep_palette="yes" if keep_palette else "no",
        include_construction_notes="yes" if include_construction_notes else "no",
        custom_system_prompt=custom_system_prompt.strip() or "No additional guidance.",
    )


def build_outfit_sheet_user_prompt(character_text: str) -> str:
    return f"""Create an outfit sheet for this character:

{character_text}

Output valid JSON only."""


POSE_SHEET_SYSTEM_PROMPT = """You are a pose sheet director and character model-sheet artist.

Create a same-character pose sheet. The character identity, outfit, proportions, and style must stay unchanged.

Controls:
- Pose set: {pose_set}
- Number of poses: {pose_count}
- Sheet style: {sheet_style}
- Include silhouette notes: {include_silhouette_notes}
- Custom guidance: {custom_system_prompt}

Respond in this EXACT JSON format:
{{
    "pose_sheet_prompt": "A single prompt for a multi-panel pose sheet with {pose_count} poses, same character, same outfit, consistent scale, clean production layout.",
    "pose_breakdown": "Numbered pose descriptions with body, hands, gaze, weight, and gesture notes.",
    "silhouette_notes": "Readable silhouette and action-line notes.",
    "continuity_anchors": "Identity and outfit details that must not change.",
    "negative_prompt": "Avoid anatomy drift, changing outfit, extra limbs, duplicate bodies merging, inconsistent face, messy panels."
}}

Output ONLY valid JSON."""


def build_pose_sheet_system_prompt(
    pose_set: str, pose_count: int, sheet_style: str,
    include_silhouette_notes: bool, custom_system_prompt: str,
) -> str:
    return POSE_SHEET_SYSTEM_PROMPT.format(
        pose_set=pose_set,
        pose_count=pose_count,
        sheet_style=sheet_style,
        include_silhouette_notes="yes" if include_silhouette_notes else "no",
        custom_system_prompt=custom_system_prompt.strip() or "No additional guidance.",
    )


def build_pose_sheet_user_prompt(character_text: str) -> str:
    return f"""Create a pose sheet for this character:

{character_text}

Output valid JSON only."""


EXPRESSION_SHEET_SYSTEM_PROMPT = """You are a facial expression sheet artist and character continuity supervisor.

Create a same-character expression sheet. Preserve face structure, age, hairline, eyes, markings, makeup/facial hair, and style.

Controls:
- Expression set: {expression_set}
- Number of expressions: {expression_count}
- Sheet style: {sheet_style}
- Include acting notes: {include_acting_notes}
- Custom guidance: {custom_system_prompt}

Respond in this EXACT JSON format:
{{
    "expression_sheet_prompt": "A single prompt for a clean face/expression grid with {expression_count} expressions, same character identity, same angle family, same lighting.",
    "expression_breakdown": "Numbered expression descriptions with eyebrows, eyes, mouth, cheeks, head tilt, and emotional intent.",
    "face_continuity_anchors": "Face details that must remain identical.",
    "acting_notes": "Performance notes for readable expressions without changing identity.",
    "negative_prompt": "Avoid face drift, different age, different hairstyle, asymmetrical errors, messy panels, extra faces."
}}

Output ONLY valid JSON."""


def build_expression_sheet_system_prompt(
    expression_set: str, expression_count: int, sheet_style: str,
    include_acting_notes: bool, custom_system_prompt: str,
) -> str:
    return EXPRESSION_SHEET_SYSTEM_PROMPT.format(
        expression_set=expression_set,
        expression_count=expression_count,
        sheet_style=sheet_style,
        include_acting_notes="yes" if include_acting_notes else "no",
        custom_system_prompt=custom_system_prompt.strip() or "No additional guidance.",
    )


def build_expression_sheet_user_prompt(character_text: str) -> str:
    return f"""Create an expression sheet for this character:

{character_text}

Output valid JSON only."""


CONTINUITY_CHECK_SYSTEM_PROMPT = """You are a prompt continuity checker. Compare multiple prompts intended to describe the same character, style, or scene.

Check mode: {check_mode}

Respond in this EXACT JSON format:
{{
    "continuity_score": "Score from 0-10 with one-sentence reason.",
    "stable_anchors": "Details that remain consistent.",
    "drift_risks": "Details that may drift or contradict each other.",
    "conflicts": "Direct contradictions between prompts.",
    "missing_anchors": "Important consistency details that are missing.",
    "fixed_master_prompt": "A consolidated prompt that resolves conflicts and locks continuity.",
    "negative_prompt": "Things to avoid to preserve continuity."
}}

Output ONLY valid JSON."""


def build_continuity_check_system_prompt(check_mode: str) -> str:
    return CONTINUITY_CHECK_SYSTEM_PROMPT.format(check_mode=check_mode)


def build_continuity_check_user_prompt(prompt_a: str, prompt_b: str, prompt_c: str) -> str:
    return f"""Compare these prompts for continuity:

PROMPT A:
{prompt_a}

PROMPT B:
{prompt_b}

PROMPT C:
{prompt_c}

Output valid JSON only."""
