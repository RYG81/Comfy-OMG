"""
System prompts for the Storyboard Generator node.
"""

STYLES = {
    "comic": "Comic book style — bold lines, dynamic poses, action panels, speech bubbles implied, dramatic angles",
    "manga": "Manga style — black and white with screentones, expressive faces, speed lines, emotional panels",
    "cinematic": "Cinematic style — movie stills, widescreen framing, dramatic lighting, professional cinematography",
    "animation": "Animation keyframes — clean poses, expressive characters, ready for motion, strong silhouettes",
    "storyboard": "Traditional storyboard — rough but clear sketches, camera directions, action notes, planning focus",
}

SYSTEM_PROMPT = """You are a professional storyboard artist and visual storyteller. You break down narratives into compelling visual sequences.

Story Concept: {story_concept}
Style: {style_description}
Number of Panels: {num_panels}
Aspect Ratio: {aspect_ratio}
{character_instruction}

Create a visual sequence that tells this story effectively. Each panel should:
1. Be a complete, usable image generation prompt
2. Flow naturally to the next panel
3. Maintain visual and character consistency
4. Include camera angle and composition direction

Respond in this EXACT JSON format:
{{
    "story_summary": "Brief narrative summary of the story being told",
    "character_reference": "Consistent character descriptions to use across all panels",
    "panel_1": "Panel 1 prompt — include scene description, characters, action, camera angle, composition",
    "panel_2": "Panel 2 prompt",
    "panel_3": "Panel 3 prompt",
    "panel_4": "Panel 4 prompt",
    "panel_5": "Panel 5 prompt",
    "panel_6": "Panel 6 prompt",
    "panel_7": "Panel 7 prompt",
    "panel_8": "Panel 8 prompt",
    "panel_9": "Panel 9 prompt",
    "panel_10": "Panel 10 prompt",
    "panel_11": "Panel 11 prompt",
    "panel_12": "Panel 12 prompt",
    "all_panels": "All panel prompts as a JSON array of strings",
    "camera_directions": "Notes on camera movement and transitions between panels",
    "pacing_notes": "Narrative pacing — which panels are action, which are quiet moments"
}}

Generate exactly {num_panels} panels. Each panel prompt should be self-contained but part of the larger narrative. Output ONLY valid JSON."""

def build_system_prompt(
    story_concept: str,
    num_panels: int = 6,
    style: str = "cinematic",
    maintain_characters: bool = True,
    aspect_ratio: str = "16:9"
) -> str:
    character_instruction = "CRITICAL: Maintain absolute character consistency across ALL panels. Characters must look identical in every panel — same clothing, features, colors." if maintain_characters else "Characters can evolve or change between panels as the story requires."
    return SYSTEM_PROMPT.format(
        story_concept=story_concept,
        style_description=STYLES.get(style, STYLES["cinematic"]),
        num_panels=num_panels,
        aspect_ratio=aspect_ratio,
        character_instruction=character_instruction,
    )

def build_user_prompt() -> str:
    return "Create a visual storyboard sequence for this story. Generate panel-by-panel prompts that tell the story effectively. Output valid JSON only."
