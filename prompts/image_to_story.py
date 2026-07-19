"""
System prompts for the Image to Story node.
Generate narrative content from an image.
"""

STORY_TYPE = [
    "custom", "narrative description", "short story", "flash fiction",
    "poem", "haiku", "screenplay scene", "novel excerpt", "journal entry",
    "news article", "documentary narration", "mythology", "fairy tale",
    "horror story", "romance scene", "action sequence", "mystery setup",
]

TONE = [
    "custom", "neutral", "dramatic", "comedic", "mysterious", "romantic",
    "tragic", "hopeful", "dark", "whimsical", "epic", "intimate",
    "suspenseful", "melancholic", "joyful", "philosophical",
]

PERSPECTIVE = [
    "custom", "third person omniscient", "third person limited",
    "first person - character in scene", "first person - observer",
    "second person", "documentary style", "narrator outside story",
]

LENGTH = [
    "brief - 2-3 sentences",
    "short - 1 paragraph",
    "medium - 2-3 paragraphs",
    "long - 4-5 paragraphs",
    "detailed - 6+ paragraphs",
]

SYSTEM_PROMPT = """You are a creative writer who crafts stories and narratives from visual scenes.

Story Parameters:
- Type: {story_type}
- Tone: {tone}
- Perspective: {perspective}
- Length: {length}

Analyze the image and create narrative content that:
1. Captures the visual elements and mood
2. Creates compelling characters from the subjects
3. Builds atmosphere and setting
4. Tells an engaging story or creates vivid description
5. Matches the specified tone and style

Respond in this EXACT JSON format:
{{
    "story": "The main narrative content",
    "title": "A title for this piece",
    "setting_description": "Expanded setting details for world-building",
    "character_profiles": "Brief profiles of characters shown",
    "mood_analysis": "The emotional atmosphere captured",
    "narrative_hooks": "Story elements that could be expanded",
    "dialogue_sample": "Sample dialogue these characters might have",
    "before_scene": "What might have happened before this moment",
    "after_scene": "What might happen after this moment",
    "themes": "Themes present in this visual narrative"
}}

Create engaging, vivid narrative content. Output ONLY valid JSON."""

def build_system_prompt(**kwargs) -> str:
    return SYSTEM_PROMPT.format(
        story_type=kwargs.get("story_type", "narrative description"),
        tone=kwargs.get("tone", "neutral"),
        perspective=kwargs.get("perspective", "third person omniscient"),
        length=kwargs.get("length", "medium - 2-3 paragraphs"),
    )

USER_PROMPT = "Analyze this image and create a narrative based on what you see. Output valid JSON only."
