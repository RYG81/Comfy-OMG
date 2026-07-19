"""
System prompts for the Emotion Director node.
Direct and intensify emotional content in prompts.
"""

PRIMARY_EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust",
    "love", "hope", "pride", "shame", "guilt", "envy", "jealousy",
    "anxiety", "serenity", "melancholy", "nostalgia", "longing",
    "determination", "despair", "awe", "wonder", "confusion",
]

EMOTION_INTENSITY = [
    "whisper - barely perceptible, subtle hint",
    "gentle - soft, understated presence",
    "present - clearly visible, balanced",
    "strong - dominant, unmistakable",
    "overwhelming - consuming, intense",
    "explosive - extreme, over the top",
]

EXPRESSION_STYLE = [
    "realistic - natural human expression",
    "stylized - exaggerated for art style",
    "subtle - micro-expressions and nuance",
    "dramatic - theatrical and bold",
    "anime - anime/manga expression style",
    "painterly - artistic interpretation",
]

BODY_LANGUAGE_INTENSITY = [
    "none - neutral body language",
    "minimal - slight postural cues",
    "moderate - clear body language",
    "expressive - strong physical expression",
    "dramatic - full body emotion",
]

SYSTEM_PROMPT = """You are an expert at conveying emotion through visual descriptions for AI image generation.

Emotion Direction:
- Primary Emotion: {primary_emotion}
- Secondary Emotion (blend): {secondary_emotion}
- Intensity: {intensity}
- Expression Style: {expression_style}
- Body Language: {body_language}

Your task: Transform or enhance the prompt to convey the specified emotion(s).

Consider ALL aspects that convey emotion:
1. Facial expression details
2. Eye appearance and gaze
3. Body posture and gesture
4. Color palette suggestions
5. Lighting mood
6. Environmental elements that reinforce mood
7. Composition choices

Respond in this EXACT JSON format:
{{
    "emotional_prompt": "The prompt rewritten to convey the specified emotion(s)",
    "facial_direction": "Specific facial expression guidance",
    "eye_description": "How the eyes should convey emotion",
    "body_language": "Posture and gesture guidance",
    "color_mood": "Color palette that reinforces the emotion",
    "lighting_mood": "Lighting that enhances the emotional tone",
    "environmental_mood": "Environmental elements that support the mood",
    "composition_tips": "How composition can enhance emotion",
    "emotional_tags": "Emotion-related tags for the prompt",
    "avoid": "What to avoid that would undermine the emotion"
}}

Output ONLY valid JSON."""

def build_system_prompt(
    primary_emotion: str = "joy",
    secondary_emotion: str = "none",
    intensity: str = "present - clearly visible, balanced",
    expression_style: str = "realistic - natural human expression",
    body_language: str = "moderate - clear body language",
) -> str:
    return SYSTEM_PROMPT.format(
        primary_emotion=primary_emotion,
        secondary_emotion=secondary_emotion if secondary_emotion != "none" else "none (single emotion)",
        intensity=intensity,
        expression_style=expression_style,
        body_language=body_language,
    )

def build_user_prompt(prompt: str) -> str:
    return f"""Direct emotion in this prompt:

"{prompt}"

Apply the specified emotional direction. Output valid JSON only."""
