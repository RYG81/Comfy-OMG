"""
System prompts for the Hand Pose Helper node.
Generate detailed, AI-friendly hand pose descriptions.
"""

HAND_ACTION = [
    "custom", "relaxed open", "relaxed closed", "fist", "pointing",
    "peace sign", "thumbs up", "ok sign", "waving", "beckoning",
    "grabbing", "holding object", "pinching", "spreading fingers",
    "interlocked fingers", "prayer hands", "cupping", "claw",
    "gun gesture", "rock on", "counting", "typing", "writing",
    "playing instrument", "casting spell", "reaching", "pushing",
    "pulling", "caressing", "punching", "slapping", "catching",
]

HAND_POSITION = [
    "custom", "at sides", "in front of body", "behind back",
    "on hips", "crossed on chest", "raised above head", "reaching up",
    "reaching forward", "reaching to side", "near face", "touching face",
    "in pockets", "clasped together", "resting on surface", "gesturing",
    "one hand up one down", "asymmetric natural",
]

WHICH_HANDS = [
    "both hands same pose", "right hand only", "left hand only",
    "both hands different poses", "one hand visible one hidden",
]

DETAIL_LEVEL = [
    "basic - general shape",
    "standard - clear finger positions",
    "detailed - specific finger angles",
    "hyper-detailed - every joint described",
]

SYSTEM_PROMPT = """You are an expert at describing hand poses for AI image generation. Hands are notoriously difficult for AI — your descriptions must be exceptionally clear.

Hand Pose Specifications:
- Action: {action}
- Position: {position}
- Which Hands: {which_hands}
- Detail Level: {detail_level}
- Holding Object: {holding}
- Interaction: {interaction}

CRITICAL RULES FOR AI-FRIENDLY HAND DESCRIPTIONS:
1. Specify EXACT finger positions (extended, curled, bent at which joint)
2. State number of fingers visible
3. Describe thumb position explicitly
4. Mention palm direction (facing up, down, toward viewer, away)
5. Avoid ambiguous poses — simpler is better for AI
6. If holding something, describe grip clearly

Respond in this EXACT JSON format:
{{
    "hand_prompt": "Complete hand description optimized for AI generation",
    "right_hand": "Detailed right hand description (if visible)",
    "left_hand": "Detailed left hand description (if visible)",
    "finger_positions": "Each finger described (thumb through pinky)",
    "palm_orientation": "Which way palms are facing",
    "wrist_angle": "Wrist position and angle",
    "hand_interaction": "How hands interact with each other or objects",
    "simplification_tips": "How to simplify if AI struggles",
    "hand_tags": "Tags that help with hand generation",
    "negative_prompt": "Common hand errors to avoid",
    "difficulty_rating": "How hard this pose is for AI (easy/medium/hard)",
    "alternative_pose": "Simpler alternative if this fails"
}}

Make the description as clear and unambiguous as possible. Output ONLY valid JSON."""

def build_system_prompt(**kwargs) -> str:
    return SYSTEM_PROMPT.format(
        action=kwargs.get("action", "relaxed open"),
        position=kwargs.get("position", "at sides"),
        which_hands=kwargs.get("which_hands", "both hands same pose"),
        detail_level=kwargs.get("detail_level", "standard - clear finger positions"),
        holding=kwargs.get("holding", "nothing"),
        interaction=kwargs.get("interaction", "none"),
    )

def build_user_prompt(description: str = "") -> str:
    if description.strip():
        return f"Create hand pose for:\n\n{description}\n\nOutput valid JSON only."
    return "Generate a detailed, AI-friendly hand pose description. Output valid JSON only."
