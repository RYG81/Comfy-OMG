"""
System prompts for the Action Choreographer node.
Design dynamic action sequences and poses.
"""

ACTION_TYPE = [
    "custom", "combat - melee", "combat - ranged", "combat - martial arts",
    "combat - sword fight", "combat - magic battle", "chase - running",
    "chase - vehicle", "chase - flying", "sports - general", "sports - ball",
    "sports - combat", "dance - solo", "dance - partner", "dance - group",
    "acrobatics", "parkour", "swimming", "climbing", "falling", "jumping",
    "transformation", "power activation", "dramatic gesture",
]

INTENSITY_LEVEL = [
    "calm - controlled movement",
    "moderate - active but composed",
    "energetic - dynamic motion",
    "intense - peak action",
    "explosive - maximum energy",
]

MOTION_PHASE = [
    "custom", "anticipation - winding up", "action - mid-motion",
    "follow-through - completing motion", "recovery - returning to stance",
    "peak moment - height of action", "impact moment - collision/contact",
    "freeze frame - dramatic pause",
]

CAMERA_STYLE = [
    "custom", "static - clear view", "tracking - following motion",
    "action lines - manga style", "motion blur - speed emphasis",
    "freeze frame - stopped time", "dynamic angle - dramatic perspective",
    "multi-exposure - motion trail",
]

PARTICIPANT_COUNT = [
    "solo - single figure",
    "duel - two opponents",
    "group vs one - many against one",
    "team vs team - group battle",
    "crowd - mass action",
]

SYSTEM_PROMPT = """You are an expert action choreographer and fight director. You design dynamic, visually compelling action moments.

Design an action scene with these parameters:
- Action Type: {action_type}
- Intensity: {intensity}
- Motion Phase: {motion_phase}
- Camera Style: {camera_style}
- Participants: {participants}
- Environment: {environment}
- Weapons/Props: {weapons}
- Special Effects: {effects}

{custom_instructions}

Respond in this EXACT JSON format:
{{
    "action_prompt": "Complete action scene description for image generation",
    "pose_description": "Detailed body positions of each participant",
    "motion_direction": "Direction and flow of movement",
    "impact_point": "Where the action is focused/connecting",
    "facial_expressions": "Expressions during the action",
    "clothing_dynamics": "How clothing/hair responds to motion",
    "environmental_interaction": "How the scene interacts with environment",
    "energy_visualization": "Visual effects (motion lines, impact effects)",
    "camera_position": "Optimal viewing angle for this action",
    "timing_notes": "What moment in the action is captured",
    "composition_tips": "How to frame the action effectively",
    "negative_prompt": "What to avoid for clean action"
}}

Create a dynamic, readable action moment. Output ONLY valid JSON."""

def build_system_prompt(**kwargs) -> str:
    return SYSTEM_PROMPT.format(
        action_type=kwargs.get("action_type", "custom"),
        intensity=kwargs.get("intensity", "energetic - dynamic motion"),
        motion_phase=kwargs.get("motion_phase", "action - mid-motion"),
        camera_style=kwargs.get("camera_style", "dynamic angle - dramatic perspective"),
        participants=kwargs.get("participants", "solo - single figure"),
        environment=kwargs.get("environment", "custom"),
        weapons=kwargs.get("weapons", "none"),
        effects=kwargs.get("effects", "custom"),
        custom_instructions=f"Additional Instructions: {kwargs.get('custom_instructions', '')}" if kwargs.get('custom_instructions') else "",
    )

def build_user_prompt(action_description: str = "") -> str:
    if action_description.strip():
        return f"Choreograph this action:\n\n{action_description}\n\nOutput valid JSON only."
    return "Design a dynamic action moment based on the specified parameters. Output valid JSON only."
