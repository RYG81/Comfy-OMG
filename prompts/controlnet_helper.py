"""
System prompts for the ControlNet Helper node.
Optimize prompts for ControlNet workflows.
"""

CONTROLNET_TYPE = [
    "canny", "depth", "pose/openpose", "scribble", "lineart",
    "softedge", "normal", "segmentation", "shuffle", "tile",
    "inpaint", "ip-adapter", "reference", "qr_code", "brightness",
]

CONTROLNET_STRENGTH = [
    "very_low - 0.2-0.3",
    "low - 0.4-0.5",
    "medium - 0.6-0.7",
    "high - 0.8-0.9",
    "very_high - 0.95-1.0",
]

SYSTEM_PROMPT = """You are an expert at optimizing prompts for ControlNet workflows in Stable Diffusion.

ControlNet Type: {controlnet_type}
Expected Strength: {strength}

ControlNet-specific considerations:
- Canny/Lineart: Prompt should describe line quality and edge details
- Depth: Describe depth layers, foreground/background relationships
- Pose: Focus on subject details, let pose handle positioning
- Segmentation: Describe each segment's content clearly
- Tile: Describe patterns and textures for upscaling
- Reference/IP-Adapter: Focus on differences from reference

Respond in this EXACT JSON format:
{{
    "optimized_prompt": "Prompt optimized for this ControlNet type",
    "structural_elements": "Elements the ControlNet will handle (don't over-describe)",
    "detail_elements": "Elements to describe in detail (ControlNet won't handle)",
    "style_elements": "Style, lighting, and mood descriptions",
    "strength_notes": "How strength affects what prompt needs",
    "what_to_omit": "What to leave out (handled by ControlNet)",
    "what_to_emphasize": "What to emphasize in prompt",
    "negative_prompt": "Optimized negative for this ControlNet",
    "common_mistakes": "Common mistakes with this ControlNet type",
    "workflow_tips": "Tips for best results"
}}

Output ONLY valid JSON."""

def build_system_prompt(
    controlnet_type: str = "canny",
    strength: str = "medium - 0.6-0.7",
) -> str:
    return SYSTEM_PROMPT.format(
        controlnet_type=controlnet_type,
        strength=strength,
    )

def build_user_prompt(prompt: str, control_description: str = "") -> str:
    desc = f"\n\nControl Image Description: {control_description}" if control_description else ""
    return f"""Optimize this prompt for ControlNet:

"{prompt}"{desc}

Make it work best with the specified ControlNet type. Output valid JSON only."""
