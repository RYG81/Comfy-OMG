"""
System prompts and dropdown options for video model prompt generators.
"""

VIDEO_STYLE = [
    "custom", "cinematic realism", "documentary realism", "anime",
    "stylized 3D", "concept art motion", "music video", "commercial product",
    "fashion film", "dark fantasy", "sci-fi", "dreamlike surreal",
]

SHOT_TYPE = [
    "custom", "establishing shot", "wide shot", "medium shot", "close-up",
    "extreme close-up", "two-shot", "tracking shot", "point-of-view",
    "over-the-shoulder", "insert shot", "macro shot",
]

CAMERA_MOTION = [
    "custom", "locked-off tripod", "slow push-in", "slow pull-back",
    "left-to-right dolly", "right-to-left dolly", "orbit around subject",
    "handheld subtle sway", "crane up", "crane down", "drone glide",
    "tilt up", "tilt down", "pan left", "pan right", "rack focus",
]

SUBJECT_MOTION = [
    "custom", "almost still", "gentle natural movement", "walking",
    "running", "turning toward camera", "turning away from camera",
    "reaching", "gesturing", "hair and clothing moving in wind",
    "object rotating", "vehicle moving", "crowd motion", "explosive action",
]

MOTION_INTENSITY = [
    "custom", "very subtle", "subtle", "moderate", "dynamic", "fast",
    "chaotic but readable",
]

PACING = [
    "custom", "slow contemplative", "steady cinematic", "rising tension",
    "quick energetic", "single continuous action", "loop-friendly",
]

LIGHTING = [
    "custom", "natural daylight", "golden hour", "blue hour", "moonlight",
    "soft studio", "dramatic low-key", "neon practicals", "firelight",
    "volumetric rays", "high contrast", "soft overcast",
]

TRANSITION_STYLE = [
    "custom", "no transition", "smooth continuous motion", "start still then move",
    "end on held frame", "loopable ending", "match first and last frame",
]

MODEL_PROFILES = {
    "LTXV 2.3": {
        "name": "LTXV 2.3",
        "bias": (
            "Favor concise but explicit video prompts with clear camera movement, "
            "temporal continuity, physical plausibility, and a clean negative prompt. "
            "Avoid overloading the prompt with too many simultaneous actions."
        ),
    },
    "Wan 2.2": {
        "name": "Wan 2.2",
        "bias": (
            "Favor richly described cinematic prompts with strong subject action, "
            "shot progression, camera path, lighting continuity, and motion constraints. "
            "Keep identities, clothing, and scene geometry stable across frames."
        ),
    },
}

SYSTEM_PROMPT = """You are an expert text-to-video prompt director for {model_name}.

Model-specific guidance:
{model_bias}

Create prompts for short generative video clips. The prompt must control time, movement, camera behavior, subject continuity, lighting continuity, and what must remain stable across frames.

Video controls:
- Duration: {duration}
- Aspect ratio: {aspect_ratio}
- Style: {video_style}
- Shot type: {shot_type}
- Camera motion: {camera_motion}
- Subject motion: {subject_motion}
- Motion intensity: {motion_intensity}
- Pacing: {pacing}
- Lighting: {lighting}
- Transition / loop behavior: {transition_style}
- First frame: {first_frame}
- Last frame: {last_frame}
- Additional details: {additional_details}

Custom system guidance from the user:
{custom_system_prompt}

Respond in this EXACT JSON format:
{{
    "positive_prompt": "A direct, model-ready video prompt describing subject, scene, camera movement, subject movement, lighting, style, temporal progression, and continuity.",
    "negative_prompt": "Video artifacts and unwanted elements to avoid, including flicker, morphing, identity drift, warped anatomy, unstable camera, broken motion, and style drift.",
    "shot_plan": "A compact timeline of the clip from first frame to last frame.",
    "motion_notes": "Specific motion and continuity constraints for camera, subject, environment, fabric/hair, particles, and lighting.",
    "model_settings": "Recommended practical notes for this model profile, such as prompt emphasis, loop suitability, and whether the prompt is better for image-to-video or text-to-video."
}}

Rules:
- Keep the positive prompt usable as a single prompt string.
- Make motion specific, physically plausible, and easy for a video model to follow.
- Do not request cuts unless the user asks; prefer one continuous shot.
- Preserve subject identity, wardrobe, lighting direction, and scene layout over time.
- Output ONLY valid JSON."""


def build_system_prompt(
    model_profile: str,
    duration: str,
    aspect_ratio: str,
    video_style: str,
    shot_type: str,
    camera_motion: str,
    subject_motion: str,
    motion_intensity: str,
    pacing: str,
    lighting: str,
    transition_style: str,
    first_frame: str,
    last_frame: str,
    additional_details: str,
    custom_system_prompt: str,
) -> str:
    profile = MODEL_PROFILES[model_profile]

    def clean(value: str, fallback: str = "infer from concept") -> str:
        text = str(value or "").strip()
        if not text or text == "custom":
            return fallback
        return text

    return SYSTEM_PROMPT.format(
        model_name=profile["name"],
        model_bias=profile["bias"],
        duration=clean(duration),
        aspect_ratio=clean(aspect_ratio),
        video_style=clean(video_style),
        shot_type=clean(shot_type),
        camera_motion=clean(camera_motion),
        subject_motion=clean(subject_motion),
        motion_intensity=clean(motion_intensity),
        pacing=clean(pacing),
        lighting=clean(lighting),
        transition_style=clean(transition_style),
        first_frame=clean(first_frame, "infer a strong opening frame"),
        last_frame=clean(last_frame, "infer a coherent ending frame"),
        additional_details=clean(additional_details, "none"),
        custom_system_prompt=custom_system_prompt.strip() or "No additional custom system guidance.",
    )


def build_user_prompt(concept: str) -> str:
    return f"""Create a model-ready video generation prompt for this concept:

{concept}

Output valid JSON only."""
