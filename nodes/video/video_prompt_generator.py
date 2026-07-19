"""
Video prompt generator nodes for LTXV 2.3 and Wan 2.2 style workflows.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.video_prompt import (
    CAMERA_MOTION,
    LIGHTING,
    MOTION_INTENSITY,
    PACING,
    SHOT_TYPE,
    SUBJECT_MOTION,
    TRANSITION_STYLE,
    VIDEO_STYLE,
    build_system_prompt,
    build_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class _BaseVideoPromptGenerator:
    CATEGORY = "Ollama-Magic-Nodes/Video"
    FUNCTION = "generate_video_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "positive_prompt", "negative_prompt", "shot_plan",
        "motion_notes", "model_settings",
    )
    OUTPUT_TOOLTIPS = (
        "Model-ready positive video prompt",
        "Negative prompt focused on video artifacts",
        "First-to-last-frame shot plan",
        "Motion and continuity constraints",
        "Practical notes for this model profile",
    )
    MODEL_PROFILE = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "concept": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Describe the video you want to generate",
                }),
            },
            "optional": {
                "duration": (["custom", "2 seconds", "5 seconds", "10 seconds", "15 seconds", "20 seconds", "30 seconds", ], {
                    "default": "5 seconds",
                    "tooltip": "Target clip duration",
                }),
                "duration_custom": ("STRING", {"default": "", "tooltip": "Custom duration"}),
                "aspect_ratio": (["custom", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {
                    "default": "16:9",
                    "tooltip": "Target video aspect ratio",
                }),
                "aspect_custom": ("STRING", {"default": "", "tooltip": "Custom aspect ratio"}),
                "video_style": (VIDEO_STYLE, {"default": "cinematic realism", "tooltip": "Overall visual style"}),
                "style_custom": ("STRING", {"default": "", "tooltip": "Custom video style"}),
                "shot_type": (SHOT_TYPE, {"default": "medium shot", "tooltip": "Shot size/type"}),
                "shot_custom": ("STRING", {"default": "", "tooltip": "Custom shot type"}),
                "camera_motion": (CAMERA_MOTION, {"default": "slow push-in", "tooltip": "Camera path over time"}),
                "camera_custom": ("STRING", {"default": "", "tooltip": "Custom camera motion"}),
                "subject_motion": (SUBJECT_MOTION, {"default": "gentle natural movement", "tooltip": "Subject action over time"}),
                "subject_motion_custom": ("STRING", {"default": "", "tooltip": "Custom subject motion"}),
                "motion_intensity": (MOTION_INTENSITY, {"default": "subtle", "tooltip": "How strong the movement should be"}),
                "motion_intensity_custom": ("STRING", {"default": "", "tooltip": "Custom motion intensity"}),
                "pacing": (PACING, {"default": "steady cinematic", "tooltip": "Temporal feel"}),
                "pacing_custom": ("STRING", {"default": "", "tooltip": "Custom pacing"}),
                "lighting": (LIGHTING, {"default": "golden hour", "tooltip": "Lighting continuity"}),
                "lighting_custom": ("STRING", {"default": "", "tooltip": "Custom lighting"}),
                "transition_style": (TRANSITION_STYLE, {"default": "smooth continuous motion", "tooltip": "Clip start/end behavior"}),
                "transition_custom": ("STRING", {"default": "", "tooltip": "Custom transition/loop behavior"}),
                "first_frame": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional opening frame description",
                }),
                "last_frame": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional ending frame description",
                }),
                "additional_details": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Continuity, wardrobe, environment, effects, or model notes",
                }),
                "custom_system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "High-priority guidance for the LLM's video prompt style",
                }),
            },
        }

    def generate_video_prompt(self, ollama_model: dict, concept: str, **kwargs):
        cfg = ollama_model

        def get_val(dropdown_key: str, custom_key: str) -> str:
            dropdown_val = kwargs.get(dropdown_key, "custom")
            custom_val = str(kwargs.get(custom_key, "") or "").strip()
            if dropdown_val == "custom" and custom_val:
                return custom_val
            if dropdown_val != "custom":
                return dropdown_val
            return "custom"

        system = build_system_prompt(
            model_profile=self.MODEL_PROFILE,
            duration=get_val("duration", "duration_custom"),
            aspect_ratio=get_val("aspect_ratio", "aspect_custom"),
            video_style=get_val("video_style", "style_custom"),
            shot_type=get_val("shot_type", "shot_custom"),
            camera_motion=get_val("camera_motion", "camera_custom"),
            subject_motion=get_val("subject_motion", "subject_motion_custom"),
            motion_intensity=get_val(
                "motion_intensity", "motion_intensity_custom"),
            pacing=get_val("pacing", "pacing_custom"),
            lighting=get_val("lighting", "lighting_custom"),
            transition_style=get_val("transition_style", "transition_custom"),
            first_frame=kwargs.get("first_frame", ""),
            last_frame=kwargs.get("last_frame", ""),
            additional_details=kwargs.get("additional_details", ""),
            custom_system_prompt=kwargs.get("custom_system_prompt", ""),
        )
        user_prompt = build_user_prompt(concept)

        _log.info("[OllamaNodes] Generating %s video prompt with %s",
                  self.MODEL_PROFILE, cfg["model"])

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.6),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2048,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning(
                "[OllamaNodes] Failed to parse JSON for %s video prompt", self.MODEL_PROFILE)
            return (raw, "", "", "", "")

        return (
            parsed.get("positive_prompt", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("shot_plan", ""),
            parsed.get("motion_notes", ""),
            parsed.get("model_settings", ""),
        )


class OllamaLTXVVideoPrompt(_BaseVideoPromptGenerator):
    """Generates LTXV 2.3-oriented text-to-video or image-to-video prompts."""

    MODEL_PROFILE = "LTXV 2.3"


class OllamaWanVideoPrompt(_BaseVideoPromptGenerator):
    """Generates Wan 2.2-oriented text-to-video or image-to-video prompts."""

    MODEL_PROFILE = "Wan 2.2"
