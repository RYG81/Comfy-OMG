"""
Video shot director node.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.video_shot_director import (
    SEQUENCE_STYLE,
    SHOT_COUNTS,
    VIDEO_MODELS,
    build_system_prompt,
    build_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaVideoShotDirector:
    """Expands a scene/video concept into a shot-by-shot prompt sequence."""

    CATEGORY = "Ollama-Magic-Nodes/Video"
    FUNCTION = "direct_video_shots"
    RETURN_TYPES = ("STRING",) * 6
    RETURN_NAMES = (
        "sequence_prompt", "shot_list", "per_shot_prompts",
        "continuity_notes", "negative_prompt", "model_notes",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "scene_prompt": ("STRING", {"default": "", "multiline": True}),
                "target_model": (VIDEO_MODELS, {"default": "Wan 2.2"}),
                "shot_count": (SHOT_COUNTS, {"default": "5 shots"}),
            },
            "optional": {
                "sequence_style": (SEQUENCE_STYLE, {"default": "single scene coverage"}),
                "total_duration": ("STRING", {"default": "10 seconds"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def direct_video_shots(
        self, ollama_model: dict, scene_prompt: str,
        target_model: str = "Wan 2.2", shot_count: str = "5 shots",
        sequence_style: str = "single scene coverage",
        total_duration: str = "10 seconds", custom_system_prompt: str = "",
    ):
        cfg = ollama_model
        system = build_system_prompt(
            target_model, shot_count, sequence_style, total_duration, custom_system_prompt,
        )
        user_prompt = build_user_prompt(scene_prompt)

        _log.info("[OllamaNodes] Directing video shot sequence")

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.6),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=3500,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            return (raw,) + ("",) * 5

        return (
            parsed.get("sequence_prompt", ""),
            parsed.get("shot_list", ""),
            parsed.get("per_shot_prompts", ""),
            parsed.get("continuity_notes", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("model_notes", ""),
        )
