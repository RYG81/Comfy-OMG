"""
OllamaStoryboardGenerator — Storyboard Generator
Creates a sequence of prompts for visual storytelling.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.storyboard import build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

STYLE_OPTIONS = ["comic", "manga", "cinematic", "animation", "storyboard"]
RATIO_OPTIONS = ["1:1", "16:9", "9:16", "4:3", "3:4"]


class OllamaStoryboardGenerator:
    """Creates sequences of prompts for comics/storyboards."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "generate_storyboard"
    RETURN_TYPES = ("STRING",) * 16
    RETURN_NAMES = (
        "panel_1", "panel_2", "panel_3", "panel_4", "panel_5", "panel_6",
        "panel_7", "panel_8", "panel_9", "panel_10", "panel_11", "panel_12",
        "all_panels", "character_reference", "story_summary", "camera_directions",
    )
    OUTPUT_TOOLTIPS = tuple([f"Panel {i} prompt" for i in range(1, 13)] + [
        "All panels as JSON array",
        "Character descriptions for consistency",
        "Narrative summary",
        "Camera movement notes",
    ])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "story_concept": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The story or scene sequence to visualize",
                }),
                "num_panels": ("INT", {
                    "default": 6,
                    "min": 2, "max": 12, "step": 1,
                    "tooltip": "Number of panels to generate",
                }),
                "style": (STYLE_OPTIONS, {
                    "default": "cinematic",
                    "tooltip": "Visual style for the panels",
                }),
                "maintain_characters": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep character consistency across panels",
                }),
                "aspect_ratio": (RATIO_OPTIONS, {
                    "default": "16:9",
                    "tooltip": "Aspect ratio for panels",
                }),
            },
        }

    def generate_storyboard(self, ollama_model: dict, story_concept: str,
                           num_panels: int = 6, style: str = "cinematic",
                           maintain_characters: bool = True, aspect_ratio: str = "16:9"):
        cfg = ollama_model
        system = build_system_prompt(story_concept, num_panels, style, maintain_characters, aspect_ratio)
        user_prompt = build_user_prompt()

        _log.info("[OllamaNodes] Generating storyboard (%d panels, style=%s)", num_panels, style)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.7),
            num_ctx=cfg.get("num_ctx", 16384),
            num_predict=6000,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for storyboard")
            return (raw,) + ("",) * 15

        panels = []
        for i in range(1, 13):
            panels.append(parsed.get(f"panel_{i}", ""))

        return tuple(panels) + (
            parsed.get("all_panels", ""),
            parsed.get("character_reference", ""),
            parsed.get("story_summary", ""),
            parsed.get("camera_directions", ""),
        )
