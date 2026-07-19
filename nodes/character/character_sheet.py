"""
OllamaCharacterSheet — Character Sheet Creator
Generates structured prompts for character reference sheets.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.character_sheet import (
    BACKGROUND_STYLES,
    CONSISTENCY_LEVELS,
    EXPRESSION_SETS,
    MODEL_TARGETS,
    POSE_STANDARDS,
    SHEET_FORMATS,
    STYLES,
    build_system_prompt,
    build_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaCharacterSheet:
    """Creates multi-view character sheet prompts from a text description."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "create_sheet"
    RETURN_TYPES = ("STRING",) * 12
    RETURN_NAMES = (
        "character_summary",
        "full_body_front", "full_body_side", "full_body_back", "full_body_3quarter",
        "headshot_front", "headshot_3quarter", "headshot_profile",
        "expression_sheet",
        "full_body_grid", "face_grid", "master_sheet",
    )
    OUTPUT_TOOLTIPS = (
        "Summary of the finalized character design",
        "Full body front view prompt",
        "Full body side view prompt",
        "Full body back view prompt",
        "Full body 3/4 view prompt",
        "Head/face front view prompt",
        "Head/face 3/4 view prompt",
        "Head/face profile view prompt",
        "Expression variations grid prompt",
        "Full body turnaround grid prompt",
        "Face reference grid prompt",
        "Complete master character sheet prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "character_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Describe your character — appearance, personality, key features",
                }),
                "style": (STYLES, {
                    "default": "Anime/Manga",
                    "tooltip": "Art style for the character sheet",
                }),
                "num_expressions": ("INT", {
                    "default": 6,
                    "min": 2, "max": 12, "step": 1,
                    "tooltip": "Number of expression variations to generate",
                }),
            },
            "optional": {
                "sheet_format": (SHEET_FORMATS, {
                    "default": "production turnaround sheet",
                    "tooltip": "Reference-sheet layout/use case",
                }),
                "pose_standard": (POSE_STANDARDS, {
                    "default": "relaxed neutral stance",
                    "tooltip": "Pose style for turnaround views",
                }),
                "background_style": (BACKGROUND_STYLES, {
                    "default": "clean white background",
                    "tooltip": "Sheet background/layout treatment",
                }),
                "consistency_level": (CONSISTENCY_LEVELS, {
                    "default": "strict - exact outfit and proportions",
                    "tooltip": "How aggressively to lock design continuity",
                }),
                "model_target": (MODEL_TARGETS, {
                    "default": "general image models",
                    "tooltip": "Prompt phrasing target",
                }),
                "expression_set": (EXPRESSION_SETS, {
                    "default": "core emotions",
                    "tooltip": "Expression set style",
                }),
                "include_callouts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ask for material/color/accessory callouts in the master sheet",
                }),
                "custom_system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "High-priority guidance for sheet style, continuity, or formatting",
                }),
            },
        }

    def create_sheet(self, ollama_model: dict, character_description: str,
                     style: str = "Anime/Manga", num_expressions: int = 6,
                     sheet_format: str = "production turnaround sheet",
                     pose_standard: str = "relaxed neutral stance",
                     background_style: str = "clean white background",
                     consistency_level: str = "strict - exact outfit and proportions",
                     model_target: str = "general image models",
                     expression_set: str = "core emotions",
                     include_callouts: bool = True,
                     custom_system_prompt: str = ""):
        cfg = ollama_model
        system = build_system_prompt(
            style=style,
            num_expressions=num_expressions,
            sheet_format=sheet_format,
            pose_standard=pose_standard,
            background_style=background_style,
            consistency_level=consistency_level,
            model_target=model_target,
            expression_set=expression_set,
            include_callouts=include_callouts,
            custom_system_prompt=custom_system_prompt,
        )
        user_prompt = build_user_prompt(character_description)

        _log.info("[OllamaNodes] Creating character sheet with %s (style=%s)", cfg["model"], style)

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
            _log.warning("[OllamaNodes] Failed to parse JSON for character sheet")
            return (raw,) + ("",) * 11

        return (
            parsed.get("character_summary", ""),
            parsed.get("full_body_front", ""),
            parsed.get("full_body_side", ""),
            parsed.get("full_body_back", ""),
            parsed.get("full_body_3quarter", ""),
            parsed.get("headshot_front", ""),
            parsed.get("headshot_3quarter", ""),
            parsed.get("headshot_profile", ""),
            parsed.get("expression_sheet", ""),
            parsed.get("full_body_grid", ""),
            parsed.get("face_grid", ""),
            parsed.get("master_sheet", ""),
        )
