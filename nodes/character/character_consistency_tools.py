"""
Character continuity and sheet utility nodes.
"""
from __future__ import annotations
import logging

from ...ollama_client import generate
from ...prompts.character_consistency import (
    ANCHOR_MODES,
    EXPRESSION_SETS,
    OUTFIT_THEMES,
    POSE_SETS,
    SHEET_STYLES,
    build_anchor_system_prompt,
    build_anchor_user_prompt,
    build_continuity_check_system_prompt,
    build_continuity_check_user_prompt,
    build_expression_sheet_system_prompt,
    build_expression_sheet_user_prompt,
    build_outfit_sheet_system_prompt,
    build_outfit_sheet_user_prompt,
    build_pose_sheet_system_prompt,
    build_pose_sheet_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


def _run_json(cfg: dict, user_prompt: str, system: str, num_predict: int = 3000) -> dict | str:
    raw = generate(
        base_url=cfg["base_url"],
        model=cfg["model"],
        prompt=user_prompt,
        system=system,
        temperature=cfg.get("temperature", 0.55),
        num_ctx=cfg.get("num_ctx", 8192),
        num_predict=num_predict,
        seed=cfg.get("seed", -1),
        keep_alive=cfg.get("keep_alive", "5m"),
        response_format="json",
    )
    return extract_json_block(raw) or raw


class OllamaCharacterAnchorExtractor:
    """Extracts compact reusable identity anchors from character text."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "extract_anchors"
    RETURN_TYPES = ("STRING",) * 7
    RETURN_NAMES = (
        "identity_anchors", "face_anchors", "body_anchors", "wardrobe_anchors",
        "style_anchors", "negative_prompt", "compact_anchor_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "source_text": ("STRING", {"default": "", "multiline": True}),
                "mode": (ANCHOR_MODES, {"default": "strict identity lock"}),
            },
        }

    def extract_anchors(self, ollama_model: dict, source_text: str, mode: str = "strict identity lock"):
        _log.info("[OllamaNodes] Extracting character anchors")
        parsed = _run_json(ollama_model, build_anchor_user_prompt(source_text), build_anchor_system_prompt(mode))
        if not isinstance(parsed, dict):
            return (parsed,) + ("",) * 6
        return (
            parsed.get("identity_anchors", ""),
            parsed.get("face_anchors", ""),
            parsed.get("body_anchors", ""),
            parsed.get("wardrobe_anchors", ""),
            parsed.get("style_anchors", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("compact_anchor_prompt", ""),
        )


class OllamaOutfitSheetGenerator:
    """Creates same-character outfit sheet prompts."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "generate_outfit_sheet"
    RETURN_TYPES = ("STRING",) * 5
    RETURN_NAMES = (
        "outfit_sheet_prompt", "outfit_breakdown", "material_color_notes",
        "continuity_anchors", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "character_text": ("STRING", {"default": "", "multiline": True}),
                "outfit_theme": (OUTFIT_THEMES, {"default": "casual"}),
                "outfit_count": ("INT", {"default": 6, "min": 2, "max": 12, "step": 1}),
            },
            "optional": {
                "sheet_style": (SHEET_STYLES, {"default": "clean production sheet"}),
                "keep_palette": ("BOOLEAN", {"default": False}),
                "include_construction_notes": ("BOOLEAN", {"default": True}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def generate_outfit_sheet(
        self, ollama_model: dict, character_text: str, outfit_theme: str = "casual",
        outfit_count: int = 6, sheet_style: str = "clean production sheet",
        keep_palette: bool = False, include_construction_notes: bool = True,
        custom_system_prompt: str = "",
    ):
        _log.info("[OllamaNodes] Generating outfit sheet")
        system = build_outfit_sheet_system_prompt(
            outfit_theme, outfit_count, sheet_style, keep_palette,
            include_construction_notes, custom_system_prompt,
        )
        parsed = _run_json(ollama_model, build_outfit_sheet_user_prompt(character_text), system)
        if not isinstance(parsed, dict):
            return (parsed,) + ("",) * 4
        return (
            parsed.get("outfit_sheet_prompt", ""),
            parsed.get("outfit_breakdown", ""),
            parsed.get("material_color_notes", ""),
            parsed.get("continuity_anchors", ""),
            parsed.get("negative_prompt", ""),
        )


class OllamaPoseSheetGenerator:
    """Creates same-character pose sheet prompts."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "generate_pose_sheet"
    RETURN_TYPES = ("STRING",) * 5
    RETURN_NAMES = (
        "pose_sheet_prompt", "pose_breakdown", "silhouette_notes",
        "continuity_anchors", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "character_text": ("STRING", {"default": "", "multiline": True}),
                "pose_set": (POSE_SETS, {"default": "basic turnaround poses"}),
                "pose_count": ("INT", {"default": 6, "min": 2, "max": 12, "step": 1}),
            },
            "optional": {
                "sheet_style": (SHEET_STYLES, {"default": "clean production sheet"}),
                "include_silhouette_notes": ("BOOLEAN", {"default": True}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def generate_pose_sheet(
        self, ollama_model: dict, character_text: str,
        pose_set: str = "basic turnaround poses", pose_count: int = 6,
        sheet_style: str = "clean production sheet",
        include_silhouette_notes: bool = True, custom_system_prompt: str = "",
    ):
        _log.info("[OllamaNodes] Generating pose sheet")
        system = build_pose_sheet_system_prompt(
            pose_set, pose_count, sheet_style, include_silhouette_notes, custom_system_prompt,
        )
        parsed = _run_json(ollama_model, build_pose_sheet_user_prompt(character_text), system)
        if not isinstance(parsed, dict):
            return (parsed,) + ("",) * 4
        return (
            parsed.get("pose_sheet_prompt", ""),
            parsed.get("pose_breakdown", ""),
            parsed.get("silhouette_notes", ""),
            parsed.get("continuity_anchors", ""),
            parsed.get("negative_prompt", ""),
        )


class OllamaExpressionSheetGenerator:
    """Creates same-character expression sheet prompts."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "generate_expression_sheet"
    RETURN_TYPES = ("STRING",) * 5
    RETURN_NAMES = (
        "expression_sheet_prompt", "expression_breakdown", "face_continuity_anchors",
        "acting_notes", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "character_text": ("STRING", {"default": "", "multiline": True}),
                "expression_set": (EXPRESSION_SETS, {"default": "core emotions"}),
                "expression_count": ("INT", {"default": 8, "min": 2, "max": 16, "step": 1}),
            },
            "optional": {
                "sheet_style": (SHEET_STYLES, {"default": "clean production sheet"}),
                "include_acting_notes": ("BOOLEAN", {"default": True}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def generate_expression_sheet(
        self, ollama_model: dict, character_text: str,
        expression_set: str = "core emotions", expression_count: int = 8,
        sheet_style: str = "clean production sheet",
        include_acting_notes: bool = True, custom_system_prompt: str = "",
    ):
        _log.info("[OllamaNodes] Generating expression sheet")
        system = build_expression_sheet_system_prompt(
            expression_set, expression_count, sheet_style, include_acting_notes, custom_system_prompt,
        )
        parsed = _run_json(ollama_model, build_expression_sheet_user_prompt(character_text), system)
        if not isinstance(parsed, dict):
            return (parsed,) + ("",) * 4
        return (
            parsed.get("expression_sheet_prompt", ""),
            parsed.get("expression_breakdown", ""),
            parsed.get("face_continuity_anchors", ""),
            parsed.get("acting_notes", ""),
            parsed.get("negative_prompt", ""),
        )


class OllamaPromptContinuityChecker:
    """Compares prompts and fixes identity/style drift."""

    CATEGORY = "Ollama-Magic-Nodes/Utility"
    FUNCTION = "check_continuity"
    RETURN_TYPES = ("STRING",) * 7
    RETURN_NAMES = (
        "continuity_score", "stable_anchors", "drift_risks", "conflicts",
        "missing_anchors", "fixed_master_prompt", "negative_prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt_a": ("STRING", {"default": "", "multiline": True}),
                "prompt_b": ("STRING", {"default": "", "multiline": True}),
                "check_mode": ([
                    "same character", "same scene", "same style", "character sheet views",
                    "video continuity", "general prompt set",
                ], {"default": "same character"}),
            },
            "optional": {
                "prompt_c": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def check_continuity(
        self, ollama_model: dict, prompt_a: str, prompt_b: str,
        check_mode: str = "same character", prompt_c: str = "",
    ):
        _log.info("[OllamaNodes] Checking prompt continuity")
        parsed = _run_json(
            ollama_model,
            build_continuity_check_user_prompt(prompt_a, prompt_b, prompt_c),
            build_continuity_check_system_prompt(check_mode),
        )
        if not isinstance(parsed, dict):
            return (parsed,) + ("",) * 6
        return (
            parsed.get("continuity_score", ""),
            parsed.get("stable_anchors", ""),
            parsed.get("drift_risks", ""),
            parsed.get("conflicts", ""),
            parsed.get("missing_anchors", ""),
            parsed.get("fixed_master_prompt", ""),
            parsed.get("negative_prompt", ""),
        )
