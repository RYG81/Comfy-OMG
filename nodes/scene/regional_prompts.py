"""
OllamaRegionalPrompts — Regional Prompt Generator
Divides image into regions and generates prompts for each.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.regional_prompts import build_system_prompt, USER_PROMPT
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

GRID_OPTIONS = ["2x2", "3x3", "2x3", "3x2", "1x3", "3x1"]


class OllamaRegionalPrompts:
    """Generates region-specific prompts for compositional control."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "generate_regions"
    RETURN_TYPES = ("STRING",) * 12
    RETURN_NAMES = (
        "top_left", "top_center", "top_right",
        "middle_left", "center", "middle_right",
        "bottom_left", "bottom_center", "bottom_right",
        "full_composition", "regional_weights", "overlap_notes",
    )
    OUTPUT_TOOLTIPS = tuple([
        "Top-left region", "Top-center region", "Top-right region",
        "Middle-left region", "Center region", "Middle-right region",
        "Bottom-left region", "Bottom-center region", "Bottom-right region",
        "Full composition description", "Suggested attention weights", "Elements spanning regions",
    ])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Reference image to analyze"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "grid_size": (GRID_OPTIONS, {
                    "default": "3x3",
                    "tooltip": "How to divide the image",
                }),
            },
            "optional": {
                "focus_areas": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Specific areas to focus on",
                }),
            },
        }

    def generate_regions(self, image, ollama_model: dict, grid_size: str = "3x3",
                        focus_areas: str = ""):
        cfg = ollama_model
        img_b64 = tensor_to_base64(image)
        system = build_system_prompt(grid_size, focus_areas)

        _log.info("[OllamaNodes] Generating regional prompts (grid=%s)", grid_size)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=USER_PROMPT,
            system=system,
            temperature=cfg.get("temperature", 0.4),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=3000,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for regional prompts")
            return (raw,) + ("",) * 11

        return (
            parsed.get("top_left", ""),
            parsed.get("top_center", ""),
            parsed.get("top_right", ""),
            parsed.get("middle_left", ""),
            parsed.get("center", ""),
            parsed.get("middle_right", ""),
            parsed.get("bottom_left", ""),
            parsed.get("bottom_center", ""),
            parsed.get("bottom_right", ""),
            parsed.get("full_composition", ""),
            parsed.get("regional_weights", ""),
            parsed.get("overlap_notes", ""),
        )
