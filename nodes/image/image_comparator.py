"""
OllamaImageComparator — Image Comparator
Compares two images and describes differences/similarities.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.image_comparator import build_system_prompt, USER_PROMPT
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

FOCUS_OPTIONS = ["all", "composition", "style", "subject", "colors", "quality"]


class OllamaImageComparator:
    """Compares two images and identifies similarities/differences."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "compare"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "comparison_summary", "similarities", "differences",
        "image_a_unique", "image_b_unique", "quality_comparison", "recommendation",
    )
    OUTPUT_TOOLTIPS = (
        "Overall comparison summary",
        "What's the same",
        "What's different",
        "Elements only in Image A",
        "Elements only in Image B",
        "Which is better and why",
        "Suggestions for improvement",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "First image"}),
                "image_b": ("IMAGE", {"tooltip": "Second image"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "comparison_focus": (FOCUS_OPTIONS, {
                    "default": "all",
                    "tooltip": "What aspect to focus the comparison on",
                }),
            },
        }

    def compare(self, image_a, image_b, ollama_model: dict,
                comparison_focus: str = "all"):
        cfg = ollama_model
        img_a_b64 = tensor_to_base64(image_a)
        img_b_b64 = tensor_to_base64(image_b)
        system = build_system_prompt(comparison_focus)

        _log.info("[OllamaNodes] Comparing images (focus=%s)", comparison_focus)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=USER_PROMPT,
            system=system,
            temperature=cfg.get("temperature", 0.4),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2048,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_a_b64, img_b_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for comparison")
            return (raw, "", "", "", "", "", "")

        return (
            parsed.get("comparison_summary", ""),
            parsed.get("similarities", ""),
            parsed.get("differences", ""),
            parsed.get("image_a_unique", ""),
            parsed.get("image_b_unique", ""),
            parsed.get("quality_comparison", ""),
            parsed.get("recommendation", ""),
        )
