"""
OllamaAspectOptimizer — Aspect Ratio Optimizer
Optimizes prompts for different aspect ratios.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.aspect_optimizer import build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

RATIO_OPTIONS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "2:3", "3:2"]
COMPOSITION_OPTIONS = ["subject_centered", "rule_of_thirds", "golden_ratio", "symmetrical", "dynamic"]


class OllamaAspectOptimizer:
    """Optimizes prompts for specific aspect ratios."""

    CATEGORY = "Ollama-Magic-Nodes/Scene"
    FUNCTION = "optimize"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "optimized_prompt", "composition_notes", "cropping_suggestions", "negative_additions",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt optimized for the aspect ratio",
        "Composition adjustments made",
        "Safe zones and cropping info",
        "Additional negatives for this ratio",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Original prompt to optimize",
                }),
                "target_ratio": (RATIO_OPTIONS, {
                    "default": "16:9",
                    "tooltip": "Target aspect ratio",
                }),
                "composition_priority": (COMPOSITION_OPTIONS, {
                    "default": "rule_of_thirds",
                    "tooltip": "Composition style priority",
                }),
            },
        }

    def optimize(self, ollama_model: dict, prompt: str,
                target_ratio: str = "16:9", composition_priority: str = "rule_of_thirds"):
        cfg = ollama_model
        system = build_system_prompt(target_ratio, composition_priority)
        user_prompt = build_user_prompt(prompt)

        _log.info("[OllamaNodes] Optimizing for aspect ratio %s (composition=%s)", target_ratio, composition_priority)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.5),
            num_ctx=cfg.get("num_ctx", 4096),
            num_predict=2000,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for aspect optimization")
            return (raw, "", "", "")

        return (
            parsed.get("optimized_prompt", ""),
            parsed.get("composition_notes", ""),
            parsed.get("cropping_suggestions", ""),
            parsed.get("negative_additions", ""),
        )
