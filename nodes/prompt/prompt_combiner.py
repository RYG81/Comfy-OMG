"""
OllamaPromptCombiner — Intelligently blend multiple prompts.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.prompt_combiner import BLEND_MODES, build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaPromptCombiner:
    """Intelligently combine multiple prompts into one cohesive prompt."""

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "combine"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "combined_prompt", "prompt_analysis", "conflict_resolution",
        "element_breakdown", "negative_prompt",
    )
    OUTPUT_TOOLTIPS = (
        "Final combined prompt",
        "Analysis of what each input contributed",
        "How conflicts were resolved",
        "Which elements came from which prompt",
        "Negative prompt for the result",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt_1": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "First prompt to combine",
                }),
                "blend_mode": (list(BLEND_MODES.keys()), {
                    "default": "merge",
                    "tooltip": "How to combine the prompts",
                }),
            },
            "optional": {
                "prompt_2": ("STRING", {"default": "", "multiline": True, "tooltip": "Second prompt"}),
                "prompt_3": ("STRING", {"default": "", "multiline": True, "tooltip": "Third prompt"}),
                "prompt_4": ("STRING", {"default": "", "multiline": True, "tooltip": "Fourth prompt"}),
                "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "weight_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
        }

    def combine(self, ollama_model: dict, prompt_1: str, blend_mode: str = "merge",
                prompt_2: str = "", prompt_3: str = "", prompt_4: str = "",
                weight_1: float = 1.0, weight_2: float = 1.0,
                weight_3: float = 1.0, weight_4: float = 1.0):
        cfg = ollama_model
        
        # Collect non-empty prompts
        prompts = []
        weights = []
        for p, w in [(prompt_1, weight_1), (prompt_2, weight_2), 
                     (prompt_3, weight_3), (prompt_4, weight_4)]:
            if p.strip():
                prompts.append(p.strip())
                weights.append(w)
        
        if len(prompts) < 1:
            return ("", "", "", "", "")
        
        if len(prompts) == 1:
            return (prompts[0], "Single prompt provided", "", "", "")
        
        weights_str = ", ".join([f"Prompt {i+1}: {w:.1f}" for i, w in enumerate(weights)])
        system = build_system_prompt(blend_mode, len(prompts), weights_str)
        user_prompt = build_user_prompt(prompts, weights)

        _log.info("[OllamaNodes] Combining %d prompts (mode=%s)", len(prompts), blend_mode)

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
            return (raw, "", "", "", "")

        return (
            parsed.get("combined_prompt", ""),
            parsed.get("prompt_analysis", ""),
            parsed.get("conflict_resolution", ""),
            parsed.get("element_breakdown", ""),
            parsed.get("negative_prompt", ""),
        )
