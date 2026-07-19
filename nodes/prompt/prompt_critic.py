"""
OllamaPromptCritic — Prompt Critic
Analyzes prompts and provides feedback and improvements.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.prompt_critic import build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

TARGET_MODELS = ["SDXL", "SD1.5", "Flux", "Midjourney", "DALL-E"]


class OllamaPromptCritic:
    """Analyzes prompts and provides detailed feedback."""

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "critique"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "overall_score", "clarity_feedback", "detail_feedback", "consistency_feedback",
        "improvement_suggestions", "improved_prompt", "missing_elements", "redundant_elements",
    )
    OUTPUT_TOOLTIPS = (
        "Score out of 10 with reasoning",
        "Clarity feedback",
        "Detail level feedback",
        "Consistency/contradiction check",
        "Specific improvement suggestions",
        "Rewritten improved version",
        "What's missing",
        "What's unnecessary",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Prompt to analyze",
                }),
                "target_model": (TARGET_MODELS, {
                    "default": "SDXL",
                    "tooltip": "Target AI model",
                }),
            },
            "optional": {
                "intended_result": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What you're trying to achieve",
                }),
            },
        }

    def critique(self, ollama_model: dict, prompt: str,
                target_model: str = "SDXL", intended_result: str = ""):
        cfg = ollama_model
        system = build_system_prompt(target_model, intended_result)
        user_prompt = build_user_prompt(prompt)

        _log.info("[OllamaNodes] Critiquing prompt (target=%s)", target_model)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.5),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2500,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for prompt critique")
            return (raw,) + ("",) * 7

        return (
            parsed.get("overall_score", ""),
            parsed.get("clarity_feedback", ""),
            parsed.get("detail_feedback", ""),
            parsed.get("consistency_feedback", ""),
            parsed.get("improvement_suggestions", ""),
            parsed.get("improved_prompt", ""),
            parsed.get("missing_elements", ""),
            parsed.get("redundant_elements", ""),
        )
