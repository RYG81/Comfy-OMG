"""
OllamaLoraSuggester — Suggest LoRAs for a prompt.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.lora_suggester import MODEL_ECOSYSTEM, build_system_prompt, build_user_prompt
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaLoraSuggester:
    """Analyze a prompt and suggest helpful LoRA types."""

    CATEGORY = "Ollama-Magic-Nodes/Utility"
    FUNCTION = "suggest_loras"
    RETURN_TYPES = ("STRING",) * 8
    RETURN_NAMES = (
        "prompt_analysis", "style_loras", "concept_loras", "quality_loras",
        "priority_loras", "lora_keywords", "weight_suggestions", "prompt_optimization",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "ecosystem": (MODEL_ECOSYSTEM, {"default": "SDXL"}),
            },
        }

    def suggest_loras(self, ollama_model: dict, prompt: str, ecosystem: str = "SDXL"):
        cfg = ollama_model
        
        system = build_system_prompt(ecosystem)
        user_prompt = build_user_prompt(prompt)
        
        _log.info("[OllamaNodes] Suggesting LoRAs for %s", ecosystem)
        
        raw = generate(
            base_url=cfg["base_url"], model=cfg["model"],
            prompt=user_prompt, system=system,
            temperature=cfg.get("temperature", 0.5),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2000,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            return (raw,) + ("",) * 7

        # Convert lists to strings
        style_loras = parsed.get("style_loras", [])
        concept_loras = parsed.get("concept_loras", [])
        quality_loras = parsed.get("quality_loras", [])
        
        return (
            parsed.get("prompt_analysis", ""),
            ", ".join(style_loras) if isinstance(style_loras, list) else style_loras,
            ", ".join(concept_loras) if isinstance(concept_loras, list) else concept_loras,
            ", ".join(quality_loras) if isinstance(quality_loras, list) else quality_loras,
            parsed.get("priority_loras", ""),
            parsed.get("lora_keywords", ""),
            parsed.get("weight_suggestions", ""),
            parsed.get("prompt_optimization", ""),
        )
