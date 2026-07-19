"""
OllamaControlNetHelper — Optimize prompts for ControlNet workflows.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.controlnet_helper import (
    CONTROLNET_TYPE, CONTROLNET_STRENGTH,
    build_system_prompt, build_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaControlNetHelper:
    """Optimize prompts for ControlNet workflows."""

    CATEGORY = "Ollama-Magic-Nodes/Utility"
    FUNCTION = "optimize_for_controlnet"
    RETURN_TYPES = ("STRING",) * 9
    RETURN_NAMES = (
        "optimized_prompt", "structural_elements", "detail_elements",
        "style_elements", "what_to_omit", "what_to_emphasize",
        "negative_prompt", "common_mistakes", "workflow_tips",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "controlnet_type": (CONTROLNET_TYPE, {"default": "canny"}),
            },
            "optional": {
                "strength": (CONTROLNET_STRENGTH, {"default": "medium - 0.6-0.7"}),
                "control_description": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Describe the control image",
                }),
            },
        }

    def optimize_for_controlnet(self, ollama_model: dict, prompt: str,
                                controlnet_type: str = "canny", **kwargs):
        cfg = ollama_model
        
        system = build_system_prompt(
            controlnet_type=controlnet_type,
            strength=kwargs.get("strength", "medium - 0.6-0.7"),
        )
        user_prompt = build_user_prompt(prompt, kwargs.get("control_description", ""))
        
        _log.info("[OllamaNodes] Optimizing for ControlNet: %s", controlnet_type)
        
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
            return (raw,) + ("",) * 8

        return (
            parsed.get("optimized_prompt", ""),
            parsed.get("structural_elements", ""),
            parsed.get("detail_elements", ""),
            parsed.get("style_elements", ""),
            parsed.get("what_to_omit", ""),
            parsed.get("what_to_emphasize", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("common_mistakes", ""),
            parsed.get("workflow_tips", ""),
        )
