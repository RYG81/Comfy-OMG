"""
OllamaImageEdit converts analyzer output and user changes into an edit-model prompt.
"""
from __future__ import annotations

import json
import logging

from ...ollama_client import generate
from ...prompts.image_edit import (
    IMAGE_EDIT_MODELS,
    build_system_prompt,
    build_user_prompt,
)
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


def _as_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


class OllamaImageEdit:
    """Builds model-specific image-edit prompts from structured image analysis."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "build_edit_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "edit_prompt",
        "preservation_notes",
        "change_summary",
        "full_result",
    )
    OUTPUT_TOOLTIPS = (
        "Model-ready image editing prompt",
        "Source-image details that should remain unchanged",
        "Summary of requested changes",
        "Complete structured result as JSON",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analyzed_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Connect full_analysis from Ollama Image Analyzer",
                }),
                "ollama_model": ("OLLAMA_MODEL",),
                "requested_changes": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Describe only what should change in the source image",
                }),
                "image_edit_model": (IMAGE_EDIT_MODELS, {
                    "default": "FLUX 2",
                    "tooltip": "Target image-edit model used to structure the final prompt",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        analyzed_data: str,
        ollama_model: dict,
        requested_changes: str,
        image_edit_model: str = "FLUX 2",
    ):
        cfg = ollama_model or {}
        cache_key = {
            "analyzed_data": analyzed_data,
            "requested_changes": requested_changes,
            "image_edit_model": image_edit_model,
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "temperature": cfg.get("temperature", 0.3),
            "num_ctx": cfg.get("num_ctx", 8192),
            "seed": cfg.get("seed", -1),
        }
        return json.dumps(cache_key, sort_keys=True, separators=(",", ":"))

    def build_edit_prompt(
        self,
        analyzed_data: str,
        ollama_model: dict,
        requested_changes: str,
        image_edit_model: str = "FLUX 2",
    ):
        analysis = analyzed_data.strip()
        changes = requested_changes.strip()
        if not analysis:
            raise ValueError("analyzed_data is empty. Connect full_analysis from the analyzer.")
        if not changes:
            raise ValueError("requested_changes is empty. Describe what should change.")

        cfg = ollama_model
        _log.info(
            "[OllamaNodes] Building image edit prompt with %s for %s",
            cfg["model"],
            image_edit_model,
        )

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=build_user_prompt(analysis, changes, image_edit_model),
            system=build_system_prompt(image_edit_model),
            temperature=min(cfg.get("temperature", 0.3), 0.5),
            top_p=cfg.get("top_p", 0.9),
            top_k=cfg.get("top_k", 40),
            repeat_penalty=cfg.get("repeat_penalty", 1.1),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2048,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if not isinstance(parsed, dict):
            _log.warning("[OllamaNodes] Failed to parse image edit JSON")
            return (raw.strip(), "", changes, raw)

        result = {
            "edit_prompt": _as_string(parsed.get("edit_prompt")),
            "preservation_notes": _as_string(parsed.get("preservation_notes")),
            "change_summary": _as_string(parsed.get("change_summary")) or changes,
        }
        return (
            result["edit_prompt"],
            result["preservation_notes"],
            result["change_summary"],
            json.dumps(result, indent=2, ensure_ascii=False),
        )
