"""
OllamaStyleTransfer — Style Transfer Prompt Generator
Extracts artistic style from an image and applies it to a new subject.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.style_transfer import build_system_prompt, build_user_prompt
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaStyleTransfer:
    """Analyzes an image's style and generates prompts to apply it to a new subject."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "transfer_style"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "styled_prompt", "style_description", "style_tags",
        "color_palette", "technique_notes",
    )
    OUTPUT_TOOLTIPS = (
        "Prompt with source style applied to the target subject",
        "Detailed description of the extracted style",
        "Comma-separated style tags",
        "Color palette from the source style",
        "Technical notes on achieving this style",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE", {"tooltip": "Image whose artistic style to extract"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "target_subject": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What to depict in the extracted style",
                }),
                "style_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly to apply the style (0=subtle, 1=dominant)",
                }),
            },
        }

    def transfer_style(self, style_image, ollama_model: dict,
                       target_subject: str, style_strength: float = 0.7):
        cfg = ollama_model
        img_b64 = tensor_to_base64(style_image)

        system = build_system_prompt(style_strength)
        user_prompt = build_user_prompt(target_subject)

        _log.info("[OllamaNodes] Style transfer with %s (strength=%.2f)", cfg["model"], style_strength)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.5),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2048,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for style transfer")
            return (raw, "", "", "", "")

        return (
            parsed.get("styled_prompt", ""),
            parsed.get("style_description", ""),
            parsed.get("style_tags", ""),
            parsed.get("color_palette", ""),
            parsed.get("technique_notes", ""),
        )
