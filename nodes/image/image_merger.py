"""
OllamaImageMerger — Two Pictures to One
Analyzes two images + user instruction and generates a merge prompt.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.image_merge import BLEND_STYLES, build_system_prompt, build_user_prompt
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaImageMerger:
    """Takes two images and a text instruction, generates a prompt to create a merged composite."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "merge"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "merged_prompt", "image_1_analysis", "image_2_analysis",
        "negative_prompt", "composition_notes",
    )
    OUTPUT_TOOLTIPS = (
        "Ready-to-use prompt for the merged/composite image",
        "Analysis of Image 1 (source)",
        "Analysis of Image 2 (target/base)",
        "Suggested negative prompt",
        "Composition and spatial arrangement notes",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE", {"tooltip": "Source image — take elements FROM this"}),
                "image_2": ("IMAGE", {"tooltip": "Target/base image — merge elements INTO this"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What to take from Image 1 and how to place it in Image 2",
                }),
                "blend_style": (list(BLEND_STYLES.keys()), {
                    "default": "seamless",
                    "tooltip": "How the images should be blended together",
                }),
            },
        }

    def merge(self, image_1, image_2, ollama_model: dict, instruction: str,
              blend_style: str = "seamless"):
        cfg = ollama_model
        img1_b64 = tensor_to_base64(image_1)
        img2_b64 = tensor_to_base64(image_2)

        system = build_system_prompt(blend_style)
        user_prompt = build_user_prompt(instruction)

        _log.info("[OllamaNodes] Merging images with %s (style=%s)", cfg["model"], blend_style)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=cfg.get("temperature", 0.5),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=4096,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img1_b64, img2_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for image merge")
            return (raw, "", "", "", "")

        return (
            parsed.get("merged_prompt", ""),
            parsed.get("image_1_analysis", ""),
            parsed.get("image_2_analysis", ""),
            parsed.get("negative_prompt", ""),
            parsed.get("composition_notes", ""),
        )
