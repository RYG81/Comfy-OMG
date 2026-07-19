"""
OllamaPoseDescriptor — Pose Descriptor
Extracts detailed pose descriptions from images.
"""
from __future__ import annotations
import json
import logging

from ...ollama_client import generate
from ...prompts.pose_descriptor import build_system_prompt, USER_PROMPT
from ...utils.image_utils import tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


class OllamaPoseDescriptor:
    """Extracts detailed pose/body position descriptions from images."""

    CATEGORY = "Ollama-Magic-Nodes/Character"
    FUNCTION = "describe_pose"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "pose_description", "body_position", "limb_positions",
        "hand_description", "face_expression", "pose_tags",
    )
    OUTPUT_TOOLTIPS = (
        "Natural language pose description",
        "Core body positioning",
        "Detailed limb placement",
        "Hand poses and gestures",
        "Facial expression details",
        "Comma-separated pose tags",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image containing the pose"}),
                "ollama_model": ("OLLAMA_MODEL",),
                "detail_level": (["simple", "detailed", "anatomical"], {
                    "default": "detailed",
                    "tooltip": "How detailed the pose description should be",
                }),
                "include_hands": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extra detail for hand positions",
                }),
                "include_face": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extra detail for facial expression",
                }),
            },
        }

    def describe_pose(self, image, ollama_model: dict, detail_level: str = "detailed",
                     include_hands: bool = True, include_face: bool = True):
        cfg = ollama_model
        img_b64 = tensor_to_base64(image)
        system = build_system_prompt(detail_level, include_hands, include_face)

        _log.info("[OllamaNodes] Describing pose (detail=%s)", detail_level)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=USER_PROMPT,
            system=system,
            temperature=cfg.get("temperature", 0.3),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2048,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON for pose description")
            return (raw, "", "", "", "", "")

        return (
            parsed.get("pose_description", ""),
            parsed.get("body_position", ""),
            parsed.get("limb_positions", ""),
            parsed.get("hand_description", ""),
            parsed.get("face_expression", ""),
            parsed.get("pose_tags", ""),
        )
