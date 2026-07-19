"""
OllamaImageSequenceAnalyzer - Analyze up to 4 sequential images for video prompts.
"""
from __future__ import annotations

import json
import logging

from ...ollama_client import generate
from ...prompts.image_analysis import build_system_prompt, build_user_prompt
from ...utils.image_utils import image_tensor_hash, tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


VIDEO_SYSTEM_PROMPT = """You are an expert image-to-video prompt director.

You will receive structured analyses for 1 to 4 images. Treat them as ordered keyframes in a single continuous video sequence: image 1 leads to image 2, then image 3, then image 4.

Create one model-ready positive video prompt that preserves subject identity, wardrobe, setting, lighting direction, style, and camera continuity while describing the temporal progression between the keyframes.

Respond in this EXACT JSON format:
{
    "video_prompt": "One direct positive prompt only. No labels, markdown, negative prompt, explanation, or notes.",
    "video_notes": "Short optional explanation of keyframe progression, camera motion, continuity locks, and timing decisions."
}

Rules:
- The video_prompt value must be usable directly in a video generation prompt box.
- Mention the keyframe progression in order inside the video_prompt.
- Prefer one continuous shot unless the analyses clearly require a transition.
- Include camera movement, subject movement, environment movement, lighting continuity, and temporal pacing.
- Keep all explanation out of video_prompt and place it only in video_notes.
- Output ONLY valid JSON.
"""


def _compact_analysis(parsed) -> str:
    if isinstance(parsed, dict):
        parts = []
        for key in (
            "subject", "body_details", "clothing", "pose", "location", "lighting", "camera",
            "color_palette", "art_style", "mood", "technical",
            "reconstruction_prompt",
        ):
            value = parsed.get(key, "")
            if value:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)
    return str(parsed or "")


class OllamaImageSequenceAnalyzer:
    """Analyzes up to four images one by one and creates a sequential video prompt."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "analyze_sequence"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "full_sequence_analysis",
        "image_1_analysis",
        "image_2_analysis",
        "image_3_analysis",
        "image_4_analysis",
        "video_prompt",
        "video_notes",
    )
    OUTPUT_TOOLTIPS = (
        "Combined JSON analysis for every provided image",
        "Full JSON analysis for image 1",
        "Full JSON analysis for image 2",
        "Full JSON analysis for image 3",
        "Full JSON analysis for image 4",
        "Clean positive video prompt using the images as ordered keyframes",
        "Explanation, shot-plan, and continuity notes for the video prompt",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "ollama_model": ("OLLAMA_MODEL",),
                "detail_level": (["basic", "detailed", "exhaustive"], {
                    "default": "detailed",
                    "tooltip": "How detailed each image analysis should be",
                }),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "custom_categories": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional categories to analyze for every image",
                }),
                "video_direction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional guidance for motion, camera, duration, model, or style",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image_1, ollama_model: dict, detail_level: str = "detailed",
                   image_2=None, image_3=None, image_4=None,
                   custom_categories: str = "", video_direction: str = "", **_):
        cfg = ollama_model or {}
        image_hashes = [
            image_tensor_hash(img) if img is not None else ""
            for img in (image_1, image_2, image_3, image_4)
        ]
        cache_key = {
            "images": image_hashes,
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "temperature": cfg.get("temperature", 0.3),
            "num_ctx": cfg.get("num_ctx", 8192),
            "seed": cfg.get("seed", -1),
            "keep_alive": cfg.get("keep_alive", "5m"),
            "detail_level": detail_level,
            "custom_categories": custom_categories,
            "video_direction": video_direction,
        }
        return json.dumps(cache_key, sort_keys=True, separators=(",", ":"))

    def analyze_sequence(
        self,
        image_1,
        ollama_model: dict,
        detail_level: str = "detailed",
        image_2=None,
        image_3=None,
        image_4=None,
        custom_categories: str = "",
        video_direction: str = "",
    ):
        cfg = ollama_model
        images = [img for img in (image_1, image_2, image_3, image_4) if img is not None]
        if not images:
            empty_analysis = json.dumps({"images": [], "error": "No images provided."}, indent=2)
            return (empty_analysis, "", "", "", "", "", "")

        system = build_system_prompt(detail_level, custom_categories)
        user_prompt = build_user_prompt(detail_level)
        analyses = []

        for index, image in enumerate(images, start=1):
            _log.info(
                "[OllamaNodes] Analyzing sequence image %d/%d with %s",
                index, len(images), cfg["model"],
            )
            raw = generate(
                base_url=cfg["base_url"],
                model=cfg["model"],
                prompt=f"Image {index} of {len(images)}.\n{user_prompt}",
                system=system,
                temperature=min(float(cfg.get("temperature", 0.2)), 0.2),
                num_ctx=cfg.get("num_ctx", 8192),
                num_predict=4096,
                seed=cfg.get("seed", -1),
                keep_alive=cfg.get("keep_alive", "5m"),
                images=[tensor_to_base64(image)],
                response_format="json",
            )

            parsed = extract_json_block(raw)
            if parsed is None:
                _log.warning("[OllamaNodes] Failed to parse JSON for sequence image %d", index)
                parsed = {"raw_output": raw}

            analyses.append({
                "frame": index,
                "analysis": parsed,
            })

        sequence_payload = {
            "frame_count": len(analyses),
            "frames": analyses,
            "video_direction": video_direction.strip(),
        }
        full_analysis = json.dumps(sequence_payload, indent=2)
        frame_outputs = [
            json.dumps(item["analysis"], indent=2) for item in analyses
        ]
        while len(frame_outputs) < 4:
            frame_outputs.append("")

        compact_frames = "\n\n".join(
            f"Keyframe {item['frame']}:\n{_compact_analysis(item['analysis'])}"
            for item in analyses
        )
        direction = video_direction.strip() or "Create a coherent cinematic image-to-video prompt."
        raw_video = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=(
                f"User video direction:\n{direction}\n\n"
                f"Ordered keyframe analyses:\n{compact_frames}\n\n"
                "Return JSON with video_prompt and video_notes."
            ),
            system=VIDEO_SYSTEM_PROMPT,
            temperature=cfg.get("temperature", 0.6),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=2048,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            response_format="json",
        ).strip()

        parsed_video = extract_json_block(raw_video)
        if parsed_video is None:
            video_prompt = raw_video
            video_notes = ""
        else:
            video_prompt = str(parsed_video.get("video_prompt", "") or "").strip()
            video_notes = str(parsed_video.get("video_notes", "") or "").strip()
            if not video_prompt:
                video_prompt = raw_video

        return (full_analysis, *frame_outputs[:4], video_prompt, video_notes)
