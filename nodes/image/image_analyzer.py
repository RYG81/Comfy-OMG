"""
OllamaImageAnalyzer - Image Composition Extractor
Analyzes an image and extracts exhaustive details in structured categories.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ...ollama_client import generate
from ...prompts.image_analysis import (
    IMAGE_GENERATION_MODELS,
    REQUIRED_KEYS,
    SYSTEM_PROMPT_PRESETS,
    build_system_prompt,
    build_user_prompt,
)
from ...utils.image_utils import image_tensor_hash, tensor_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)


BBOX_MODES = ("opencv_foreground", "model_estimated", "full_image")


def _normalize_analysis(parsed: dict) -> dict:
    normalized = {}
    for key in REQUIRED_KEYS:
        value = parsed.get(key, "")
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        normalized[key] = value
    return normalized


def _image_dimensions(image) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        return 1000, 1000
    if len(shape) == 4:
        return int(shape[2]), int(shape[1])
    return int(shape[1]), int(shape[0])


def _normalize_bbox(bbox, width: int, height: int) -> list[int]:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return [0, 0, 1000, 1000]

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * 1000.0, x2 * 1000.0
        y1, y2 = y1 * 1000.0, y2 * 1000.0
    elif (
        max(abs(x1), abs(x2)) <= width
        and max(abs(y1), abs(y2)) <= height
        and (width != 1000 or height != 1000)
    ):
        x1, x2 = x1 / max(width, 1) * 1000.0, x2 / max(width, 1) * 1000.0
        y1, y2 = y1 / max(height, 1) * 1000.0, y2 / max(height, 1) * 1000.0

    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 1000, 1000]
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _foreground_bbox_1000(image) -> list[int]:
    try:
        frame = image[0] if len(image.shape) == 4 else image
        if hasattr(frame, "detach"):
            frame = frame.detach()
        if hasattr(frame, "cpu"):
            frame = frame.cpu()
        arr = frame.numpy() if hasattr(frame, "numpy") else frame

        import numpy as np

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return [0, 0, 1000, 1000]
        arr = arr[..., :3]
        h, w = arr.shape[:2]

        border = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
        )
        bg = np.median(border, axis=0)
        diff = np.linalg.norm(arr - bg, axis=2)
        threshold = max(0.08, float(np.percentile(diff, 82)))
        mask = diff > threshold

        try:
            import cv2

            mask_u8 = (mask.astype(np.uint8) * 255)
            kernel = np.ones((5, 5), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > (w * h * 0.01):
                    x, y, bw, bh = cv2.boundingRect(contour)
                    pad_x = int(w * 0.02)
                    pad_y = int(h * 0.02)
                    return _normalize_bbox(
                        [x - pad_x, y - pad_y, x + bw + pad_x, y + bh + pad_y], w, h
                    )
        except Exception:
            pass

        ys, xs = np.where(mask)
        if len(xs) < w * h * 0.01:
            return [0, 0, 1000, 1000]
        pad_x = int(w * 0.02)
        pad_y = int(h * 0.02)
        return _normalize_bbox(
            [xs.min() - pad_x, ys.min() - pad_y, xs.max() + pad_x, ys.max() + pad_y],
            w,
            h,
        )
    except Exception as exc:
        _log.debug("[OllamaNodes] Foreground bbox failed: %s", exc)
        return [0, 0, 1000, 1000]


def _parse_scene_elements(value: Any, width: int, height: int) -> list[dict]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    if not isinstance(value, list):
        return []

    elements = []
    for item in value:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("desc", "") or "").strip()
        if not desc:
            continue
        elem_type = str(item.get("type", "obj") or "obj").strip() or "obj"
        elements.append({
            "type": elem_type,
            "bbox": _normalize_bbox(item.get("bbox", [0, 0, 1000, 1000]), width, height),
            "desc": desc,
        })
    return elements


def _clean_phrase(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.replace("\n", " ").split())
    return text.strip(" ,.;:")


def _join_clause(parts: list[str]) -> str:
    return ", ".join(part for part in (_clean_phrase(p) for p in parts) if part)


def _scene_element_summary(parsed: dict) -> str:
    elements = _parse_scene_elements(parsed.get("scene_elements", ""), 1000, 1000)
    summaries = []
    for item in elements[:4]:
        desc = _clean_phrase(item.get("desc", ""))
        elem_type = _clean_phrase(item.get("type", ""))
        if desc and elem_type and elem_type not in desc.lower():
            summaries.append(f"{elem_type}: {desc}")
        elif desc:
            summaries.append(desc)
    return "; ".join(summaries)


def _prompt_is_weak(prompt: str) -> bool:
    text = _clean_phrase(prompt)
    if len(text) < 120:
        return True
    lowered = text.lower()
    bad_markers = (
        '"subject"', '"clothing"', '"pose"', "subject:", "clothing:", "pose:",
        "body_details", "scene_elements", "create ", " situated ", "compose the image",
        " with the scene ", " and finish it ", "use a neutral palette",
    )
    return any(marker in lowered for marker in bad_markers)


def _build_reconstruction_prompt(parsed: dict, image_generation_model: str) -> str:
    subject_cluster = _join_clause([
        parsed.get("subject", ""),
        parsed.get("body_details", ""),
        parsed.get("clothing", ""),
    ])
    pose = _clean_phrase(parsed.get("pose", ""))
    setting = _join_clause([
        parsed.get("location", ""),
        _scene_element_summary(parsed),
    ])
    camera = _clean_phrase(parsed.get("camera", ""))
    lighting = _clean_phrase(parsed.get("lighting", ""))
    look = _join_clause([
        parsed.get("color_palette", ""),
        parsed.get("art_style", ""),
        parsed.get("mood", ""),
        parsed.get("technical", ""),
    ])

    sentences = []
    if subject_cluster:
        first = subject_cluster
        if pose:
            first = f"{first}, {pose}"
        sentences.append(first[0].upper() + first[1:] if first else first)
    elif pose:
        sentences.append(pose[0].upper() + pose[1:])

    if setting:
        sentences.append(f"Set in {setting}.")
    if camera:
        sentences.append(f"Framed with {camera}.")
    if lighting:
        sentences.append(f"Lit by {lighting}.")
    if look:
        sentences.append(f"Rendered with {look}.")

    prompt = " ".join(sentence.rstrip(" .") + "." for sentence in sentences if sentence)
    if image_generation_model == "FLUX":
        return prompt
    if image_generation_model in {"Stable Diffusion XL", "Stable Diffusion 1.5"}:
        return prompt.replace(". ", ", ").rstrip(".")
    return prompt


def _select_reconstruction_prompt(parsed: dict, image_generation_model: str) -> str:
    model_prompt = _clean_phrase(parsed.get("reconstruction_prompt", ""))
    if model_prompt and not _prompt_is_weak(model_prompt):
        return model_prompt
    return _build_reconstruction_prompt(parsed, image_generation_model)

def _build_scene_graph(parsed: dict, image, bbox_mode: str) -> str:
    width, height = _image_dimensions(image)
    bbox_mode = bbox_mode if bbox_mode in BBOX_MODES else "opencv_foreground"

    high_level = str(parsed.get("reconstruction_prompt", "") or "").strip()
    if not high_level:
        high_level = _build_reconstruction_prompt(parsed, "General / Auto")

    elements = []
    if bbox_mode == "model_estimated":
        elements = _parse_scene_elements(parsed.get("scene_elements", ""), width, height)

    if not elements:
        bbox = [0, 0, 1000, 1000] if bbox_mode == "full_image" else _foreground_bbox_1000(image)
        desc_parts = [
            str(parsed.get("subject", "") or "").strip(),
            str(parsed.get("body_details", "") or "").strip(),
            str(parsed.get("clothing", "") or "").strip(),
            str(parsed.get("pose", "") or "").strip(),
        ]
        desc = ", ".join(part for part in desc_parts if part)
        elements = [{"type": "main_subject", "bbox": bbox, "desc": desc or high_level}]

    payload = {
        "high_level_description": high_level,
        "style_description": {
            "aesthetics": str(parsed.get("mood", "") or "").strip(),
            "lighting": str(parsed.get("lighting", "") or "").strip(),
            "photo": str(parsed.get("camera", "") or "").strip(),
            "medium": str(parsed.get("art_style", "") or "").strip(),
        },
        "compositional_deconstruction": {
            "background": str(parsed.get("location", "") or "").strip(),
            "bbox_mode": bbox_mode,
            "bbox_coordinate_space": "normalized_1000x1000_xyxy",
            "elements": elements,
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


class OllamaImageAnalyzer:
    """Extracts detailed composition information from an image using Ollama vision models."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "analyze"
    RETURN_TYPES = ("STRING",) * 12
    RETURN_NAMES = (
        "full_analysis", "subject", "clothing", "pose", "location",
        "lighting", "camera", "color_palette", "art_style", "mood",
        "reconstruction_prompt", "scene_graph_json",
    )
    OUTPUT_TOOLTIPS = (
        "Complete structured analysis as JSON",
        "Subject description",
        "Clothing/attire details",
        "Pose/action description",
        "Location/setting",
        "Lighting analysis",
        "Camera/composition",
        "Color palette information",
        "Art style details",
        "Mood/atmosphere",
        "Ready-to-use reconstruction prompt",
        "Scene graph JSON with high-level description, style, background, elements, and bboxes",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ollama_model": ("OLLAMA_MODEL",),
                "detail_level": (["basic", "detailed", "exhaustive"], {
                    "default": "detailed",
                    "tooltip": "How detailed the analysis should be",
                }),
                "image_generation_model": (IMAGE_GENERATION_MODELS, {
                    "default": "General / Auto",
                    "tooltip": "Structures the reconstruction prompt for the selected image generator",
                }),
                "system_prompt_preset": (list(SYSTEM_PROMPT_PRESETS), {
                    "default": "Claude",
                    "tooltip": "Claude (forensic), GPT (high-fidelity reconstruction), or Grok (adult-content-aware)",
                }),
            },
            "optional": {
                "custom_categories": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional categories to analyze (comma-separated)",
                }),
                "bbox_mode": (list(BBOX_MODES), {
                    "default": "opencv_foreground",
                    "tooltip": "BBox source for scene_graph_json. OpenCV is lightweight/approximate; model_estimated uses the vision model; full_image uses the whole canvas.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image, ollama_model: dict, detail_level: str = "detailed",
                   image_generation_model: str = "General / Auto",
                   system_prompt_preset: str = "Claude",
                   custom_categories: str = "", bbox_mode: str = "opencv_foreground", **_):
        cfg = ollama_model or {}
        cache_key = {
            "image": image_tensor_hash(image),
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "temperature": cfg.get("temperature", 0.3),
            "num_ctx": cfg.get("num_ctx", 8192),
            "seed": cfg.get("seed", -1),
            "keep_alive": cfg.get("keep_alive", "5m"),
            "detail_level": detail_level,
            "image_generation_model": image_generation_model,
            "system_prompt_preset": system_prompt_preset,
            "custom_categories": custom_categories,
            "bbox_mode": bbox_mode,
            "num_predict": 4096,
            "response_format": "json",
            "request_method": "generate",
        }
        return json.dumps(cache_key, sort_keys=True, separators=(",", ":"))

    def analyze(self, image, ollama_model: dict, detail_level: str = "detailed",
                image_generation_model: str = "General / Auto",
                system_prompt_preset: str = "Claude", custom_categories: str = "",
                bbox_mode: str = "opencv_foreground"):
        cfg = ollama_model
        img_b64 = tensor_to_base64(image)

        system = build_system_prompt(
            detail_level, custom_categories, image_generation_model,
            system_prompt_preset=system_prompt_preset,
        )
        user_prompt = build_user_prompt(detail_level, image_generation_model)

        _log.info("[OllamaNodes] Analyzing image with %s (detail=%s)", cfg["model"], detail_level)

        raw = generate(
            base_url=cfg["base_url"],
            model=cfg["model"],
            prompt=user_prompt,
            system=system,
            temperature=min(float(cfg.get("temperature", 0.2)), 0.2),
            top_p=cfg.get("top_p", 0.9),
            top_k=cfg.get("top_k", 40),
            repeat_penalty=cfg.get("repeat_penalty", 1.1),
            num_ctx=cfg.get("num_ctx", 8192),
            num_predict=4096,
            seed=cfg.get("seed", -1),
            keep_alive=cfg.get("keep_alive", "5m"),
            images=[img_b64],
            response_format="json",
        )

        parsed = extract_json_block(raw)
        if parsed is None:
            _log.warning("[OllamaNodes] Failed to parse JSON, returning raw text")
            return (raw, raw, "", "", "", "", "", "", "", "", "", "")

        parsed = _normalize_analysis(parsed)
        parsed["reconstruction_prompt"] = _select_reconstruction_prompt(
            parsed, image_generation_model
        )
        scene_graph_json = _build_scene_graph(parsed, image, bbox_mode)
        full_json = json.dumps(parsed, indent=2, ensure_ascii=False)

        return (
            full_json,
            parsed.get("subject", ""),
            parsed.get("clothing", ""),
            parsed.get("pose", ""),
            parsed.get("location", ""),
            parsed.get("lighting", ""),
            parsed.get("camera", ""),
            parsed.get("color_palette", ""),
            parsed.get("art_style", ""),
            parsed.get("mood", ""),
            parsed.get("reconstruction_prompt", ""),
            scene_graph_json,
        )