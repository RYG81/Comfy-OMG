"""
Analyze a folder of related images as one consistent photoset.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from ...ollama_client import generate
from ...prompts.image_analysis import REQUIRED_KEYS, build_system_prompt, build_user_prompt
from ...utils.image_utils import pil_to_base64
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

PHOTOSET_SYSTEM_PROMPT = """You are an expert image prompt engineer and continuity director.

You will receive analyses from multiple images that belong to the same photoset. Your job is to identify what is visibly consistent across the set and create a stable generation prompt that can reproduce the same subject/style consistently across new images.

Photoset hard rule:
All generated images must look like they were captured in the same shoot at the same time, in the same location, with the same subject identity, same wardrobe/clothing coverage, same lighting setup, same camera/style, and same overall environment. A photoset is not a new scene each time. It is a set of closely related shots from one continuous session.

Focus on repeated evidence across images:
- subject identity and body/face traits
- clothing coverage/status, garments, accessories, and outfit continuity
- hair, makeup, pose language, expression range
- location, props, background, lighting, camera, lens, color palette, mood, and art/photography style
- small elements that vary between images and should remain flexible, such as pose, gaze, hand placement, micro-expression, crop, or slight camera framing

Do not overfit to one image if other images contradict it. If a detail is inconsistent or unclear, mark one direct visibility/variation statement without listing alternate guesses. Do not make the final prompt change location, time of day, outfit, lighting, art style, or subject identity between images.

Respond in this EXACT JSON format:
{
    "photoset_description": "Brief overview of the photoset as a coherent set.",
    "consistent_subject_identity": "Stable subject identity and visual traits across the set.",
    "consistent_clothing": "One stable clothing coverage/status when visible evidence supports it, plus stable garments, accessories, materials, and colors. If not stable, give one direct limitation without alternate guesses.",
    "consistent_environment": "Stable setting, props, background, and atmosphere.",
    "consistent_camera_lighting_style": "Stable camera, framing, lens, lighting, color, and style traits.",
    "variable_elements": "Only small photoset-safe variations, such as pose, gaze, gesture, crop, expression, or slight framing. Do not vary location, time, outfit, subject identity, lighting setup, or core style.",
    "consistent_prompt": "A single model-ready prompt for generating more images from this same photoset. It must lock same subject, same clothing coverage/outfit, same location, same time, same lighting, same camera/style, and only allow small pose/framing/expression variation.",
    "negative_prompt": "Things to avoid that would break photoset consistency, such as identity drift, outfit drift, lighting mismatch, wrong setting, warped anatomy, or inconsistent style."
}

Output ONLY valid JSON."""

PHOTOSET_UPDATE_SYSTEM_PROMPT = """You are an expert photoset continuity tracker.

You will receive:
1. The current running consistency state for a photoset.
2. The analysis of one newly processed image from that same photoset.

Photoset hard rule:
All generated images must look like they were captured in the same shoot at the same time, in the same location, with the same subject identity, same wardrobe/clothing coverage, same lighting setup, same camera/style, and same overall environment. A photoset is not a new scene each time. It is a set of closely related shots from one continuous session.

Update the consistency state using only evidence from the images seen so far. Keep stable repeated traits, mark contradictions with one direct variation statement, and avoid overfitting to one image. Variations should be small and photoset-safe: pose, gaze, hand placement, expression, crop, or slight framing. Never turn variable elements into a new location, outfit, time of day, lighting setup, subject identity, or visual style.

Respond in this EXACT JSON format:
{
    "photoset_description": "Brief overview of the photoset as understood so far.",
    "consistent_subject_identity": "Stable subject identity and visual traits across images seen so far.",
    "consistent_clothing": "One stable clothing coverage/status when visible evidence supports it, plus stable garments, accessories, materials, and colors. If not stable, give one direct limitation without alternate guesses.",
    "consistent_environment": "Stable setting, props, background, and atmosphere.",
    "consistent_camera_lighting_style": "Stable camera, framing, lens, lighting, color, and style traits.",
    "variable_elements": "Only small photoset-safe variations, such as pose, gaze, gesture, crop, expression, or slight framing. Do not vary location, time, outfit, subject identity, lighting setup, or core style.",
    "consistent_prompt": "A single model-ready prompt for generating more images from this same photoset. It must lock same subject, same clothing coverage/outfit, same location, same time, same lighting, same camera/style, and only allow small pose/framing/expression variation.",
    "negative_prompt": "Things to avoid that would break photoset consistency."
}

Output ONLY valid JSON."""

EMPTY_PHOTOSET_STATE = {
    "photoset_description": "",
    "consistent_subject_identity": "",
    "consistent_clothing": "",
    "consistent_environment": "",
    "consistent_camera_lighting_style": "",
    "variable_elements": "",
    "consistent_prompt": "",
    "negative_prompt": "",
}


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _resolve_folder(folder_path: str) -> Path:
    raw = str(folder_path or "").strip().strip('"')
    if not raw:
        raise ValueError("folder_path is empty")

    path = Path(raw)
    if path.is_absolute():
        return path

    try:
        import folder_paths

        return Path(folder_paths.get_input_directory()) / path
    except Exception:
        return Path.cwd() / path


def _list_images(folder: Path, recursive: bool, filename_filter: str, max_images: int) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    filter_text = str(filename_filter or "").strip().lower()
    images = []
    for path in folder.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if filter_text and filter_text not in path.name.lower():
            continue
        images.append(path)
    images.sort(key=_natural_key)
    return images[:max_images]


def _load_image_base64(path: Path, max_size: int = 1024) -> str:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return pil_to_base64(img)


def _load_image_tensor(path: Path, width: int, height: int, pad_color: str):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = ImageOps.pad(img, (width, height), method=Image.LANCZOS, color=pad_color, centering=(0.5, 0.5))
        arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


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


def _compact_analysis(item: dict) -> str:
    analysis = item.get("analysis", {})
    if not isinstance(analysis, dict):
        return str(analysis)
    lines = [f"Image: {item.get('filename', '')}"]
    for key in (
        "subject", "body_details", "clothing", "pose", "location", "lighting", "camera",
        "color_palette", "art_style", "mood", "reconstruction_prompt",
    ):
        value = analysis.get(key, "")
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


class OllamaPhotosetFolderAnalyzer:
    """Analyze a folder of same-photoset images and create a consistent prompt."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "analyze_folder"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "photoset_analysis_json",
        "consistent_prompt",
        "per_image_prompts",
        "identity_style_guide",
        "negative_prompt",
        "current_image",
        "current_index",
        "photo_count",
        "current_image_path",
    )
    OUTPUT_TOOLTIPS = (
        "Full JSON containing source files, per-image analyses, and photoset synthesis",
        "Single prompt for generating more images from the same photoset",
        "Per-image reconstruction prompts from the folder",
        "Stable identity, clothing, camera, lighting, and style guide",
        "Negative prompt for avoiding photoset consistency breaks",
        "Selected image from the folder for preview",
        "Selected image index, clamped to folder range",
        "Actual image count found in the folder before max_images limit",
        "Path to the selected preview image",
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute folder path, or path relative to ComfyUI input folder",
                }),
                "ollama_model": ("OLLAMA_MODEL",),
                "detail_level": (["basic", "detailed", "exhaustive"], {
                    "default": "detailed",
                    "tooltip": "Detail level for each image analysis",
                }),
                "image_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "1-based image index to preview/select from the sorted folder list",
                }),
            },
            "optional": {
                "max_images": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Maximum number of images to analyze from the folder",
                }),
                "recursive": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include images in subfolders",
                }),
                "filename_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional text that filenames must contain",
                }),
                "custom_categories": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Additional analysis guidance for every image",
                }),
                "consistency_focus": ("STRING", {
                    "default": "same photoset/same shoot: lock identity, clothing coverage and outfit, location, time, lighting setup, camera style, color palette; vary only pose, expression, gesture, crop, and slight framing",
                    "multiline": True,
                    "tooltip": "What the final photoset prompt should keep most consistent",
                }),
                "consistency_mode": (["incremental", "final_batch"], {
                    "default": "incremental",
                    "tooltip": "Incremental updates a running guide after each image. Final batch summarizes all image analyses at the end.",
                }),
                "analysis_scope": (["selected_image_only", "up_to_current_index", "first_max_images"], {
                    "default": "selected_image_only",
                    "tooltip": "Choose which folder images to analyze this run. selected_image_only is fastest and uses image_index.",
                }),
                "preview_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Preview image width",
                }),
                "preview_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Preview image height",
                }),
                "preview_pad_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Padding color for preview image",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, folder_path: str, ollama_model: dict, detail_level: str = "detailed",
                   image_index: int = 1,
                   max_images: int = 12, recursive: bool = False, filename_filter: str = "",
                   custom_categories: str = "", consistency_focus: str = "",
                   consistency_mode: str = "incremental", preview_width: int = 512,
                   preview_height: int = 512, preview_pad_color: str = "#000000",
                   analysis_scope: str = "selected_image_only", **_):
        cfg = ollama_model or {}
        try:
            folder = _resolve_folder(folder_path)
            files = _list_images(folder, recursive, filename_filter, max_images)
            file_state = [
                (str(path), path.stat().st_mtime_ns, path.stat().st_size)
                for path in files
            ]
        except Exception as exc:
            file_state = [f"error:{exc}"]

        cache_key = {
            "files": file_state,
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
            "temperature": cfg.get("temperature", 0.3),
            "num_ctx": cfg.get("num_ctx", 8192),
            "seed": cfg.get("seed", -1),
            "keep_alive": cfg.get("keep_alive", "5m"),
            "detail_level": detail_level,
            "image_index": image_index,
            "custom_categories": custom_categories,
            "consistency_focus": consistency_focus,
            "consistency_mode": consistency_mode,
            "analysis_scope": analysis_scope,
            "preview_width": preview_width,
            "preview_height": preview_height,
            "preview_pad_color": preview_pad_color,
        }
        return json.dumps(cache_key, sort_keys=True, separators=(",", ":"), default=str)

    def analyze_folder(
        self,
        folder_path: str,
        ollama_model: dict,
        detail_level: str = "detailed",
        image_index: int = 1,
        max_images: int = 12,
        recursive: bool = False,
        filename_filter: str = "",
        custom_categories: str = "",
        consistency_focus: str = "same photoset/same shoot: lock identity, clothing coverage and outfit, location, time, lighting setup, camera style, color palette; vary only pose, expression, gesture, crop, and slight framing",
        consistency_mode: str = "incremental",
        analysis_scope: str = "selected_image_only",
        preview_width: int = 512,
        preview_height: int = 512,
        preview_pad_color: str = "#000000",
    ):
        cfg = ollama_model
        try:
            folder = _resolve_folder(folder_path)
            if not folder.exists() or not folder.is_dir():
                raise ValueError(f"Folder not found: {folder}")
            all_files = _list_images(folder, recursive, filename_filter, 999999)
            photo_count = len(all_files)
            if not all_files:
                raise ValueError(f"No images found in folder: {folder}")
            selected_index = max(1, min(int(image_index), photo_count))
            selected_path = all_files[selected_index - 1]
            current_image = _load_image_tensor(selected_path, preview_width, preview_height, preview_pad_color)
            if analysis_scope == "up_to_current_index":
                files = all_files[:min(selected_index, max_images)]
            elif analysis_scope == "first_max_images":
                files = all_files[:max_images]
            else:
                files = [selected_path]
        except Exception as exc:
            error_json = json.dumps({"error": str(exc), "folder_path": folder_path}, indent=2)
            fallback = torch.zeros((1, preview_height, preview_width, 3), dtype=torch.float32)
            return (error_json, "", "", "", "", fallback, 0, 0, "")

        system = build_system_prompt(detail_level, custom_categories)
        user_prompt = build_user_prompt(detail_level)
        analyses = []
        running_state = dict(EMPTY_PHOTOSET_STATE)

        for index, path in enumerate(files, start=1):
            _log.info(
                "[OllamaNodes] Photoset folder analyzer image %d/%d: %s",
                index, len(files), path.name,
            )
            raw = generate(
                base_url=cfg["base_url"],
                model=cfg["model"],
                prompt=f"Image {index} of {len(files)} from one related photoset.\n{user_prompt}",
                system=system,
                temperature=min(float(cfg.get("temperature", 0.2)), 0.2),
                top_p=cfg.get("top_p", 0.9),
                top_k=cfg.get("top_k", 40),
                repeat_penalty=cfg.get("repeat_penalty", 1.1),
                num_ctx=cfg.get("num_ctx", 8192),
                num_predict=4096,
                seed=cfg.get("seed", -1),
                keep_alive=cfg.get("keep_alive", "5m"),
                images=[_load_image_base64(path)],
                response_format="json",
            )
            parsed = extract_json_block(raw)
            if parsed is None:
                parsed = {"raw_output": raw}
            else:
                parsed = _normalize_analysis(parsed)
            analyses.append({
                "index": index,
                "filename": path.name,
                "path": str(path),
                "analysis": parsed,
            })

            if consistency_mode == "incremental":
                update_raw = generate(
                    base_url=cfg["base_url"],
                    model=cfg["model"],
                    prompt=(
                        f"Consistency focus:\n{consistency_focus}\n\n"
                        f"Images processed so far: {index} of {len(files)}\n\n"
                        f"Current running consistency state:\n"
                        f"{json.dumps(running_state, indent=2, ensure_ascii=False)}\n\n"
                        f"New image analysis:\n{_compact_analysis(analyses[-1])}\n\n"
                        "Update the photoset consistency state JSON."
                    ),
                    system=PHOTOSET_UPDATE_SYSTEM_PROMPT,
                    temperature=min(float(cfg.get("temperature", 0.2)), 0.2),
                    top_p=cfg.get("top_p", 0.9),
                    top_k=cfg.get("top_k", 40),
                    repeat_penalty=cfg.get("repeat_penalty", 1.1),
                    num_ctx=cfg.get("num_ctx", 8192),
                    num_predict=3072,
                    seed=cfg.get("seed", -1),
                    keep_alive=cfg.get("keep_alive", "5m"),
                    response_format="json",
                )
                running_state = extract_json_block(update_raw) or running_state

        if consistency_mode == "incremental":
            synthesis = running_state
        else:
            compact = "\n\n".join(_compact_analysis(item) for item in analyses)
            synthesis_raw = generate(
                base_url=cfg["base_url"],
                model=cfg["model"],
                prompt=(
                    f"Consistency focus:\n{consistency_focus}\n\n"
                    f"Folder: {folder}\n"
                    f"Images analyzed: {len(analyses)}\n\n"
                    f"Per-image analyses:\n{compact}\n\n"
                    "Create the final consistent photoset prompt JSON."
                ),
                system=PHOTOSET_SYSTEM_PROMPT,
                temperature=min(float(cfg.get("temperature", 0.2)), 0.2),
                top_p=cfg.get("top_p", 0.9),
                top_k=cfg.get("top_k", 40),
                repeat_penalty=cfg.get("repeat_penalty", 1.1),
                num_ctx=cfg.get("num_ctx", 16384),
                num_predict=4096,
                seed=cfg.get("seed", -1),
                keep_alive=cfg.get("keep_alive", "5m"),
                response_format="json",
            )
            synthesis = extract_json_block(synthesis_raw) or {"raw_output": synthesis_raw}

        payload = {
            "folder": str(folder),
            "photo_count": photo_count,
            "current_index": selected_index,
            "current_image_path": str(selected_path),
            "image_count": len(analyses),
            "analysis_scope": analysis_scope,
            "consistency_focus": consistency_focus,
            "consistency_mode": consistency_mode,
            "images": analyses,
            "photoset_synthesis": synthesis,
        }
        full_json = json.dumps(payload, indent=2, ensure_ascii=False)

        per_image_prompts = []
        for item in analyses:
            analysis = item.get("analysis", {})
            if isinstance(analysis, dict):
                prompt = analysis.get("reconstruction_prompt", "")
                if prompt:
                    per_image_prompts.append(f"{item['filename']}: {prompt}")

        guide_parts = [
            str(synthesis.get("consistent_subject_identity", "")) if isinstance(synthesis, dict) else "",
            str(synthesis.get("consistent_clothing", "")) if isinstance(synthesis, dict) else "",
            str(synthesis.get("consistent_environment", "")) if isinstance(synthesis, dict) else "",
            str(synthesis.get("consistent_camera_lighting_style", "")) if isinstance(synthesis, dict) else "",
            str(synthesis.get("variable_elements", "")) if isinstance(synthesis, dict) else "",
        ]
        identity_style_guide = "\n\n".join(part for part in guide_parts if part.strip())

        return (
            full_json,
            str(synthesis.get("consistent_prompt", "")) if isinstance(synthesis, dict) else synthesis_raw,
            "\n\n".join(per_image_prompts),
            identity_style_guide,
            str(synthesis.get("negative_prompt", "")) if isinstance(synthesis, dict) else "",
            current_image,
            selected_index,
            photo_count,
            str(selected_path),
        )
