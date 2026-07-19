"""
Save prompts and analysis text produced by analyzer nodes.
"""
from __future__ import annotations

import json
import os
import re
import time


def _safe_filename(text: str, fallback: str = "analyzer_prompt") -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip()).strip("._")
    return name or fallback


def _extract_reconstruction_prompt(full_analysis: str) -> str:
    try:
        data = json.loads(full_analysis)
    except Exception:
        return ""

    if isinstance(data, dict):
        value = data.get("reconstruction_prompt")
        if isinstance(value, str):
            return value.strip()

        frames = data.get("frames")
        if isinstance(frames, list):
            prompts = []
            for item in frames:
                if not isinstance(item, dict):
                    continue
                analysis = item.get("analysis")
                if isinstance(analysis, dict):
                    prompt = analysis.get("reconstruction_prompt")
                    if isinstance(prompt, str) and prompt.strip():
                        frame = item.get("frame", len(prompts) + 1)
                        prompts.append(f"Image {frame}: {prompt.strip()}")
            return "\n\n".join(prompts)

    return ""


def _load_prompt_store(filepath: str) -> dict:
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {"version": 1, "prompts": []}

    with open(filepath, "r", encoding="utf-8") as handle:
        raw = handle.read()

    if not raw.strip():
        return {"version": 1, "prompts": []}

    try:
        loaded = json.loads(raw)
    except Exception:
        return {
            "version": 1,
            "prompts": [],
            "legacy_text": raw,
        }

    if isinstance(loaded, dict):
        prompts = loaded.get("prompts")
        if isinstance(prompts, list):
            loaded.setdefault("version", 1)
            return loaded
        return {
            "version": 1,
            "prompts": [loaded],
        }

    if isinstance(loaded, list):
        return {
            "version": 1,
            "prompts": loaded,
        }

    return {
        "version": 1,
        "prompts": [{
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "legacy_value": loaded,
        }],
    }


def _write_text_atomic(filepath: str, text: str) -> None:
    tmp_path = f"{filepath}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, filepath)


def _build_entry(
    content_mode: str,
    prompt: str,
    saved_text: str,
    analysis: str,
    custom: str,
    write_mode: str,
) -> dict:
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "content_mode": content_mode,
        "write_mode": write_mode,
        "prompt": prompt,
        "saved_text": saved_text,
    }
    if analysis:
        try:
            entry["full_analysis"] = json.loads(analysis)
        except Exception:
            entry["full_analysis"] = analysis
    if custom:
        entry["custom_text"] = custom
    return entry


def _write_json_store(filepath: str, entry: dict, append: bool) -> str:
    existing = _load_prompt_store(filepath) if append else {"version": 1, "prompts": []}
    prompts = existing.get("prompts")
    if not isinstance(prompts, list):
        prompts = []
        existing["prompts"] = prompts

    prompts.append(entry)
    existing["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    existing["count"] = len(prompts)
    saved_json = json.dumps(existing, indent=2, ensure_ascii=False)
    _write_text_atomic(filepath, saved_json)
    return saved_json


class OllamaAnalyzerPromptSaver:
    """Save analyzer-generated prompts or full analysis text to disk."""

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION = "save_prompt"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filepath", "saved_text")
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = (
        "Path to the saved text file",
        "The exact text written to disk",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "analyzer_prompt",
                    "multiline": False,
                    "tooltip": "Filename prefix without extension",
                }),
                "save_directory": (["output", "temp"], {
                    "default": "output",
                    "tooltip": "ComfyUI folder to save into",
                }),
                "content_mode": ([
                    "prompt_only",
                    "full_analysis",
                    "prompt_and_analysis",
                    "custom_text",
                ], {
                    "default": "prompt_only",
                    "tooltip": "What to write to the file",
                }),
            },
            "optional": {
                "reconstruction_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Prompt output from the analyzer",
                }),
                "full_analysis": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Full JSON analysis output from the analyzer",
                }),
                "custom_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Any prompt/text to save",
                }),
                "extension": (["txt", "json", "md"], {
                    "default": "txt",
                    "tooltip": "File extension",
                }),
                "write_mode": (["overwrite", "append_text", "append_json_entry"], {
                    "default": "overwrite",
                    "tooltip": "Append to an existing file instead of overwriting. append_json_entry stores prompts in one JSON file.",
                }),
                "add_timestamp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append timestamp to avoid overwriting previous saves",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def save_prompt(
        self,
        filename_prefix: str = "analyzer_prompt",
        save_directory: str = "output",
        content_mode: str = "prompt_only",
        reconstruction_prompt: str = "",
        full_analysis: str = "",
        custom_text: str = "",
        extension: str = "txt",
        write_mode: str = "overwrite",
        add_timestamp: bool = True,
    ):
        import folder_paths

        if save_directory == "temp":
            base_dir = folder_paths.get_temp_directory()
        else:
            base_dir = folder_paths.get_output_directory()
        os.makedirs(base_dir, exist_ok=True)

        prompt = str(reconstruction_prompt or "").strip()
        if not prompt and full_analysis:
            prompt = _extract_reconstruction_prompt(full_analysis)

        analysis = str(full_analysis or "").strip()
        custom = str(custom_text or "").strip()

        if content_mode == "full_analysis":
            saved_text = analysis or custom or prompt
        elif content_mode == "prompt_and_analysis":
            parts = []
            if prompt:
                parts.append(prompt)
            if analysis:
                parts.append(analysis)
            saved_text = "\n\n--- FULL ANALYSIS ---\n\n".join(parts) if parts else custom
        elif content_mode == "custom_text":
            saved_text = custom or prompt or analysis
        else:
            saved_text = prompt or custom or analysis

        prefix = _safe_filename(filename_prefix)
        ext = "json" if write_mode == "append_json_entry" else extension
        ext = ext if ext in {"txt", "json", "md"} else "txt"
        if add_timestamp and write_mode != "append_json_entry":
            prefix = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
        filepath = os.path.join(base_dir, f"{prefix}.{ext}")

        if ext == "json":
            entry = _build_entry(content_mode, prompt, saved_text, analysis, custom, write_mode)
            saved_text = _write_json_store(
                filepath,
                entry,
                append=write_mode in {"append_text", "append_json_entry"},
            )
        elif write_mode == "append_text":
            with open(filepath, "a", encoding="utf-8") as handle:
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    handle.write("\n\n")
                handle.write(saved_text)
        else:
            _write_text_atomic(filepath, saved_text)

        print(f"[OllamaNodes] Analyzer prompt saved: {filepath}")
        return (filepath, saved_text)
