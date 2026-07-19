"""
ollama_json_extractor.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama JSON Extractor

Runs a structured-output query and parses the JSON response.
Supports two modes:
  • schema_guided – you provide a JSON schema; the model is instructed
    to return data that matches it (enforced via Ollama's format=json)
  • key_extract    – simple dot-notation path extraction from the result

Outputs
  STRING     – extracted / formatted value (stringified)
  FLOAT      – numeric value if the extracted field is a number, else 0.0
  INT        – integer value if the extracted field is an integer, else 0
  STRING     – full raw JSON string
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .ollama_client import generate_stream, OllamaConnectionError


def _dot_get(data: Any, path: str) -> Any:
    """Resolve a dot-notation path like 'result.score' in nested dicts."""
    if not path:
        return data
    parts = path.split(".")
    cur = data
    for part in parts:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def _extract_json_block(text: str) -> str:
    """Try to pull a JSON block out of free text (handles ```json fences)."""
    # Strip markdown code fences
    import re
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        return fenced.group(1).strip()
    # Find outermost { } or [ ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end   = text.rfind(end_char)
        if start != -1 and end > start:
            return text[start : end + 1]
    return text.strip()


class OllamaJSONExtractor:

    CATEGORY = "Ollama-Magic-Nodes/Utility"
    FUNCTION      = "extract_json"
    RETURN_TYPES  = ("STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES  = ("extracted_value", "float_value", "int_value", "raw_json")
    OUTPUT_NODE   = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default":   (
                            "Analyse the sentiment of this text and return JSON:\n"
                            '"I absolutely love this product! It exceeded all my expectations."'
                        ),
                        "description": "Natural language query (the model will respond in JSON)",
                    },
                ),
            },
            "optional": {
                "json_schema": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": json.dumps(
                            {
                                "sentiment": "positive | neutral | negative",
                                "confidence": 0.95,
                                "keywords": ["love", "exceeded"],
                                "summary": "Short sentence",
                            },
                            indent=2,
                        ),
                        "description": "Example / schema shown to the model (leave blank for free-form JSON)",
                    },
                ),
                "extract_path": (
                    "STRING",
                    {
                        "multiline":   False,
                        "default":     "sentiment",
                        "description": "Dot-notation path to the value you want (e.g. 'result.score'). Leave blank for the full object.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default":   (
                            "You are a structured-data extraction assistant. "
                            "Always respond with valid JSON only. No markdown, no explanation."
                        ),
                    },
                ),
                "num_predict": (
                    "INT",
                    {"default": 512, "min": 64, "max": 4096, "step": 1},
                ),
                "temperature_override": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01,
                     "description": "0 = deterministic JSON (recommended)"},
                ),
                "strict_json": (
                    "BOOLEAN",
                    {
                        "default":     True,
                        "description": "Use Ollama JSON mode (format=json) for guaranteed valid JSON output",
                    },
                ),
                "schema_format": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "description": "Optional JSON schema for Ollama format field. Overrides strict_json when valid.",
                    },
                ),
                "stream_to_console": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "think_mode": (
                    ["off", "on", "low", "medium", "high"],
                    {"default": "off"},
                ),
            },
        }

    def extract_json(
        self,
        ollama_model:          dict,
        prompt:                str,
        json_schema:           str  = "",
        extract_path:          str  = "",
        system_prompt:         str  = "",
        num_predict:           int  = 512,
        temperature_override:  float = 0.0,
        strict_json:           bool = True,
        schema_format:         str  = "",
        stream_to_console:     bool = False,
        think_mode:            str  = "off",
    ):
        base_url   = ollama_model["base_url"]
        model      = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")

        temperature = (
            temperature_override
            if temperature_override >= 0
            else ollama_model.get("temperature", 0.7)
        )

        format_payload = "json" if strict_json else ""
        if schema_format.strip():
            try:
                format_payload = json.loads(schema_format)
            except json.JSONDecodeError:
                print("[OllamaJSONExtractor] ⚠ schema_format is invalid JSON, falling back to strict_json flag.")

        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)

        # Build the full prompt with schema hint if provided
        full_prompt = prompt.strip()
        if json_schema.strip():
            full_prompt += f"\n\nRespond with JSON matching this structure:\n{json_schema.strip()}"

        if stream_to_console:
            print(f"\n[OllamaJSONExtractor] ▶ {model} extracting JSON…\n", flush=True)

        tokens: list[str] = []
        try:
            for token in generate_stream(
                base_url    = base_url,
                model       = model,
                prompt      = full_prompt,
                system      = system_prompt,
                temperature = temperature,
                top_p       = ollama_model.get("top_p",   0.9),
                top_k       = ollama_model.get("top_k",   40),
                num_predict = num_predict,
                seed        = ollama_model.get("seed",    -1),
                format      = format_payload,
                think       = think,
                keep_alive  = keep_alive,
            ):
                tokens.append(token)
                if stream_to_console:
                    sys.stdout.write(token)
                    sys.stdout.flush()
        except OllamaConnectionError as exc:
            raise RuntimeError(str(exc)) from exc

        if stream_to_console:
            print("\n[OllamaJSONExtractor] ✅ Done.", flush=True)

        raw_text = "".join(tokens)

        # ── Parse JSON ────────────────────────────────────────────
        json_str = _extract_json_block(raw_text)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[OllamaJSONExtractor] ⚠ JSON parse error: {e}\nRaw: {raw_text}")
            return (raw_text, 0.0, 0, raw_text)

        # ── Extract value ─────────────────────────────────────────
        extracted: Any = _dot_get(parsed, extract_path.strip()) if extract_path.strip() else parsed

        # Serialise to string
        if isinstance(extracted, (dict, list)):
            extracted_str = json.dumps(extracted, indent=2)
        else:
            extracted_str = str(extracted) if extracted is not None else ""

        # Numeric coercions
        try:
            float_val = float(extracted)
        except (TypeError, ValueError):
            float_val = 0.0

        try:
            int_val = int(extracted)
        except (TypeError, ValueError):
            int_val = 0

        print(
            f"[OllamaJSONExtractor] ✅  path='{extract_path}' → {extracted_str[:80]}"
        )

        raw_json = json.dumps(parsed, indent=2)
        return (extracted_str, float_val, int_val, raw_json)
