"""
text_utils.py — Text parsing and extraction utilities.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_json_block(text: str) -> Optional[dict]:
    """
    Extract the first JSON object from text, handling markdown code fences.
    Returns parsed dict or None if no valid JSON found.
    """
    # Try markdown code block first
    patterns = [
        r'```json\s*\n?(.*?)\n?\s*```',
        r'```\s*\n?(.*?)\n?\s*```',
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            candidate = match.strip()
            if not candidate.startswith("{"):
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Last resort: find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def extract_section(text: str, section_name: str) -> str:
    """
    Extract content under a section header like '## Subject' or '**Subject:**'.
    """
    patterns = [
        rf'(?:##?\s*{section_name}\s*:?\s*\n)(.*?)(?=\n##?\s|\n\*\*|\Z)',
        rf'(?:\*\*{section_name}\*\*\s*:?\s*)(.*?)(?=\n\*\*|\n##|\Z)',
        rf'(?:{section_name}\s*:\s*)(.*?)(?=\n[A-Z]|\n\*\*|\n##|\Z)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""


def clean_prompt(text: str) -> str:
    """Clean up a generated prompt — remove markdown artifacts, normalize whitespace."""
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,3}\s+', '', text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
