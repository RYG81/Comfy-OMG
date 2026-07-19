"""
ollama_prompt_builder.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama Prompt Builder

AI-powered prompt engineering node. Feed it a rough idea and it
outputs a polished, image-generation-ready positive/negative prompt pair,
plus an optional style-transfer variant.

✨ Chunked-batch mode for ultra-long generations (60-200+ prompts)

Modes
  enhance_sd       – Expand a short concept into a full SD/SDXL prompt
  enhance_flux     – Style tuned for FLUX.1 (natural language)
  summarize        – Summarise long text to a concise prompt
  translate        – Translate text to English and clean it up
  custom           – Use your own system instruction
  qwen_image       – Qwen-Image style rich scene prompts
  chatgpt_image_2  – Instruction-style prompts with constraints
  gemini_image     – Natural language visual briefs

Outputs
  STRING  – primary prompt (ready to pipe into KSampler positive)
  STRING  – negative prompt
  STRING  – raw model output (for debugging)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import sys
import logging
from typing import Optional, Dict, Any, List

from ...prompts.prompt_enhance import PROMPTFORGE_2026U_SYSTEM_PROMPT
from .ollama_client import generate_stream, OllamaConnectionError
from ...utils.text_utils import extract_json_block

_log = logging.getLogger(__name__)

# ── Preset system prompts ──────────────────────────────────────────
_SYSTEM_PROMPTS = {
    "enhance_sd": """You are an expert Stable Diffusion prompt engineer.
Transform the user's rough concept into a detailed, effective image generation prompt.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: 
NEGATIVE: 

Guidelines for POSITIVE:
- Add art style, lighting, camera angle, mood, quality boosters
- Use parentheses for emphasis: (masterpiece:1.2)
- Include artist names if appropriate

Guidelines for NEGATIVE:
- Always include: blurry, lowres, bad anatomy, watermark, text, error
""",

    "enhance_flux": """You are an expert prompt engineer for FLUX.1 image models.
FLUX works best with natural language sentences, not comma-separated tags.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: <2-4 natural language sentences describing the scene>
NEGATIVE: 

Make the description vivid, specific, and cinematic.
""",

    "2026U": PROMPTFORGE_2026U_SYSTEM_PROMPT,

    "summarize": """You are a text summarisation expert.
Condense the user's text into a concise, clear single-line summary
that captures the core subject matter.

OUTPUT FORMAT – respond with EXACTLY two lines:
POSITIVE: 
NEGATIVE: irrelevant, off-topic
""",

    "translate": """You are a professional translator and prompt cleaner.
Translate the user's text to English, fix grammar, and optimise it
for use as an image generation prompt.

OUTPUT FORMAT – respond with EXACTLY two lines:
POSITIVE: 
NEGATIVE: 
""",

    "custom": "",

    "qwen_image": """You are an expert prompt writer for Qwen-Image style models.
Create a visually rich, direct image prompt in natural language with strong scene specificity.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: 
NEGATIVE: 

Guidelines:
- Include subject, composition, camera framing, lighting, color palette, environment, mood
- Prefer concrete visual details over abstract adjectives
- If user intent is photographic, include lens/framing cues
- Keep prompt coherent and production-ready
""",

    "chatgpt_image_2": """You are an expert prompt writer for ChatGPT image generation style workflows.
Convert the user's idea into a clear instruction-style image prompt that emphasizes intent and constraints.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: 
NEGATIVE: 

Guidelines:
- Structure prompt as: subject + context + style + composition + lighting + quality targets
- Mention specific materials/textures where relevant
- Include "no text or watermark" unless user explicitly asks for text
- Keep it concise but specific
""",

    "gemini_image": """You are an expert prompt writer for Gemini image generation style prompting.
Produce a polished, natural-language visual brief suitable for multimodal image generation.

OUTPUT FORMAT – respond with EXACTLY these two lines, no other text:
POSITIVE: <2-5 sentence visual brief with cinematic clarity>
NEGATIVE: 

Guidelines:
- Emphasize narrative clarity, visual hierarchy, and realistic lighting behavior
- Describe foreground/midground/background when helpful
- Add rendering intent (photo, illustration, 3D, watercolor, etc.) based on user request
- Avoid contradictory style instructions
""",
}

# ── Parsing & Helper Utilities ────────────────────────────────────
_MARKDOWN_BLOCK = re.compile(
    r"```(?:\w+)?\s*\n?([\s\S]*?)\n?```", re.IGNORECASE)
_POS_RE = re.compile(r"^\s*POSITIVE\s*:\s*", re.IGNORECASE)
_NEG_RE = re.compile(r"^\s*NEGATIVE\s*:\s*", re.IGNORECASE)


def _parse_positive_negative(raw: str) -> tuple[str, str]:
    """Extract POSITIVE / NEGATIVE lines with markdown stripping and safe fallbacks."""
    clean = _MARKDOWN_BLOCK.sub(r"\1", raw).strip()
    positive = negative = ""
    found_pos = found_neg = False

    for line in clean.splitlines():
        if m := _POS_RE.match(line):
            positive = line[m.end():].strip()
            found_pos = True
        elif m := _NEG_RE.match(line):
            negative = line[m.end():].strip()
            found_neg = True

    if not found_pos:
        paragraphs = [p.strip()
                      for p in clean.split("\n\n") if len(p.strip()) > 10]
        positive = paragraphs[0] if paragraphs else clean
        _log.warning(
            "POSITIVE: label missing. Using first substantial paragraph as fallback.")

    return positive.strip(), negative.strip()


def _append_smart(base: str, extra: str, sep: str = ", ") -> str:
    """Punctuation-aware string concatenation to avoid ',, ' or '., ' collisions."""
    if not extra.strip():
        return base
    base = base.rstrip(" ,.;:")
    extra = extra.lstrip(" ,.;:")
    return f"{base}{sep}{extra}" if base else extra


def _generate_chunk(
    base_url: str,
    model: str,
    system: str,
    user_input: str,
    temperature: float,
    ollama_model: Dict[str, Any],
    num_predict: int,
    num_ctx: int,
    keep_alive: str,
    stream_to_console: bool,
    chunk_label: str = "",
) -> tuple[str, str, str]:
    """Internal helper: generate one chunk and return (positive, negative, raw)."""
    if chunk_label:
        _log.info("▶ Chunk %s | temp=%.2f | ctx=%d",
                  chunk_label, temperature, num_ctx)

    tokens: list[str] = []
    try:
        for token in generate_stream(
            base_url=base_url,
            model=model,
            prompt=user_input,
            system=system,
            temperature=temperature,
            top_p=ollama_model.get("top_p", 0.9),
            top_k=ollama_model.get("top_k", 40),
            repeat_penalty=ollama_model.get("repeat_penalty", 1.1),
            num_predict=num_predict,
            num_ctx=num_ctx,
            seed=ollama_model.get("seed", -1),
            think=False,
            keep_alive=keep_alive,
        ):
            tokens.append(token)
            if stream_to_console:
                sys.stdout.write(token)
                sys.stdout.flush()

    except OllamaConnectionError as exc:
        raise RuntimeError(f"Ollama chunk generation failed: {exc}") from exc

    raw_output = "".join(tokens)
    if stream_to_console and chunk_label:
        sys.stdout.write(f"\n[Chunk {chunk_label}] ✅\n")
        sys.stdout.flush()

    if not raw_output.strip():
        raise RuntimeError("Model returned empty output for chunk.")

    # ✅ FIX: Parse 2 values, then return all 3 explicitly
    pos, neg = _parse_positive_negative(raw_output)
    return pos, neg, raw_output


# ── ComfyUI Node ──────────────────────────────────────────────────
class OllamaPromptBuilder:

    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "build_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "raw_output")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL", {}),
                "concept": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a futuristic city at night with neon lights",
                        "description": "Rough idea / concept / text to process",
                    },
                ),
                "mode": (
                    list(_SYSTEM_PROMPTS.keys()),
                    {"default": "enhance_sd"},
                ),
            },
            "optional": {
                "custom_system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "",
                        "description": "Used only when mode=custom"},
                ),
                "style_modifier": (
                    "STRING",
                    {"multiline": False, "default": "",
                        "description": "Extra style hint (e.g. 'cyberpunk', 'watercolor')"},
                ),
                "num_predict": (
                    "INT",
                    {"default": 2048, "min": 64, "max": 8192, "step": 64,
                     "description": "Max tokens per chunk. For batch mode, keep ≤4096."},
                ),
                "num_ctx": (
                    "INT",
                    {"default": 8192, "min": 2048, "max": 32768, "step": 1024,
                     "description": "Ollama context window. Must be >= input + num_predict."},
                ),
                "temperature_override": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "use_model_temperature": (
                    "BOOLEAN",
                    {"default": False,
                        "description": "If True, ignore temperature_override."},
                ),
                "stream_to_console": (
                    "BOOLEAN", {"default": True},
                ),
                "append_to_positive": (
                    "STRING", {"multiline": False, "default": "",
                               "description": "Text appended to positive output"},
                ),
                "append_to_negative": (
                    "STRING", {"multiline": False, "default": "",
                               "description": "Text appended to negative output"},
                ),
                # ✨ Chunked-batch mode for long generations
                "batch_mode": (
                    "BOOLEAN",
                    {"default": False,
                        "description": "Enable batch generation for long outputs (60+ prompts)."},
                ),
                "batch_count": (
                    "INT",
                    {"default": 10, "min": 1, "max": 100, "step": 1,
                     "description": "Number of prompts to generate in batch mode."},
                ),
                "batch_separator": (
                    "STRING",
                    {"multiline": False, "default": "\n\n",
                        "description": "Separator between batch items in output."},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, concept: str, mode: str, style_modifier: str = "",
                   custom_system_prompt: str = "", num_predict: int = 2048,
                   num_ctx: int = 8192, batch_mode: bool = False, batch_count: int = 10, **_):
        """Deterministic cache key."""
        return f"{mode}|{concept}|{style_modifier}|{custom_system_prompt}|{num_predict}|{num_ctx}|{batch_mode}|{batch_count}"

    @classmethod
    def VALIDATE_INPUTS(cls, mode: str, num_predict: int, num_ctx: int, batch_count: int, **_):
        """Pre-flight validation."""
        if mode not in _SYSTEM_PROMPTS and mode != "custom":
            return False, f"Invalid mode: {mode}"
        if not (64 <= num_predict <= 8192):
            return False, "num_predict must be between 64 and 8192"
        if not (2048 <= num_ctx <= 32768):
            return False, "num_ctx must be between 2048 and 32768"
        if not (1 <= batch_count <= 100):
            return False, "batch_count must be between 1 and 100"
        return True

    def build_prompt(
        self,
        ollama_model:           Dict[str, Any],
        concept:                str,
        mode:                   str = "enhance_sd",
        custom_system_prompt:   str = "",
        style_modifier:         str = "",
        num_predict:            int = 2048,
        num_ctx:                int = 8192,
        temperature_override:   float = 0.6,
        use_model_temperature:  bool = False,
        stream_to_console:      bool = True,
        append_to_positive:     str = "",
        append_to_negative:     str = "",
        batch_mode:             bool = False,
        batch_count:            int = 10,
        batch_separator:        str = "\n\n",
    ) -> tuple[str, str, str]:

        base_url = ollama_model["base_url"]
        model = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")

        temperature = (
            ollama_model.get("temperature", 0.7)
            if use_model_temperature
            else temperature_override
        )

        system = custom_system_prompt.strip(
        ) if mode == "custom" else _SYSTEM_PROMPTS[mode]
        user_input = concept.strip()
        if style_modifier.strip():
            user_input += f"\n\nStyle: {style_modifier.strip()}"

        # ── Batch Mode: Generate in chunks ──────────────────────────
        if batch_mode and batch_count > 1:
            _log.info(
                "🔄 Batch mode: generating %d prompts in chunks", batch_count)

            all_positives: List[str] = []
            all_negatives: List[str] = []
            all_raw: List[str] = []

            # Adjust per-chunk limits to stay safe within context
            chunk_predict = min(num_predict, 2048)
            chunk_ctx = max(num_ctx, 8192)

            for i in range(batch_count):
                chunk_label = f"{i+1}/{batch_count}"
                batched_input = f"{user_input}\n\nGenerate prompt #{i+1} of {batch_count}. Make each prompt unique and varied."

                try:
                    # ✅ Unpack 3 values from _generate_chunk
                    pos, neg, raw = _generate_chunk(
                        base_url=base_url,
                        model=model,
                        system=system,
                        user_input=batched_input,
                        temperature=temperature,
                        ollama_model=ollama_model,
                        num_predict=chunk_predict,
                        num_ctx=chunk_ctx,
                        keep_alive=keep_alive,
                        stream_to_console=stream_to_console,
                        chunk_label=chunk_label,
                    )
                    all_positives.append(pos)
                    if neg:
                        all_negatives.append(neg)
                    all_raw.append(raw)

                except Exception as e:
                    _log.error("❌ Chunk %s failed: %s", chunk_label, e)
                    continue

            # Aggregate results
            final_positive = batch_separator.join(all_positives)
            unique_negs = list(dict.fromkeys(n for n in all_negatives if n))
            final_negative = ", ".join(unique_negs) if unique_negs else ""
            final_raw = "\n\n---\n\n".join(all_raw)

            final_positive = _append_smart(final_positive, append_to_positive)
            final_negative = _append_smart(final_negative, append_to_negative)

            return final_positive, final_negative, final_raw

        # ── Single-Pass Mode (original behavior) ───────────────────
        estimated_input_tokens = (len(system) + len(user_input)) // 4
        if estimated_input_tokens + num_predict > num_ctx:
            _log.warning(
                "⚠️ Context window may truncate: input(~%d) + num_predict(%d) > num_ctx(%d)",
                estimated_input_tokens, num_predict, num_ctx
            )

        _log.info("▶ Generating with %s | mode=%s | temp=%.2f | ctx=%d",
                  model, mode, temperature, num_ctx)

        # ✅ Unpack 3 values from _generate_chunk (this was the bug location)
        pos, neg, raw = _generate_chunk(
            base_url=base_url,
            model=model,
            system=system,
            user_input=user_input,
            temperature=temperature,
            ollama_model=ollama_model,
            num_predict=num_predict,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            stream_to_console=stream_to_console,
        )

        pos = _append_smart(pos, append_to_positive)
        neg = _append_smart(neg, append_to_negative)

        # ✅ Return all 3 values
        return pos, neg, raw

_SINGLE_PROMPT_SYSTEM = """You are an expert AI image prompt engineer.
Create ONE production-ready image generation prompt from the user's input.

Respond with ONLY valid JSON. Do not include markdown, explanations, or extra text.

Exact JSON object:
{
  "positive_prompt": "single complete prompt ready for an image model",
  "negative_prompt": "concise negative prompt or empty string",
  "prompt_tags": "comma-separated tag version of the positive prompt",
  "summary": "short human-readable summary"
}

Rules:
- Generate exactly one prompt, not a list or photoset.
- Keep the subject and user intent intact.
- Add useful camera, lighting, composition, style, material, and quality details.
- Do not invent identity-sensitive details unless requested.
- If a locked prefix is provided by the user, copy it verbatim at the start of positive_prompt.
- If a negative base is provided by the user, include it in negative_prompt.
"""


def _parse_single_prompt_output(raw: str) -> tuple[str, str, str, str]:
    """Parse the robust JSON single-prompt format with plain-text fallback."""
    parsed = extract_json_block(raw)
    if isinstance(parsed, dict):
        positive = str(
            parsed.get("positive_prompt")
            or parsed.get("positive")
            or parsed.get("prompt")
            or parsed.get("constructed_prompt")
            or ""
        ).strip()
        negative = str(parsed.get("negative_prompt") or parsed.get("negative") or "").strip()
        tags = str(parsed.get("prompt_tags") or parsed.get("tags") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        if positive:
            return positive, negative, tags, summary

    positive, negative = _parse_positive_negative(raw)
    return positive.strip(), negative.strip(), "", "Parsed from non-JSON model output"


class OllamaSinglePromptBuilder:
    CATEGORY = "Ollama-Magic-Nodes/Prompt"
    FUNCTION = "build_single_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "prompt_tags", "summary", "raw_output")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL", {}),
                "concept": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a cinematic portrait photo in soft window light",
                        "tooltip": "Single idea to turn into one image-generation prompt",
                    },
                ),
            },
            "optional": {
                "model_style": (
                    ["photoreal", "flux", "sdxl", "qwen_image", "chatgpt_image", "gemini_image", "anime", "cinematic", "custom"],
                    {"default": "photoreal"},
                ),
                "custom_system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Extra system guidance; strongest when model_style=custom"},
                ),
                "style_modifier": (
                    "STRING",
                    {"multiline": False, "default": "", "tooltip": "Optional style, lens, artist, or mood hint"},
                ),
                "locked_prefix": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Text copied verbatim at the start of the positive prompt"},
                ),
                "negative_base": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Required negative text to include"},
                ),
                "num_predict": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 4096, "step": 64},
                ),
                "num_ctx": (
                    "INT",
                    {"default": 8192, "min": 2048, "max": 32768, "step": 1024},
                ),
                "temperature_override": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "use_model_temperature": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "stream_to_console": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, concept: str, model_style: str = "photoreal", style_modifier: str = "", locked_prefix: str = "", negative_base: str = "", custom_system_prompt: str = "", **_):
        return f"{concept}|{model_style}|{style_modifier}|{locked_prefix}|{negative_base}|{custom_system_prompt}"

    def build_single_prompt(
        self,
        ollama_model: Dict[str, Any],
        concept: str,
        model_style: str = "photoreal",
        custom_system_prompt: str = "",
        style_modifier: str = "",
        locked_prefix: str = "",
        negative_base: str = "",
        num_predict: int = 1024,
        num_ctx: int = 8192,
        temperature_override: float = 0.45,
        use_model_temperature: bool = False,
        stream_to_console: bool = False,
    ) -> tuple[str, str, str, str, str]:
        base_url = ollama_model["base_url"]
        model = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")
        temperature = ollama_model.get("temperature", 0.7) if use_model_temperature else temperature_override

        style_note = "" if model_style == "custom" else f"Target style/model family: {model_style}."
        custom_note = custom_system_prompt.strip()
        system = "\n\n".join(part for part in [_SINGLE_PROMPT_SYSTEM, style_note, custom_note] if part)

        user_parts = [f"Concept:\n{concept.strip()}"]
        if style_modifier.strip():
            user_parts.append(f"Style modifier:\n{style_modifier.strip()}")
        if locked_prefix.strip():
            user_parts.append(f"Locked prefix, copy verbatim at the start of positive_prompt:\n{locked_prefix.strip()}")
        if negative_base.strip():
            user_parts.append(f"Negative base, include in negative_prompt:\n{negative_base.strip()}")
        user_parts.append("Return exactly one JSON object. No markdown. No list.")
        user_input = "\n\n".join(user_parts)

        tokens: list[str] = []
        try:
            for token in generate_stream(
                base_url=base_url,
                model=model,
                prompt=user_input,
                system=system,
                temperature=temperature,
                top_p=ollama_model.get("top_p", 0.9),
                top_k=ollama_model.get("top_k", 40),
                repeat_penalty=ollama_model.get("repeat_penalty", 1.1),
                num_predict=num_predict,
                num_ctx=num_ctx,
                seed=ollama_model.get("seed", -1),
                response_format="json",
                think=False,
                keep_alive=keep_alive,
            ):
                tokens.append(token)
                if stream_to_console:
                    sys.stdout.write(token)
                    sys.stdout.flush()
        except OllamaConnectionError as exc:
            raise RuntimeError(f"Ollama single prompt generation failed: {exc}") from exc

        raw = "".join(tokens).strip()
        if not raw:
            raise RuntimeError("Model returned empty output.")

        positive, negative, tags, summary = _parse_single_prompt_output(raw)
        if locked_prefix.strip() and positive and not positive.startswith(locked_prefix.strip()):
            positive = _append_smart(locked_prefix.strip(), positive)
        if negative_base.strip():
            negative = _append_smart(negative_base.strip(), negative)
        return positive, negative, tags, summary, raw