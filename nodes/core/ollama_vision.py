"""
ollama_vision.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama Vision (Multi-modal)

Sends an IMAGE (ComfyUI tensor) together with a text prompt to a
vision-capable Ollama model (llava, bakllava, moondream, llama3.2-vision…).

Images are base64-encoded PNG and passed in the /api/chat images field.

Inputs
  ollama_model – OLLAMA_MODEL pipe  (use a vision model!)
  image        – IMAGE  (ComfyUI tensor batch)
  prompt       – STRING
  system       – STRING  (optional)
  detail_level – low / medium / high  (resizes the image before encoding)

Outputs
  STRING  – model's description / answer
  IMAGE   – pass-through of original image (for chaining)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import sys
import base64

import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    raise ImportError("[OllamaVision] Pillow is required. Run: pip install pillow")

from .ollama_client import chat_stream, OllamaConnectionError

# Maximum long-edge pixels per detail level
_DETAIL_MAX = {"low": 512, "medium": 1024, "high": 2048}


def _tensor_to_base64_png(tensor, max_px: int) -> str:
    """
    Convert a ComfyUI IMAGE tensor (B,H,W,C float32 0-1) to a base64 PNG string.
    Uses the first image in the batch.
    """
    # Take first image in batch
    img_np = tensor[0].cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    pil_img = PILImage.fromarray(img_np, mode="RGB")

    # Resize to keep within max_px on the long edge
    w, h = pil_img.size
    long_edge = max(w, h)
    if long_edge > max_px:
        scale = max_px / long_edge
        pil_img = pil_img.resize(
            (int(w * scale), int(h * scale)), PILImage.LANCZOS
        )

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class OllamaVision:

    CATEGORY = "Ollama-Magic-Nodes/Image"
    FUNCTION      = "analyze_image"
    RETURN_TYPES  = ("STRING", "IMAGE")
    RETURN_NAMES  = ("description", "image_passthrough")
    OUTPUT_NODE   = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_model": ("OLLAMA_MODEL", {}),
                "image": ("IMAGE", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default":   "Describe this image in detail.",
                        "description": "Question or instruction about the image",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default":   (
                            "You are an expert visual analyst. "
                            "Provide accurate, detailed descriptions."
                        ),
                    },
                ),
                "detail_level": (
                    ["low", "medium", "high"],
                    {
                        "default":     "medium",
                        "description": "Image resolution sent to the model (low=512px, medium=1024px, high=2048px)",
                    },
                ),
                "num_predict": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 1},
                ),
                "temperature_override": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01},
                ),
                "stream_to_console": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "output_format": (
                    ["text", "json"],
                    {"default": "text"},
                ),
                # Batch index — useful when the IMAGE tensor contains multiple frames
                "batch_index": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 255, "step": 1,
                        "description": "Which image in the batch to analyse (0 = first)",
                    },
                ),
                "think_mode": (
                    ["off", "on", "low", "medium", "high"],
                    {"default": "off"},
                ),
            },
        }

    def analyze_image(
        self,
        ollama_model:         dict,
        image,                             # torch.Tensor (B,H,W,C)
        prompt:               str,
        system_prompt:        str   = "",
        detail_level:         str   = "medium",
        num_predict:          int   = 1024,
        temperature_override: float = -1.0,
        stream_to_console:    bool  = True,
        output_format:        str   = "text",
        batch_index:          int   = 0,
        think_mode:           str   = "off",
    ):
        base_url   = ollama_model["base_url"]
        model      = ollama_model["model"]
        keep_alive = ollama_model.get("keep_alive", "5m")

        temperature    = (
            temperature_override
            if temperature_override >= 0
            else ollama_model.get("temperature", 0.7)
        )
        top_p          = ollama_model.get("top_p",          0.9)
        top_k          = ollama_model.get("top_k",          40)
        repeat_penalty = ollama_model.get("repeat_penalty", 1.1)
        seed           = ollama_model.get("seed",           -1)
        fmt            = "json" if output_format == "json" else ""
        think = False if think_mode == "off" else (True if think_mode == "on" else think_mode)

        # ── Encode image ──────────────────────────────────────────
        max_px = _DETAIL_MAX.get(detail_level, 1024)
        # Handle batch index safely
        n_images = image.shape[0]
        idx = min(batch_index, n_images - 1)
        single = image[idx : idx + 1]  # keep batch dim
        b64_img = _tensor_to_base64_png(single, max_px)

        # Build chat message with embedded image
        messages = [
            {
                "role":    "user",
                "content": prompt,
                "images":  [b64_img],
            }
        ]

        if stream_to_console:
            print(
                f"\n[OllamaVision] ▶ {model} analysing image "
                f"(detail={detail_level}, idx={idx})…\n",
                flush=True,
            )

        reply_tokens: list[str] = []
        try:
            for token in chat_stream(
                base_url       = base_url,
                model          = model,
                messages       = messages,
                system         = system_prompt,
                temperature    = temperature,
                top_p          = top_p,
                top_k          = top_k,
                repeat_penalty = repeat_penalty,
                num_predict    = num_predict,
                seed           = seed,
                format         = fmt,
                think          = think,
                keep_alive     = keep_alive,
            ):
                reply_tokens.append(token)
                if stream_to_console:
                    sys.stdout.write(token)
                    sys.stdout.flush()

        except OllamaConnectionError as exc:
            raise RuntimeError(str(exc)) from exc

        if stream_to_console:
            print("\n[OllamaVision] ✅ Done.", flush=True)

        description = "".join(reply_tokens)
        return (description, image)
