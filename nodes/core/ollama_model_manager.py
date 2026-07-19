"""
ollama_model_manager.py
─────────────────────────────────────────────────────────────────────────
🦙 Ollama Model Manager

Administrative node for managing local Ollama models without leaving
ComfyUI.  Lets you:

  • List installed models with size / quant info
  • Pull (download) a new model from the Ollama registry
  • Delete a model from disk
  • Show detailed model info (architecture, parameters, licence)
  • List currently running (VRAM-loaded) models

This is an OUTPUT_NODE — it performs side-effects and shows a
formatted status report in the STRING output.

Outputs
  STRING  – formatted status / result report
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys

from .ollama_client import (
    get_installed_models,
    get_running_models,
    get_model_info,
    pull_model_stream,
    delete_model,
    copy_model,
    check_ollama_alive,
    OllamaConnectionError,
)

_DEFAULT_URL = "http://localhost:11434"


def _fmt_size(n_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


class OllamaModelManager:

    CATEGORY = "Ollama-Magic-Nodes/Core"
    FUNCTION     = "manage"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_url": (
                    "STRING",
                    {"default": _DEFAULT_URL, "multiline": False},
                ),
                "action": (
                    ["list_installed", "list_running", "show_info", "pull", "delete", "copy"],
                    {"default": "list_installed"},
                ),
            },
            "optional": {
                "model_name": (
                    "STRING",
                    {
                        "multiline":   False,
                        "default":     "",
                        "description": (
                            "Model name for: show_info / pull / delete / copy. "
                            "For 'pull' use registry names like 'llama3.2:latest'."
                        ),
                    },
                ),
                "destination_name": (
                    "STRING",
                    {
                        "multiline":   False,
                        "default":     "",
                        "description": "New name when using 'copy' action",
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {"default": False, "description": "Include full model metadata in report"},
                ),
                "confirm_delete": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "description": "⚠ Must be TRUE to actually delete a model (safety check)",
                    },
                ),
            },
        }

    # Force re-execution every run so the list is always fresh
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ── Action handlers ───────────────────────────────────────────

    def _list_installed(self, base_url: str, verbose: bool) -> str:
        models = get_installed_models(base_url)
        if not models:
            return "No models installed."

        lines = [f"{'MODEL':<40} {'SIZE':>8}  {'PARAMS':>10}  {'QUANT':>12}  FAMILY"]
        lines.append("─" * 90)
        for m in models:
            d       = m.get("details", {})
            name    = m.get("name", "?")
            size    = _fmt_size(m.get("size", 0))
            params  = d.get("parameter_size", "?")
            quant   = d.get("quantization_level", "?")
            family  = d.get("family", "?")
            lines.append(f"{name:<40} {size:>8}  {params:>10}  {quant:>12}  {family}")
            if verbose:
                mod_at = m.get("modified_at", "")
                digest = m.get("digest", "")[:16]
                lines.append(f"  └ digest={digest}…  modified={mod_at}")

        lines.append(f"\nTotal: {len(models)} model(s)")
        return "\n".join(lines)

    def _list_running(self, base_url: str) -> str:
        running = get_running_models(base_url)
        if not running:
            return "No models currently loaded in VRAM."

        lines = [f"{'MODEL':<40} {'SIZE':>8}  EXPIRES_AT"]
        lines.append("─" * 70)
        for m in running:
            name    = m.get("name", "?")
            size    = _fmt_size(m.get("size", 0))
            expires = m.get("expires_at", "?")
            lines.append(f"{name:<40} {size:>8}  {expires}")
        return "\n".join(lines)

    def _show_info(self, base_url: str, model: str, verbose: bool) -> str:
        if not model:
            return "⚠ Provide a model_name to use show_info."
        info = get_model_info(base_url, model)
        if "error" in info:
            return f"Error: {info['error']}"

        d        = info.get("details", {})
        families = d.get("families", [d.get("family", "?")])
        lines = [
            f"Model        : {model}",
            f"Family       : {', '.join(families)}",
            f"Format       : {d.get('format', '?')}",
            f"Parameters   : {d.get('parameter_size', '?')}",
            f"Quantization : {d.get('quantization_level', '?')}",
        ]
        if verbose:
            mi = info.get("model_info", {})
            for k, v in mi.items():
                lines.append(f"  {k}: {v}")
            if info.get("license"):
                lines.append(f"\nLicense:\n{info['license'][:400]}…")
        return "\n".join(lines)

    def _pull(self, base_url: str, model: str) -> str:
        if not model:
            return "⚠ Provide a model_name to pull."
        print(f"[OllamaModelManager] 🔽 Pulling {model}…", flush=True)
        lines = [f"Pulling {model}…\n"]
        try:
            for event in pull_model_stream(base_url, model):
                status   = event.get("status", "")
                total    = event.get("total",    0)
                completed= event.get("completed", 0)
                if total and total > 0:
                    pct = completed / total * 100
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    msg = f"[{bar}] {pct:5.1f}%  {_fmt_size(completed)}/{_fmt_size(total)}  {status}"
                else:
                    msg = status
                print(f"\r[OllamaModelManager] {msg}", end="", flush=True)
                lines.append(msg)
        except Exception as exc:
            return f"Pull failed: {exc}"
        print(flush=True)
        lines.append(f"\n✅ Pull complete: {model}")
        return "\n".join(lines[-5:] + [lines[-1]])  # keep last few + success line

    def _delete(self, base_url: str, model: str, confirmed: bool) -> str:
        if not model:
            return "⚠ Provide a model_name to delete."
        if not confirmed:
            return (
                f"⚠ Safety: set confirm_delete=True to actually delete '{model}'.\n"
                "This action is IRREVERSIBLE."
            )
        ok = delete_model(base_url, model)
        if ok:
            return f"✅ Model '{model}' deleted successfully."
        return f"❌ Failed to delete '{model}'. Check the server logs."

    def _copy(self, base_url: str, source: str, destination: str) -> str:
        if not source or not destination:
            return "⚠ Provide both model_name (source) and destination_name."
        ok = copy_model(base_url, source, destination)
        if ok:
            return f"✅ '{source}' copied to '{destination}'."
        return f"❌ Failed to copy. Check the server logs."

    # ── Main execution ────────────────────────────────────────────

    def manage(
        self,
        ollama_url:       str,
        action:           str,
        model_name:       str  = "",
        destination_name: str  = "",
        verbose:          bool = False,
        confirm_delete:   bool = False,
    ):
        base_url = ollama_url.rstrip("/")

        if not check_ollama_alive(base_url):
            report = (
                f"❌ Cannot reach Ollama at {base_url}.\n"
                "  Start the server with: ollama serve"
            )
            return (report,)

        model_name       = model_name.strip()
        destination_name = destination_name.strip()

        if action == "list_installed":
            report = self._list_installed(base_url, verbose)
        elif action == "list_running":
            report = self._list_running(base_url)
        elif action == "show_info":
            report = self._show_info(base_url, model_name, verbose)
        elif action == "pull":
            report = self._pull(base_url, model_name)
        elif action == "delete":
            report = self._delete(base_url, model_name, confirm_delete)
        elif action == "copy":
            report = self._copy(base_url, model_name, destination_name)
        else:
            report = f"Unknown action: {action}"

        print(f"\n[OllamaModelManager]\n{report}\n")
        return (report,)
