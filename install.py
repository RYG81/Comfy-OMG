"""
install.py – ComfyUI-OllamaNodes dependency installer
Runs automatically when ComfyUI loads custom nodes via its install hook.
"""
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).parent / "requirements.txt"


def install_requirements():
    if REQUIREMENTS.exists():
        print("[OllamaNodes] Installing / verifying dependencies…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS), "--quiet"]
        )
        print("[OllamaNodes] ✅ Dependencies ready.")


if __name__ == "__main__":
    install_requirements()
