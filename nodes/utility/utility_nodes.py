"""
OllamaNodes - Utility Nodes
Debug, seed management, resolution presets, timers, notes, and metadata.
"""

import time
import os
import json
import hashlib


class OllamaNodesSeed:
    """Advanced seed generator with randomize/increment/fixed modes."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mode": (["fixed", "random", "increment", "decrement"],),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate_seed"
    CATEGORY = "Ollama-Magic-Nodes/Utility"
    DESCRIPTION = "Seed generator with modes: fixed, random, increment, decrement."

    @classmethod
    def IS_CHANGED(cls, seed, mode):
        if mode != "fixed":
            return float("NaN")
        return seed

    def generate_seed(self, seed, mode):
        import random
        if mode == "random":
            return (random.randint(0, 0xffffffffffffffff),)
        elif mode == "increment":
            return ((seed + 1) % (0xffffffffffffffff + 1),)
        elif mode == "decrement":
            return ((seed - 1) % (0xffffffffffffffff + 1),)
        return (seed,)


class OllamaNodesResolution:
    """Resolution presets for common model formats."""
    
    PRESETS = {
        "SD 1.5 - 512x512": (512, 512),
        "SD 1.5 - 512x768": (512, 768),
        "SD 1.5 - 768x512": (768, 512),
        "SD 1.5 - 768x768": (768, 768),
        "SDXL - 1024x1024": (1024, 1024),
        "SDXL - 896x1152": (896, 1152),
        "SDXL - 1152x896": (1152, 896),
        "SDXL - 832x1216": (832, 1216),
        "SDXL - 1216x832": (1216, 832),
        "SDXL - 768x1344": (768, 1344),
        "SDXL - 1344x768": (1344, 768),
        "SD3 - 1024x1024": (1024, 1024),
        "SD3 - 1536x1024": (1536, 1024),
        "SD3 - 1024x1536": (1024, 1536),
        "Flux - 1024x1024": (1024, 1024),
        "Flux - 1360x768": (1360, 768),
        "Flux - 768x1360": (768, 1360),
        "Custom": (512, 512),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()),),
                "custom_width": ("INT", {"default": 512, "min": 64, "max": 16384, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 16384, "step": 8}),
                "swap_dimensions": ("BOOLEAN", {"default": False}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "resolution_text")
    FUNCTION = "get_resolution"
    CATEGORY = "Ollama-Magic-Nodes/Utility"
    DESCRIPTION = "Quick resolution presets for SD1.5, SDXL, SD3, Flux, or custom. Supports scaling and dimension swap."

    def get_resolution(self, preset, custom_width, custom_height, swap_dimensions, scale_factor):
        if preset == "Custom":
            w, h = custom_width, custom_height
        else:
            w, h = self.PRESETS.get(preset, (512, 512))
        
        w = int(w * scale_factor)
        h = int(h * scale_factor)
        
        # Ensure divisible by 8
        w = (w // 8) * 8
        h = (h // 8) * 8
        
        if swap_dimensions:
            w, h = h, w
        
        return (w, h, f"{w}x{h}")


class OllamaNodesTimer:
    """Measure execution time between queue events."""
    
    _start_times = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["start", "stop", "lap"],),
                "timer_name": ("STRING", {"default": "default"}),
            },
            "optional": {
                "passthrough": ("*",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "*")
    RETURN_NAMES = ("elapsed_seconds", "formatted_time", "passthrough")
    FUNCTION = "timer"
    CATEGORY = "Ollama-Magic-Nodes/Utility"
    DESCRIPTION = "Measure execution time. Start, stop, or lap a named timer."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def timer(self, operation, timer_name, passthrough=None):
        current = time.time()
        
        if operation == "start":
            OllamaNodesTimer._start_times[timer_name] = current
            return (0.0, "0.000s", passthrough)
        elif operation in ("stop", "lap"):
            start = OllamaNodesTimer._start_times.get(timer_name, current)
            elapsed = current - start
            
            if elapsed < 1:
                formatted = f"{elapsed*1000:.1f}ms"
            elif elapsed < 60:
                formatted = f"{elapsed:.3f}s"
            elif elapsed < 3600:
                mins = int(elapsed // 60)
                secs = elapsed % 60
                formatted = f"{mins}m {secs:.1f}s"
            else:
                hours = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                secs = elapsed % 60
                formatted = f"{hours}h {mins}m {secs:.1f}s"
            
            if operation == "stop":
                OllamaNodesTimer._start_times.pop(timer_name, None)
            
            return (elapsed, formatted, passthrough)
        
        return (0.0, "0.000s", passthrough)


class OllamaNodesNote:
    """Rich text annotation node. JS widget provides markdown editor."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "# Note\nWrite your notes here...", "multiline": True}),
                "color": ("STRING", {"default": "#2a2a3e"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "passthrough"
    CATEGORY = "Ollama-Magic-Nodes/Utility"
    DESCRIPTION = "Rich text note/annotation node. Use for documenting workflow sections."

    def passthrough(self, text, color):
        return (text,)


class OllamaNodesDebug:
    """Debug display — inspect any value type. JS widget shows value details."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "Debug"}),
            },
            "optional": {
                "value_str": ("STRING", {"default": "", "forceInput": True}),
                "value_int": ("INT", {"default": 0, "forceInput": True}),
                "value_float": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "value_bool": ("BOOLEAN", {"default": False, "forceInput": True}),
                "value_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug_info",)
    FUNCTION = "debug"
    CATEGORY = "Ollama-Magic-Nodes/Utility"
    OUTPUT_NODE = True
    DESCRIPTION = "Debug inspector — display the value and type of any connected input."

    def debug(self, label, value_str=None, value_int=None, value_float=None, value_bool=None, value_image=None):
        parts = [f"=== {label} ==="]
        
        if value_str is not None:
            parts.append(f"STRING: {repr(value_str)}")
        if value_int is not None:
            parts.append(f"INT: {value_int}")
        if value_float is not None:
            parts.append(f"FLOAT: {value_float}")
        if value_bool is not None:
            parts.append(f"BOOL: {value_bool}")
        if value_image is not None:
            shape = value_image.shape
            parts.append(f"IMAGE: shape={list(shape)}, dtype={value_image.dtype}")
            parts.append(f"  batch={shape[0]}, H={shape[1]}, W={shape[2]}, C={shape[3]}")
            parts.append(f"  min={value_image.min().item():.4f}, max={value_image.max().item():.4f}")
        
        info = "\n".join(parts)
        print(info)
        return (info,)


class OllamaNodesSaveText:
    """Save text to a file."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "filename": ("STRING", {"default": "output.txt"}),
                "mode": (["overwrite", "append"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "Ollama-Magic-Nodes/Utility"
    OUTPUT_NODE = True
    DESCRIPTION = "Save text content to a file in the ComfyUI output directory."

    def save(self, text, filename, mode):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, filename)
        
        write_mode = 'w' if mode == "overwrite" else 'a'
        with open(filepath, write_mode, encoding='utf-8') as f:
            f.write(text)
            if mode == "append":
                f.write("\n")
        
        return (filepath,)


class OllamaNodesLoadText:
    """Load text from a file."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "input.txt"}),
                "default_text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = "Ollama-Magic-Nodes/Utility"
    DESCRIPTION = "Load text from a file. Returns default_text if file not found."

    def load(self, filename, default_text):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        filepath = os.path.join(input_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return (f.read(),)
        return (default_text,)

    @classmethod
    def IS_CHANGED(cls, filename, default_text):
        return float("NaN")


class OllamaNodesImageInfo:
    """Get image dimensions and metadata."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "batch_size", "channels", "info_text")
    FUNCTION = "get_info"
    CATEGORY = "Ollama-Magic-Nodes/Utility"
    DESCRIPTION = "Get image dimensions: width, height, batch size, channels, and formatted info text."

    def get_info(self, image):
        b, h, w, c = image.shape
        info = f"Size: {w}x{h} | Batch: {b} | Channels: {c} | Dtype: {image.dtype}"
        return (w, h, b, c, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesSeed": OllamaNodesSeed,
    "OllamaNodesResolution": OllamaNodesResolution,
    "OllamaNodesTimer": OllamaNodesTimer,
    "OllamaNodesNote": OllamaNodesNote,
    "OllamaNodesDebug": OllamaNodesDebug,
    "OllamaNodesSaveText": OllamaNodesSaveText,
    "OllamaNodesLoadText": OllamaNodesLoadText,
    "OllamaNodesImageInfo": OllamaNodesImageInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesSeed": "🛠 Seed Generator",
    "OllamaNodesResolution": "🛠 Resolution Preset",
    "OllamaNodesTimer": "🛠 Timer",
    "OllamaNodesNote": "🛠 Note",
    "OllamaNodesDebug": "🛠 Debug Inspector",
    "OllamaNodesSaveText": "🛠 Save Text",
    "OllamaNodesLoadText": "🛠 Load Text",
    "OllamaNodesImageInfo": "🛠 Image Info",
}
