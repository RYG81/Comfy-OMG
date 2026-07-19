"""
OllamaNodes - Text & Prompt Manipulation Nodes
String operations, prompt engineering, and template systems.
"""

import re
import random
import json


class OllamaNodesTextConcat:
    """Concatenate multiple strings with a separator."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_a": ("STRING", {"default": "", "multiline": True}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {
                "text_b": ("STRING", {"default": "", "multiline": True}),
                "text_c": ("STRING", {"default": "", "multiline": True}),
                "text_d": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concat"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Concatenate up to 4 text strings with a configurable separator."

    def concat(self, text_a, separator, text_b="", text_c="", text_d=""):
        parts = [t for t in [text_a, text_b, text_c, text_d] if t.strip()]
        return (separator.join(parts),)


class OllamaNodesTextReplace:
    """Find and replace text."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "find": ("STRING", {"default": ""}),
                "replace_with": ("STRING", {"default": ""}),
                "use_regex": ("BOOLEAN", {"default": False}),
                "case_sensitive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "replacements")
    FUNCTION = "replace"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Find and replace text with optional regex and case sensitivity support."

    def replace(self, text, find, replace_with, use_regex, case_sensitive):
        if not find:
            return (text, 0)
        
        if use_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            result, count = re.subn(find, replace_with, text, flags=flags)
        else:
            if case_sensitive:
                count = text.count(find)
                result = text.replace(find, replace_with)
            else:
                pattern = re.escape(find)
                result, count = re.subn(pattern, replace_with, text, flags=re.IGNORECASE)
        
        return (result, count)


class OllamaNodesTextTemplate:
    """Template string with variable substitution."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"default": "A {quality} photo of {subject}", "multiline": True}),
            },
            "optional": {
                "var_1_name": ("STRING", {"default": "subject"}),
                "var_1_value": ("STRING", {"default": ""}),
                "var_2_name": ("STRING", {"default": "quality"}),
                "var_2_value": ("STRING", {"default": ""}),
                "var_3_name": ("STRING", {"default": "style"}),
                "var_3_value": ("STRING", {"default": ""}),
                "var_4_name": ("STRING", {"default": "extra"}),
                "var_4_value": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "apply_template"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Apply template variables: use {variable_name} in template, define up to 4 variables."

    def apply_template(self, template, var_1_name="", var_1_value="", var_2_name="", var_2_value="",
                       var_3_name="", var_3_value="", var_4_name="", var_4_value=""):
        result = template
        for name, value in [(var_1_name, var_1_value), (var_2_name, var_2_value),
                            (var_3_name, var_3_value), (var_4_name, var_4_value)]:
            if name:
                result = result.replace(f"{{{name}}}", value)
        return (result,)


class OllamaNodesTextSwitch:
    """Switch between text values based on a boolean condition."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
                "text_true": ("STRING", {"default": "", "multiline": True}),
                "text_false": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "switch"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Output one of two text values based on a boolean condition."

    def switch(self, condition, text_true, text_false):
        return (text_true if condition else text_false,)


class OllamaNodesTextList:
    """Create and manage a list of text items."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "item1\nitem2\nitem3", "multiline": True}),
                "separator": ("STRING", {"default": "\\n"}),
                "operation": (["get_all", "get_index", "get_random", "count", "sort", "reverse", "unique", "shuffle"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("result", "count")
    FUNCTION = "list_op"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Text list operations: get by index, random pick, count, sort, reverse, unique, shuffle."

    def list_op(self, text, separator, operation, index, seed):
        sep = separator.replace("\\n", "\n").replace("\\t", "\t")
        items = [item.strip() for item in text.split(sep) if item.strip()]
        count = len(items)
        
        if operation == "get_all":
            return (sep.join(items), count)
        elif operation == "get_index":
            idx = min(index, count - 1) if count > 0 else 0
            return (items[idx] if count > 0 else "", count)
        elif operation == "get_random":
            random.seed(seed)
            return (random.choice(items) if count > 0 else "", count)
        elif operation == "count":
            return (str(count), count)
        elif operation == "sort":
            items.sort()
            return (sep.join(items), count)
        elif operation == "reverse":
            items.reverse()
            return (sep.join(items), count)
        elif operation == "unique":
            seen = set()
            unique_items = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)
            return (sep.join(unique_items), len(unique_items))
        elif operation == "shuffle":
            random.seed(seed)
            random.shuffle(items)
            return (sep.join(items), count)
        
        return (text, count)


class OllamaNodesTextLength:
    """Get string length and word count."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("char_count", "word_count", "line_count")
    FUNCTION = "get_length"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Get character count, word count, and line count of a text string."

    def get_length(self, text):
        char_count = len(text)
        word_count = len(text.split()) if text.strip() else 0
        line_count = len(text.splitlines()) if text.strip() else 0
        return (char_count, word_count, line_count)


class OllamaNodesTextCase:
    """Change text case."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "case": (["uppercase", "lowercase", "title_case", "sentence_case", "swap_case", "capitalize_words"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "change_case"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Change text case: uppercase, lowercase, title case, sentence case, etc."

    def change_case(self, text, case):
        if case == "uppercase":
            return (text.upper(),)
        elif case == "lowercase":
            return (text.lower(),)
        elif case == "title_case":
            return (text.title(),)
        elif case == "sentence_case":
            sentences = re.split(r'([.!?]\s*)', text)
            result = ''
            for i, s in enumerate(sentences):
                if i % 2 == 0 and s:
                    result += s[0].upper() + s[1:].lower() if len(s) > 0 else s
                else:
                    result += s
            return (result,)
        elif case == "swap_case":
            return (text.swapcase(),)
        elif case == "capitalize_words":
            return (' '.join(w.capitalize() for w in text.split()),)
        return (text,)


class OllamaNodesTextRegex:
    """Regex operations on text."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "pattern": ("STRING", {"default": ""}),
                "operation": (["find_all", "find_first", "replace", "split", "test"],),
                "replacement": ("STRING", {"default": ""}),
                "flags": (["none", "ignorecase", "multiline", "dotall"],),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("result", "matched")
    FUNCTION = "regex_op"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Regex operations: find all/first matches, replace, split, or test pattern."

    def regex_op(self, text, pattern, operation, replacement, flags):
        if not pattern:
            return (text, False)
        
        flag_map = {
            "none": 0,
            "ignorecase": re.IGNORECASE,
            "multiline": re.MULTILINE,
            "dotall": re.DOTALL,
        }
        re_flags = flag_map.get(flags, 0)
        
        try:
            if operation == "find_all":
                matches = re.findall(pattern, text, re_flags)
                return ("\n".join(matches) if matches else "", len(matches) > 0)
            elif operation == "find_first":
                match = re.search(pattern, text, re_flags)
                return (match.group() if match else "", match is not None)
            elif operation == "replace":
                result = re.sub(pattern, replacement, text, flags=re_flags)
                return (result, result != text)
            elif operation == "split":
                parts = re.split(pattern, text, flags=re_flags)
                return ("\n".join(parts), len(parts) > 1)
            elif operation == "test":
                match = re.search(pattern, text, re_flags)
                return (str(match is not None), match is not None)
        except re.error as e:
            return (f"Regex error: {str(e)}", False)
        
        return (text, False)


class OllamaNodesPromptWeight:
    """Adjust prompt token weights. JS widget provides visual per-token weight sliders."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Enter your prompt..."}),
                "global_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "weight_data": ("STRING", {"default": "{}", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("weighted_prompt",)
    FUNCTION = "apply_weights"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Apply weights to prompt tokens. Use the interactive JS widget to adjust per-token weights visually."

    def apply_weights(self, prompt, global_weight, weight_data):
        if not prompt.strip():
            return ("",)
        
        try:
            weights = json.loads(weight_data)
        except (json.JSONDecodeError, TypeError):
            weights = {}
        
        if not weights and global_weight == 1.0:
            return (prompt,)
        
        # Apply global weight
        if global_weight != 1.0 and not weights:
            return (f"({prompt}:{global_weight:.2f})",)
        
        # Apply per-token weights from the JS widget
        tokens = [t.strip() for t in prompt.split(',')]
        result_tokens = []
        for i, token in enumerate(tokens):
            if not token:
                continue
            w = weights.get(str(i), weights.get(token.strip(), 1.0))
            w = w * global_weight
            if abs(w - 1.0) > 0.01:
                result_tokens.append(f"({token}:{w:.2f})")
            else:
                result_tokens.append(token)
        
        return (", ".join(result_tokens),)


class OllamaNodesPromptStyler:
    """Apply style templates to prompts."""
    
    STYLES = {
        "none": {"prefix": "", "suffix": ""},
        "photorealistic": {"prefix": "photorealistic, highly detailed, ", "suffix": ", 8k uhd, dslr, high quality, film grain"},
        "cinematic": {"prefix": "cinematic shot, ", "suffix": ", dramatic lighting, depth of field, movie still, anamorphic"},
        "anime": {"prefix": "anime style, ", "suffix": ", trending on pixiv, detailed anime illustration, vibrant colors"},
        "oil_painting": {"prefix": "oil painting, ", "suffix": ", masterful brushstrokes, canvas texture, art gallery, fine art"},
        "watercolor": {"prefix": "watercolor painting, ", "suffix": ", soft washes, paper texture, delicate, translucent"},
        "digital_art": {"prefix": "digital art, ", "suffix": ", trending on artstation, highly detailed, vibrant, concept art"},
        "3d_render": {"prefix": "3D render, ", "suffix": ", octane render, volumetric lighting, ray tracing, realistic materials"},
        "pixel_art": {"prefix": "pixel art, ", "suffix": ", 16-bit, retro game art style, pixelated, sprite"},
        "pencil_sketch": {"prefix": "pencil sketch, ", "suffix": ", graphite drawing, detailed linework, cross-hatching, paper texture"},
        "comic_book": {"prefix": "comic book style, ", "suffix": ", bold outlines, halftone dots, dynamic composition, vibrant"},
        "minimalist": {"prefix": "minimalist, ", "suffix": ", clean lines, simple, elegant, white space, modern design"},
        "fantasy": {"prefix": "fantasy art, ", "suffix": ", magical, ethereal, mystical atmosphere, epic composition"},
        "horror": {"prefix": "horror art, ", "suffix": ", dark atmosphere, eerie, unsettling, dramatic shadows, foreboding"},
        "vintage": {"prefix": "vintage, ", "suffix": ", retro, nostalgic, aged, classic, muted colors, old photograph"},
        "neon": {"prefix": "neon lights, ", "suffix": ", cyberpunk, glowing, synthwave, dark background, vibrant neon colors"},
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "style": (list(cls.STYLES.keys()),),
                "apply_prefix": ("BOOLEAN", {"default": True}),
                "apply_suffix": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("styled_prompt",)
    FUNCTION = "apply_style"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Apply art style templates to your prompt with configurable prefix and suffix."

    def apply_style(self, prompt, style, apply_prefix, apply_suffix):
        style_data = self.STYLES.get(style, self.STYLES["none"])
        prefix = style_data["prefix"] if apply_prefix else ""
        suffix = style_data["suffix"] if apply_suffix else ""
        return (f"{prefix}{prompt}{suffix}",)


class OllamaNodesPromptCombine:
    """Combine positive and negative prompts intelligently."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"default": "", "multiline": True, "placeholder": "Positive prompt..."}),
                "negative": ("STRING", {"default": "", "multiline": True, "placeholder": "Negative prompt..."}),
                "add_quality_positive": ("BOOLEAN", {"default": False}),
                "add_quality_negative": ("BOOLEAN", {"default": False}),
                "quality_preset": (["standard", "high", "ultra"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "combine"
    CATEGORY = "Ollama-Magic-Nodes/Text"
    DESCRIPTION = "Combine prompts with optional quality presets for positive and negative."

    QUALITY_PRESETS = {
        "standard": {
            "positive": "best quality, detailed",
            "negative": "worst quality, low quality, blurry"
        },
        "high": {
            "positive": "masterpiece, best quality, highly detailed, sharp focus",
            "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy"
        },
        "ultra": {
            "positive": "masterpiece, best quality, ultra detailed, sharp focus, high resolution, 8k",
            "negative": "worst quality, low quality, normal quality, blurry, deformed, ugly, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, watermark, text"
        }
    }

    def combine(self, positive, negative, add_quality_positive, add_quality_negative, quality_preset):
        preset = self.QUALITY_PRESETS.get(quality_preset, self.QUALITY_PRESETS["standard"])
        
        pos_parts = [positive] if positive.strip() else []
        neg_parts = [negative] if negative.strip() else []
        
        if add_quality_positive:
            pos_parts.append(preset["positive"])
        if add_quality_negative:
            neg_parts.append(preset["negative"])
        
        return (", ".join(pos_parts), ", ".join(neg_parts))


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesTextConcat": OllamaNodesTextConcat,
    "OllamaNodesTextReplace": OllamaNodesTextReplace,
    "OllamaNodesTextTemplate": OllamaNodesTextTemplate,
    "OllamaNodesTextSwitch": OllamaNodesTextSwitch,
    "OllamaNodesTextList": OllamaNodesTextList,
    "OllamaNodesTextLength": OllamaNodesTextLength,
    "OllamaNodesTextCase": OllamaNodesTextCase,
    "OllamaNodesTextRegex": OllamaNodesTextRegex,
    "OllamaNodesPromptWeight": OllamaNodesPromptWeight,
    "OllamaNodesPromptStyler": OllamaNodesPromptStyler,
    "OllamaNodesPromptCombine": OllamaNodesPromptCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesTextConcat": "📝 Text Concat",
    "OllamaNodesTextReplace": "📝 Text Replace",
    "OllamaNodesTextTemplate": "📝 Text Template",
    "OllamaNodesTextSwitch": "📝 Text Switch",
    "OllamaNodesTextList": "📝 Text List",
    "OllamaNodesTextLength": "📝 Text Length",
    "OllamaNodesTextCase": "📝 Text Case",
    "OllamaNodesTextRegex": "📝 Text Regex",
    "OllamaNodesPromptWeight": "📝 Prompt Weight",
    "OllamaNodesPromptStyler": "📝 Prompt Styler",
    "OllamaNodesPromptCombine": "📝 Prompt Combine",
}
