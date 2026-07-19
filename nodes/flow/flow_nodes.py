"""
OllamaNodes - Flow Control Nodes
Conditional routing, type conversion, and data packing.
"""

import json


class OllamaNodesIfElse:
    """Conditional routing — output one of two values based on condition."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
                "if_true": ("STRING", {"default": "", "multiline": True, "forceInput": False}),
                "if_false": ("STRING", {"default": "", "multiline": True, "forceInput": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "route"
    CATEGORY = "Ollama-Magic-Nodes/Flow"
    DESCRIPTION = "Output if_true when condition is True, otherwise output if_false."

    def route(self, condition, if_true, if_false):
        return (if_true if condition else if_false,)


class OllamaNodesIfElseImage:
    """Conditional image routing."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
                "if_true": ("IMAGE",),
                "if_false": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "route"
    CATEGORY = "Ollama-Magic-Nodes/Flow"
    DESCRIPTION = "Output one of two images based on a boolean condition."

    def route(self, condition, if_true, if_false):
        return (if_true if condition else if_false,)


class OllamaNodesAnySwitch:
    """Universal type switch — pass through any value based on condition."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "value_true": ("*",),
                "value_false": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("result",)
    FUNCTION = "switch"
    CATEGORY = "Ollama-Magic-Nodes/Flow"
    DESCRIPTION = "Universal switch: pass through any type based on boolean condition."

    def switch(self, condition, value_true=None, value_false=None):
        return (value_true if condition else value_false,)


class OllamaNodesTypeConvert:
    """Convert between INT, FLOAT, STRING, and BOOLEAN types."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "0"}),
                "target_type": (["int", "float", "string", "boolean"],),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("int_val", "float_val", "string_val", "bool_val")
    FUNCTION = "convert"
    CATEGORY = "Ollama-Magic-Nodes/Flow"
    DESCRIPTION = "Convert a value between INT, FLOAT, STRING, and BOOLEAN types. Outputs all types simultaneously."

    def convert(self, value, target_type):
        try:
            float_val = float(value)
        except (ValueError, TypeError):
            float_val = 0.0
        
        int_val = int(float_val)
        string_val = str(value)
        bool_val = bool(value) if not isinstance(value, str) else (value.lower() not in ('false', '0', '', 'no', 'none'))
        
        return (int_val, float_val, string_val, bool_val)


class OllamaNodesIntToFloat:
    """Simple INT to FLOAT conversion."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float_value",)
    FUNCTION = "convert"
    CATEGORY = "Ollama-Magic-Nodes/Flow"
    DESCRIPTION = "Convert an INT value to FLOAT."

    def convert(self, value):
        return (float(value),)


class OllamaNodesFloatToInt:
    """Simple FLOAT to INT conversion with rounding mode."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10}),
                "mode": (["round", "floor", "ceil", "truncate"],),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int_value",)
    FUNCTION = "convert"
    CATEGORY = "Ollama-Magic-Nodes/Flow"
    DESCRIPTION = "Convert FLOAT to INT with rounding mode (round, floor, ceil, truncate)."

    def convert(self, value, mode):
        import math
        if mode == "round":
            return (round(value),)
        elif mode == "floor":
            return (math.floor(value),)
        elif mode == "ceil":
            return (math.ceil(value),)
        elif mode == "truncate":
            return (int(value),)
        return (int(value),)


class OllamaNodesPack:
    """Pack multiple values into a JSON string bundle."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "key_1": ("STRING", {"default": "a"}),
                "value_1": ("STRING", {"default": ""}),
                "key_2": ("STRING", {"default": "b"}),
                "value_2": ("STRING", {"default": ""}),
                "key_3": ("STRING", {"default": "c"}),
                "value_3": ("STRING", {"default": ""}),
                "key_4": ("STRING", {"default": "d"}),
                "value_4": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bundle",)
    FUNCTION = "pack"
    CATEGORY = "Ollama-Magic-Nodes/Flow"
    DESCRIPTION = "Pack up to 4 key-value pairs into a JSON bundle string."

    def pack(self, key_1="a", value_1="", key_2="b", value_2="", key_3="c", value_3="", key_4="d", value_4=""):
        data = {}
        for k, v in [(key_1, value_1), (key_2, value_2), (key_3, value_3), (key_4, value_4)]:
            if k.strip():
                data[k] = v
        return (json.dumps(data),)


class OllamaNodesUnpack:
    """Unpack values from a JSON string bundle."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle": ("STRING", {"default": "{}"}),
                "key_1": ("STRING", {"default": "a"}),
                "key_2": ("STRING", {"default": "b"}),
                "key_3": ("STRING", {"default": "c"}),
                "key_4": ("STRING", {"default": "d"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("value_1", "value_2", "value_3", "value_4")
    FUNCTION = "unpack"
    CATEGORY = "Ollama-Magic-Nodes/Flow"
    DESCRIPTION = "Unpack values from a JSON bundle by key names."

    def unpack(self, bundle, key_1, key_2, key_3, key_4):
        try:
            data = json.loads(bundle)
        except (json.JSONDecodeError, TypeError):
            data = {}
        
        return (
            str(data.get(key_1, "")),
            str(data.get(key_2, "")),
            str(data.get(key_3, "")),
            str(data.get(key_4, "")),
        )


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesIfElse": OllamaNodesIfElse,
    "OllamaNodesIfElseImage": OllamaNodesIfElseImage,
    "OllamaNodesAnySwitch": OllamaNodesAnySwitch,
    "OllamaNodesTypeConvert": OllamaNodesTypeConvert,
    "OllamaNodesIntToFloat": OllamaNodesIntToFloat,
    "OllamaNodesFloatToInt": OllamaNodesFloatToInt,
    "OllamaNodesPack": OllamaNodesPack,
    "OllamaNodesUnpack": OllamaNodesUnpack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesIfElse": "🔄 If/Else Text",
    "OllamaNodesIfElseImage": "🔄 If/Else Image",
    "OllamaNodesAnySwitch": "🔄 Any Switch",
    "OllamaNodesTypeConvert": "🔄 Type Convert",
    "OllamaNodesIntToFloat": "🔄 Int → Float",
    "OllamaNodesFloatToInt": "🔄 Float → Int",
    "OllamaNodesPack": "🔄 Pack Values",
    "OllamaNodesUnpack": "🔄 Unpack Values",
}
