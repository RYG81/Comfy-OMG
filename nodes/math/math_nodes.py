"""
OllamaNodes - Math & Logic Nodes
Arithmetic, trigonometry, comparisons, boolean logic, and expression evaluation.
"""

import math
import random
import re


class OllamaNodesMathOp:
    """Basic arithmetic operations."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "b": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power", "min", "max"],),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("result_float", "result_int")
    FUNCTION = "calculate"
    CATEGORY = "Ollama-Magic-Nodes/Math"
    DESCRIPTION = "Basic arithmetic: add, subtract, multiply, divide, modulo, power, min, max."

    def calculate(self, a, b, operation):
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            result = a / b if b != 0 else float('inf')
        elif operation == "modulo":
            result = a % b if b != 0 else 0
        elif operation == "power":
            result = a ** b
        elif operation == "min":
            result = min(a, b)
        elif operation == "max":
            result = max(a, b)
        else:
            result = 0
        return (float(result), int(result))


class OllamaNodesMathAdvanced:
    """Advanced mathematical functions."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "function": ([
                    "sin", "cos", "tan", "asin", "acos", "atan",
                    "sqrt", "abs", "floor", "ceil", "round",
                    "log", "log2", "log10", "exp",
                    "degrees", "radians",
                    "sign", "reciprocal"
                ],),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("result_float", "result_int")
    FUNCTION = "calculate"
    CATEGORY = "Ollama-Magic-Nodes/Math"
    DESCRIPTION = "Advanced math: trigonometry, logarithms, rounding, and more."

    def calculate(self, value, function):
        try:
            func_map = {
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "asin": math.asin, "acos": math.acos, "atan": math.atan,
                "sqrt": math.sqrt, "abs": abs, "floor": math.floor,
                "ceil": math.ceil, "round": round,
                "log": math.log, "log2": math.log2, "log10": math.log10,
                "exp": math.exp, "degrees": math.degrees, "radians": math.radians,
                "sign": lambda x: (1 if x > 0 else (-1 if x < 0 else 0)),
                "reciprocal": lambda x: 1.0 / x if x != 0 else float('inf'),
            }
            result = func_map[function](value)
        except (ValueError, OverflowError, ZeroDivisionError):
            result = 0.0
        return (float(result), int(result))


class OllamaNodesMathMap:
    """Map a value from one range to another."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.5, "min": -1e10, "max": 1e10, "step": 0.001}),
                "in_min": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "in_max": ("FLOAT", {"default": 1.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "out_max": ("FLOAT", {"default": 100.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "clamp": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("result_float", "result_int")
    FUNCTION = "map_value"
    CATEGORY = "Ollama-Magic-Nodes/Math"
    DESCRIPTION = "Map/remap a value from one range to another, with optional clamping."

    def map_value(self, value, in_min, in_max, out_min, out_max, clamp):
        if in_max == in_min:
            result = out_min
        else:
            normalized = (value - in_min) / (in_max - in_min)
            result = out_min + normalized * (out_max - out_min)
        
        if clamp:
            result = max(min(out_min, out_max), min(max(out_min, out_max), result))
        
        return (float(result), int(result))


class OllamaNodesMathRandom:
    """Random number generation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -1e10, "max": 1e10}),
                "mode": (["float", "int"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("result_float", "result_int")
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/Math"
    DESCRIPTION = "Generate random numbers (float or int) within a specified range with seed control."

    def generate(self, min_value, max_value, mode, seed):
        random.seed(seed)
        if mode == "float":
            result = random.uniform(min_value, max_value)
        else:
            result = random.randint(int(min_value), int(max_value))
        return (float(result), int(result))


class OllamaNodesMathExpression:
    """Evaluate mathematical expressions. JS widget provides expression editor."""
    
    SAFE_FUNCS = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
        'sqrt': math.sqrt, 'abs': abs, 'pow': pow,
        'floor': math.floor, 'ceil': math.ceil, 'round': round,
        'log': math.log, 'log2': math.log2, 'log10': math.log10,
        'exp': math.exp, 'min': min, 'max': max,
        'pi': math.pi, 'e': math.e, 'tau': math.tau,
        'clamp': lambda x, lo, hi: max(lo, min(hi, x)),
        'lerp': lambda a, b, t: a + (b - a) * t,
        'step': lambda edge, x: 0.0 if x < edge else 1.0,
        'smoothstep': lambda e0, e1, x: 0.0 if x <= e0 else (1.0 if x >= e1 else (lambda t: t*t*(3-2*t))((x-e0)/(e1-e0))),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "sin(x * pi) * 0.5 + 0.5", "multiline": False}),
                "x": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "y": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "z": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("result_float", "result_int")
    FUNCTION = "evaluate"
    CATEGORY = "Ollama-Magic-Nodes/Math"
    DESCRIPTION = "Evaluate math expressions with variables x, y, z. Supports sin, cos, sqrt, pi, lerp, clamp, etc."

    def evaluate(self, expression, x, y, z):
        safe_dict = dict(self.SAFE_FUNCS)
        safe_dict.update({'x': x, 'y': y, 'z': z})
        
        try:
            # Only allow safe characters
            clean = re.sub(r'[^a-zA-Z0-9_.+\-*/%()\s,]', '', expression)
            result = eval(clean, {"__builtins__": {}}, safe_dict)
            result = float(result)
        except Exception:
            result = 0.0
        
        return (result, int(result))


class OllamaNodesCompare:
    """Compare two values."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "b": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.001}),
                "comparison": (["equal", "not_equal", "greater", "greater_equal", "less", "less_equal"],),
                "tolerance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.0001}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("result", "result_int")
    FUNCTION = "compare"
    CATEGORY = "Ollama-Magic-Nodes/Math"
    DESCRIPTION = "Compare two values with configurable tolerance for float equality."

    def compare(self, a, b, comparison, tolerance):
        if comparison == "equal":
            result = abs(a - b) <= tolerance
        elif comparison == "not_equal":
            result = abs(a - b) > tolerance
        elif comparison == "greater":
            result = a > b
        elif comparison == "greater_equal":
            result = a >= b
        elif comparison == "less":
            result = a < b
        elif comparison == "less_equal":
            result = a <= b
        else:
            result = False
        return (result, int(result))


class OllamaNodesLogicGate:
    """Boolean logic operations."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("BOOLEAN", {"default": False}),
                "gate": (["AND", "OR", "NOT", "XOR", "NAND", "NOR", "XNOR"],),
            },
            "optional": {
                "b": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("result", "result_int")
    FUNCTION = "logic"
    CATEGORY = "Ollama-Magic-Nodes/Math"
    DESCRIPTION = "Boolean logic gates: AND, OR, NOT, XOR, NAND, NOR, XNOR."

    def logic(self, a, gate, b=False):
        if gate == "AND":
            result = a and b
        elif gate == "OR":
            result = a or b
        elif gate == "NOT":
            result = not a
        elif gate == "XOR":
            result = a ^ b
        elif gate == "NAND":
            result = not (a and b)
        elif gate == "NOR":
            result = not (a or b)
        elif gate == "XNOR":
            result = not (a ^ b)
        else:
            result = False
        return (result, int(result))


class OllamaNodesCounter:
    """Auto-incrementing counter with reset capability."""
    
    count_state = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -1000000, "max": 1000000}),
                "step": ("INT", {"default": 1, "min": -1000, "max": 1000}),
                "reset": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("count", "count_float")
    FUNCTION = "count"
    CATEGORY = "Ollama-Magic-Nodes/Math"
    DESCRIPTION = "Auto-incrementing counter. Increments each execution, can reset."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def count(self, start, step, reset, unique_id=None):
        key = str(unique_id) if unique_id else "default"
        
        if reset or key not in OllamaNodesCounter.count_state:
            OllamaNodesCounter.count_state[key] = start
        else:
            OllamaNodesCounter.count_state[key] += step
        
        val = OllamaNodesCounter.count_state[key]
        return (val, float(val))


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesMathOp": OllamaNodesMathOp,
    "OllamaNodesMathAdvanced": OllamaNodesMathAdvanced,
    "OllamaNodesMathMap": OllamaNodesMathMap,
    "OllamaNodesMathRandom": OllamaNodesMathRandom,
    "OllamaNodesMathExpression": OllamaNodesMathExpression,
    "OllamaNodesCompare": OllamaNodesCompare,
    "OllamaNodesLogicGate": OllamaNodesLogicGate,
    "OllamaNodesCounter": OllamaNodesCounter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesMathOp": "🔢 Math Operation",
    "OllamaNodesMathAdvanced": "🔢 Math Advanced",
    "OllamaNodesMathMap": "🔢 Math Map Range",
    "OllamaNodesMathRandom": "🔢 Random Number",
    "OllamaNodesMathExpression": "🔢 Math Expression",
    "OllamaNodesCompare": "🔢 Compare",
    "OllamaNodesLogicGate": "🔢 Logic Gate",
    "OllamaNodesCounter": "🔢 Counter",
}
