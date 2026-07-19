"""
OllamaNodes - Animation & Scheduling Nodes
Keyframe curves, wave generators, interpolation, and scheduling.
"""

import math
import json


class OllamaNodesCurveEditor:
    """Keyframe animation curve editor. JS widget provides interactive Bezier curve editor."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "total_frames": ("INT", {"default": 100, "min": 1, "max": 100000}),
                "curve_data": ("STRING", {
                    "default": '[{"x":0,"y":0},{"x":0.5,"y":1},{"x":1,"y":0}]',
                    "multiline": True
                }),
                "min_value": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -1e10, "max": 1e10}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("value", "value_int")
    FUNCTION = "evaluate"
    CATEGORY = "Ollama-Magic-Nodes/Animate"
    DESCRIPTION = "Interactive keyframe curve editor. Define animation curves with drag-and-drop keyframes."

    def evaluate(self, frame, total_frames, curve_data, min_value, max_value):
        try:
            points = json.loads(curve_data)
        except (json.JSONDecodeError, TypeError):
            points = [{"x": 0, "y": 0}, {"x": 1, "y": 1}]
        
        if not points:
            return (min_value, int(min_value))
        
        # Normalize frame position
        t = frame / max(total_frames - 1, 1) if total_frames > 1 else 0
        t = max(0, min(1, t))
        
        # Sort points by x
        points.sort(key=lambda p: p.get('x', 0))
        
        # Find surrounding keyframes
        if t <= points[0].get('x', 0):
            y = points[0].get('y', 0)
        elif t >= points[-1].get('x', 1):
            y = points[-1].get('y', 0)
        else:
            # Linear interpolation between keyframes
            for i in range(len(points) - 1):
                x0 = points[i].get('x', 0)
                x1 = points[i + 1].get('x', 1)
                if x0 <= t <= x1:
                    if x1 == x0:
                        local_t = 0
                    else:
                        local_t = (t - x0) / (x1 - x0)
                    y0 = points[i].get('y', 0)
                    y1 = points[i + 1].get('y', 0)
                    y = y0 + (y1 - y0) * local_t
                    break
            else:
                y = points[-1].get('y', 0)
        
        # Map to value range
        value = min_value + y * (max_value - min_value)
        return (float(value), int(value))


class OllamaNodesWaveGenerator:
    """Generate periodic wave values for animation. JS widget shows wave preview."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "wave_type": (["sine", "cosine", "triangle", "sawtooth", "square", "bounce"],),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 100.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frames_per_cycle": ("INT", {"default": 30, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("value", "value_int")
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/Animate"
    DESCRIPTION = "Generate periodic wave values: sine, cosine, triangle, sawtooth, square, bounce."

    def generate(self, frame, wave_type, frequency, amplitude, offset, phase, frames_per_cycle):
        t = (frame / frames_per_cycle * frequency + phase) % 1.0
        
        if wave_type == "sine":
            value = math.sin(t * 2 * math.pi)
        elif wave_type == "cosine":
            value = math.cos(t * 2 * math.pi)
        elif wave_type == "triangle":
            value = 1.0 - 4.0 * abs(t - 0.5)
        elif wave_type == "sawtooth":
            value = 2.0 * t - 1.0
        elif wave_type == "square":
            value = 1.0 if t < 0.5 else -1.0
        elif wave_type == "bounce":
            value = abs(math.sin(t * math.pi))
        else:
            value = 0.0
        
        result = value * amplitude + offset
        return (float(result), int(result))


class OllamaNodesScheduler:
    """Schedule parameter values over sampling steps."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "total_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "schedule_type": (["linear", "ease_in", "ease_out", "ease_in_out", "constant", "step_function"],),
                "start_value": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01}),
                "end_value": ("FLOAT", {"default": 1.0, "min": -1e10, "max": 1e10, "step": 0.01}),
                "step_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("value", "value_int")
    FUNCTION = "schedule"
    CATEGORY = "Ollama-Magic-Nodes/Animate"
    DESCRIPTION = "Schedule parameter values over steps: linear, ease in/out, constant, or step function."

    def schedule(self, step, total_steps, schedule_type, start_value, end_value, step_threshold):
        t = step / max(total_steps - 1, 1) if total_steps > 1 else 1.0
        t = max(0, min(1, t))
        
        if schedule_type == "linear":
            factor = t
        elif schedule_type == "ease_in":
            factor = t * t
        elif schedule_type == "ease_out":
            factor = 1 - (1 - t) * (1 - t)
        elif schedule_type == "ease_in_out":
            if t < 0.5:
                factor = 2 * t * t
            else:
                factor = 1 - (-2 * t + 2) ** 2 / 2
        elif schedule_type == "constant":
            factor = 0.0
        elif schedule_type == "step_function":
            factor = 0.0 if t < step_threshold else 1.0
        else:
            factor = t
        
        value = start_value + factor * (end_value - start_value)
        return (float(value), int(value))


class OllamaNodesInterpolate:
    """Interpolate between two values with various easing functions."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value_a": ("FLOAT", {"default": 0.0, "min": -1e10, "max": 1e10, "step": 0.01}),
                "value_b": ("FLOAT", {"default": 1.0, "min": -1e10, "max": 1e10, "step": 0.01}),
                "t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "easing": (["linear", "ease_in_quad", "ease_out_quad", "ease_in_out_quad",
                           "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic",
                           "ease_in_sine", "ease_out_sine", "ease_in_out_sine",
                           "ease_in_expo", "ease_out_expo", "ease_in_out_expo",
                           "ease_in_elastic", "ease_out_elastic",
                           "ease_in_bounce", "ease_out_bounce"],),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("value", "value_int")
    FUNCTION = "interpolate"
    CATEGORY = "Ollama-Magic-Nodes/Animate"
    DESCRIPTION = "Interpolate between two values with 17 easing functions."

    def _ease(self, t, easing):
        if easing == "linear":
            return t
        elif easing == "ease_in_quad":
            return t * t
        elif easing == "ease_out_quad":
            return 1 - (1 - t) ** 2
        elif easing == "ease_in_out_quad":
            return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
        elif easing == "ease_in_cubic":
            return t ** 3
        elif easing == "ease_out_cubic":
            return 1 - (1 - t) ** 3
        elif easing == "ease_in_out_cubic":
            return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
        elif easing == "ease_in_sine":
            return 1 - math.cos(t * math.pi / 2)
        elif easing == "ease_out_sine":
            return math.sin(t * math.pi / 2)
        elif easing == "ease_in_out_sine":
            return -(math.cos(math.pi * t) - 1) / 2
        elif easing == "ease_in_expo":
            return 0 if t == 0 else 2 ** (10 * t - 10)
        elif easing == "ease_out_expo":
            return 1 if t == 1 else 1 - 2 ** (-10 * t)
        elif easing == "ease_in_out_expo":
            if t == 0: return 0
            if t == 1: return 1
            return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2
        elif easing == "ease_in_elastic":
            if t == 0: return 0
            if t == 1: return 1
            return -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * (2 * math.pi / 3))
        elif easing == "ease_out_elastic":
            if t == 0: return 0
            if t == 1: return 1
            return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi / 3)) + 1
        elif easing == "ease_out_bounce":
            if t < 1 / 2.75:
                return 7.5625 * t * t
            elif t < 2 / 2.75:
                t -= 1.5 / 2.75
                return 7.5625 * t * t + 0.75
            elif t < 2.5 / 2.75:
                t -= 2.25 / 2.75
                return 7.5625 * t * t + 0.9375
            else:
                t -= 2.625 / 2.75
                return 7.5625 * t * t + 0.984375
        elif easing == "ease_in_bounce":
            return 1 - self._ease(1 - t, "ease_out_bounce")
        return t

    def interpolate(self, value_a, value_b, t, easing):
        eased_t = self._ease(max(0, min(1, t)), easing)
        value = value_a + (value_b - value_a) * eased_t
        return (float(value), int(value))


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesCurveEditor": OllamaNodesCurveEditor,
    "OllamaNodesWaveGenerator": OllamaNodesWaveGenerator,
    "OllamaNodesScheduler": OllamaNodesScheduler,
    "OllamaNodesInterpolate": OllamaNodesInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesCurveEditor": "🎬 Curve Editor",
    "OllamaNodesWaveGenerator": "🎬 Wave Generator",
    "OllamaNodesScheduler": "🎬 Scheduler",
    "OllamaNodesInterpolate": "🎬 Interpolate",
}
