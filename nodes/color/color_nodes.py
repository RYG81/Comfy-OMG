"""
OllamaNodes - Color Nodes
Color picking, solid colors, gradients, and color harmonies.
"""

import torch
import numpy as np
import colorsys
import math


def hex_to_rgb(hex_color):
    """Convert hex color string to (r, g, b) tuple with values 0-1."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


def rgb_to_hex(r, g, b):
    """Convert (r, g, b) values 0-1 to hex string."""
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


class OllamaNodesColorPicker:
    """Full-featured color picker. JS widget provides HSL wheel, RGB sliders, HEX input."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("STRING", {"default": "#ff6600"}),
                "output_format": (["hex", "rgb_comma", "rgb_0_1"],),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("color_string", "red", "green", "blue", "red_float", "green_float", "blue_float")
    FUNCTION = "pick_color"
    CATEGORY = "Ollama-Magic-Nodes/Color"
    DESCRIPTION = "Interactive color picker with HSL wheel, RGB sliders, and HEX input."

    def pick_color(self, color, output_format):
        r, g, b = hex_to_rgb(color)
        
        if output_format == "hex":
            color_str = rgb_to_hex(r, g, b)
        elif output_format == "rgb_comma":
            color_str = f"{int(r*255)}, {int(g*255)}, {int(b*255)}"
        elif output_format == "rgb_0_1":
            color_str = f"{r:.3f}, {g:.3f}, {b:.3f}"
        else:
            color_str = color
        
        return (color_str, int(r*255), int(g*255), int(b*255), r, g, b)


class OllamaNodesColorToImage:
    """Create a solid color image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("STRING", {"default": "#ff6600"}),
                "width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create"
    CATEGORY = "Ollama-Magic-Nodes/Color"
    DESCRIPTION = "Create a solid color image from a hex color code."

    def create(self, color, width, height):
        r, g, b = hex_to_rgb(color)
        image = torch.zeros(1, height, width, 3)
        image[..., 0] = r
        image[..., 1] = g
        image[..., 2] = b
        return (image,)


class OllamaNodesGradientImage:
    """Create gradient images. JS widget provides interactive gradient editor."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "color_start": ("STRING", {"default": "#000000"}),
                "color_end": ("STRING", {"default": "#ffffff"}),
                "gradient_type": (["linear_horizontal", "linear_vertical", "linear_diagonal",
                                   "radial", "angular", "diamond"],),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "color_mid": ("STRING", {"default": ""}),
                "mid_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_gradient"
    CATEGORY = "Ollama-Magic-Nodes/Color"
    DESCRIPTION = "Create gradient images: linear, radial, angular, or diamond with optional midpoint color."

    def create_gradient(self, width, height, color_start, color_end, gradient_type, center_x, center_y,
                       color_mid="", mid_position=0.5):
        r1, g1, b1 = hex_to_rgb(color_start)
        r2, g2, b2 = hex_to_rgb(color_end)
        has_mid = color_mid.strip() != ""
        
        if has_mid:
            rm, gm, bm = hex_to_rgb(color_mid)
        
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        if gradient_type == "linear_horizontal":
            t = xx
        elif gradient_type == "linear_vertical":
            t = yy
        elif gradient_type == "linear_diagonal":
            t = (xx + yy) / 2.0
        elif gradient_type == "radial":
            dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
            t = np.clip(dist / (np.sqrt(0.5) + 1e-7), 0, 1)
        elif gradient_type == "angular":
            angle = np.arctan2(yy - center_y, xx - center_x)
            t = (angle + np.pi) / (2 * np.pi)
        elif gradient_type == "diamond":
            t = np.clip(np.abs(xx - center_x) + np.abs(yy - center_y), 0, 1)
        else:
            t = xx
        
        if has_mid:
            # Two-segment gradient
            mask = t <= mid_position
            t1 = np.where(mask, t / max(mid_position, 1e-7), 1.0)
            t2 = np.where(~mask, (t - mid_position) / max(1.0 - mid_position, 1e-7), 0.0)
            
            r = np.where(mask, r1 + t1 * (rm - r1), rm + t2 * (r2 - rm))
            g = np.where(mask, g1 + t1 * (gm - g1), gm + t2 * (g2 - gm))
            b = np.where(mask, b1 + t1 * (bm - b1), bm + t2 * (b2 - bm))
        else:
            r = r1 + t * (r2 - r1)
            g = g1 + t * (g2 - g1)
            b = b1 + t * (b2 - b1)
        
        image = np.stack([r, g, b], axis=-1).astype(np.float32)
        image = np.clip(image, 0, 1)
        return (torch.from_numpy(image).unsqueeze(0),)


class OllamaNodesColorHarmonies:
    """Generate color harmonies from a base color. JS widget shows color wheel."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_color": ("STRING", {"default": "#ff6600"}),
                "harmony_type": (["complementary", "analogous", "triadic", "tetradic",
                                  "split_complementary", "monochromatic"],),
                "variation": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("colors_hex", "colors_list", "palette_image")
    FUNCTION = "generate_harmony"
    CATEGORY = "Ollama-Magic-Nodes/Color"
    DESCRIPTION = "Generate color harmonies: complementary, analogous, triadic, tetradic, split-complementary, monochromatic."

    def generate_harmony(self, base_color, harmony_type, variation):
        r, g, b = hex_to_rgb(base_color)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        colors_hsv = [(h, s, v)]  # Base color always first
        
        if harmony_type == "complementary":
            colors_hsv.append(((h + 0.5 + variation) % 1.0, s, v))
        elif harmony_type == "analogous":
            colors_hsv.append(((h + 1/12 + variation) % 1.0, s, v))
            colors_hsv.append(((h - 1/12 + variation) % 1.0, s, v))
        elif harmony_type == "triadic":
            colors_hsv.append(((h + 1/3 + variation) % 1.0, s, v))
            colors_hsv.append(((h + 2/3 + variation) % 1.0, s, v))
        elif harmony_type == "tetradic":
            colors_hsv.append(((h + 0.25 + variation) % 1.0, s, v))
            colors_hsv.append(((h + 0.5 + variation) % 1.0, s, v))
            colors_hsv.append(((h + 0.75 + variation) % 1.0, s, v))
        elif harmony_type == "split_complementary":
            colors_hsv.append(((h + 5/12 + variation) % 1.0, s, v))
            colors_hsv.append(((h + 7/12 + variation) % 1.0, s, v))
        elif harmony_type == "monochromatic":
            colors_hsv.append((h, max(0, s - 0.3), v))
            colors_hsv.append((h, min(1, s + 0.2), v))
            colors_hsv.append((h, s, max(0, v - 0.3)))
            colors_hsv.append((h, s, min(1, v + 0.2)))
        
        # Convert to hex
        hex_colors = []
        for ch, cs, cv in colors_hsv:
            cr, cg, cb = colorsys.hsv_to_rgb(ch % 1.0, max(0, min(1, cs)), max(0, min(1, cv)))
            hex_colors.append(rgb_to_hex(cr, cg, cb))
        
        # Create palette image
        swatch_w = 64
        swatch_h = 64
        num_colors = len(hex_colors)
        palette_w = swatch_w * num_colors
        
        palette = torch.zeros(1, swatch_h, palette_w, 3)
        for i, hex_c in enumerate(hex_colors):
            cr, cg, cb = hex_to_rgb(hex_c)
            palette[0, :, i*swatch_w:(i+1)*swatch_w, 0] = cr
            palette[0, :, i*swatch_w:(i+1)*swatch_w, 1] = cg
            palette[0, :, i*swatch_w:(i+1)*swatch_w, 2] = cb
        
        return (", ".join(hex_colors), "\n".join(hex_colors), palette)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesColorPicker": OllamaNodesColorPicker,
    "OllamaNodesColorToImage": OllamaNodesColorToImage,
    "OllamaNodesGradientImage": OllamaNodesGradientImage,
    "OllamaNodesColorHarmonies": OllamaNodesColorHarmonies,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesColorPicker": "🎨 Color Picker",
    "OllamaNodesColorToImage": "🎨 Color → Image",
    "OllamaNodesGradientImage": "🎨 Gradient Image",
    "OllamaNodesColorHarmonies": "🎨 Color Harmonies",
}
