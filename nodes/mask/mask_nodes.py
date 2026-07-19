"""
OllamaNodes - Mask Operation Nodes
Create, manipulate, and combine masks.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class OllamaNodesMaskCreate:
    """Create masks from geometric shapes. JS widget provides interactive shape editor."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "shape": (["rectangle", "ellipse", "triangle", "diamond"],),
                "x": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shape_width": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shape_height": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "Ollama-Magic-Nodes/Mask"
    DESCRIPTION = "Create a mask from geometric shapes (rectangle, ellipse, triangle, diamond) with feathering."

    def create_mask(self, width, height, shape, x, y, shape_width, shape_height, feather, invert):
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + shape_width) * width)
        y2 = int((y + shape_height) * height)
        
        if shape == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=255)
        elif shape == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=255)
        elif shape == "triangle":
            cx = (x1 + x2) // 2
            draw.polygon([(cx, y1), (x1, y2), (x2, y2)], fill=255)
        elif shape == "diamond":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            draw.polygon([(cx, y1), (x2, cy), (cx, y2), (x1, cy)], fill=255)
        
        if feather > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))
        
        mask_np = np.array(mask_img).astype(np.float32) / 255.0
        
        if invert:
            mask_np = 1.0 - mask_np
        
        return (torch.from_numpy(mask_np).unsqueeze(0),)


class OllamaNodesMaskGradient:
    """Create linear or radial gradient masks."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "gradient_type": (["linear_horizontal", "linear_vertical", "linear_diagonal", "radial"],),
                "start_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_gradient"
    CATEGORY = "Ollama-Magic-Nodes/Mask"
    DESCRIPTION = "Create linear or radial gradient masks."

    def create_gradient(self, width, height, gradient_type, start_value, end_value, center_x, center_y):
        if gradient_type == "linear_horizontal":
            grad = np.linspace(start_value, end_value, width)
            mask = np.tile(grad, (height, 1))
        elif gradient_type == "linear_vertical":
            grad = np.linspace(start_value, end_value, height)
            mask = np.tile(grad[:, None], (1, width))
        elif gradient_type == "linear_diagonal":
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            xx, yy = np.meshgrid(x, y)
            mask = (xx + yy) / 2.0
            mask = start_value + mask * (end_value - start_value)
        elif gradient_type == "radial":
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            xx, yy = np.meshgrid(x, y)
            dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
            dist = dist / dist.max()
            mask = start_value + dist * (end_value - start_value)
        else:
            mask = np.full((height, width), start_value)
        
        mask = np.clip(mask, 0, 1).astype(np.float32)
        return (torch.from_numpy(mask).unsqueeze(0),)


class OllamaNodesMaskCombine:
    """Combine two masks using various operations."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "operation": (["add", "subtract", "multiply", "intersect_min", "union_max", "xor", "difference"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine"
    CATEGORY = "Ollama-Magic-Nodes/Mask"
    DESCRIPTION = "Combine two masks: add, subtract, multiply, intersect, union, XOR, or difference."

    def combine(self, mask_a, mask_b, operation):
        if mask_a.shape != mask_b.shape:
            mask_b = torch.nn.functional.interpolate(
                mask_b.unsqueeze(1), size=mask_a.shape[-2:], mode='bilinear'
            ).squeeze(1)
        
        if operation == "add":
            result = torch.clamp(mask_a + mask_b, 0, 1)
        elif operation == "subtract":
            result = torch.clamp(mask_a - mask_b, 0, 1)
        elif operation == "multiply":
            result = mask_a * mask_b
        elif operation == "intersect_min":
            result = torch.min(mask_a, mask_b)
        elif operation == "union_max":
            result = torch.max(mask_a, mask_b)
        elif operation == "xor":
            result = torch.abs(mask_a - mask_b)
        elif operation == "difference":
            result = torch.abs(mask_a - mask_b)
        else:
            result = mask_a
        
        return (result,)


class OllamaNodesMaskInvert:
    """Invert a mask."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "invert"
    CATEGORY = "Ollama-Magic-Nodes/Mask"
    DESCRIPTION = "Invert a mask (swap black and white)."

    def invert(self, mask):
        return (1.0 - mask,)


class OllamaNodesMaskBlur:
    """Blur/feather mask edges."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "blur_mask"
    CATEGORY = "Ollama-Magic-Nodes/Mask"
    DESCRIPTION = "Blur/feather mask edges with Gaussian blur."

    def blur_mask(self, mask, radius):
        if radius <= 0:
            return (mask,)
        
        results = []
        for i in range(mask.shape[0]):
            m = mask[i].cpu().numpy()
            m_img = Image.fromarray((m * 255).astype(np.uint8), 'L')
            m_img = m_img.filter(ImageFilter.GaussianBlur(radius=radius))
            m_np = np.array(m_img).astype(np.float32) / 255.0
            results.append(torch.from_numpy(m_np))
        
        return (torch.stack(results),)


class OllamaNodesMaskThreshold:
    """Apply threshold to mask creating binary result."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "threshold"
    CATEGORY = "Ollama-Magic-Nodes/Mask"
    DESCRIPTION = "Apply a threshold to a mask, creating a binary (black/white) result."

    def threshold(self, mask, threshold):
        return ((mask >= threshold).float(),)


class OllamaNodesMaskExpand:
    """Expand or contract mask boundaries."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "pixels": ("INT", {"default": 5, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "expand"
    CATEGORY = "Ollama-Magic-Nodes/Mask"
    DESCRIPTION = "Expand (positive) or contract (negative) mask boundaries by pixel amount."

    def expand(self, mask, pixels):
        if pixels == 0:
            return (mask,)
        
        results = []
        for i in range(mask.shape[0]):
            m = mask[i].cpu().numpy()
            m_img = Image.fromarray((m * 255).astype(np.uint8), 'L')
            
            if pixels > 0:
                m_img = m_img.filter(ImageFilter.MaxFilter(size=pixels * 2 + 1))
            else:
                m_img = m_img.filter(ImageFilter.MinFilter(size=abs(pixels) * 2 + 1))
            
            m_np = np.array(m_img).astype(np.float32) / 255.0
            results.append(torch.from_numpy(m_np))
        
        return (torch.stack(results),)


class OllamaNodesMaskPreview:
    """Preview mask overlaid on image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": "#ff0000"}),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "Ollama-Magic-Nodes/Mask"
    DESCRIPTION = "Preview a mask, optionally overlaid on an image with configurable color and opacity."

    def preview(self, mask, image=None, color="#ff0000", opacity=0.5):
        h = mask.shape[-2]
        w = mask.shape[-1]
        
        # Parse color
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16) / 255.0, int(color[2:4], 16) / 255.0, int(color[4:6], 16) / 255.0
        
        mask_3d = mask[0].unsqueeze(-1)
        
        if image is not None:
            base = image[0:1].clone()
            if base.shape[1] != h or base.shape[2] != w:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray((base[0].cpu().numpy() * 255).astype(np.uint8))
                pil_img = pil_img.resize((w, h), PILImage.LANCZOS)
                base = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)
        else:
            base = torch.zeros(1, h, w, 3)
        
        overlay = torch.tensor([r, g, b]).view(1, 1, 1, 3).expand(1, h, w, 3)
        mask_expanded = mask_3d.unsqueeze(0).expand(1, h, w, 3) if mask_3d.dim() == 2 else mask_3d.expand(1, h, w, 3)
        
        result = base * (1.0 - mask_expanded * opacity) + overlay * mask_expanded * opacity
        return (torch.clamp(result, 0, 1),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesMaskCreate": OllamaNodesMaskCreate,
    "OllamaNodesMaskGradient": OllamaNodesMaskGradient,
    "OllamaNodesMaskCombine": OllamaNodesMaskCombine,
    "OllamaNodesMaskInvert": OllamaNodesMaskInvert,
    "OllamaNodesMaskBlur": OllamaNodesMaskBlur,
    "OllamaNodesMaskThreshold": OllamaNodesMaskThreshold,
    "OllamaNodesMaskExpand": OllamaNodesMaskExpand,
    "OllamaNodesMaskPreview": OllamaNodesMaskPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesMaskCreate": "🎭 Mask Create",
    "OllamaNodesMaskGradient": "🎭 Mask Gradient",
    "OllamaNodesMaskCombine": "🎭 Mask Combine",
    "OllamaNodesMaskInvert": "🎭 Mask Invert",
    "OllamaNodesMaskBlur": "🎭 Mask Blur",
    "OllamaNodesMaskThreshold": "🎭 Mask Threshold",
    "OllamaNodesMaskExpand": "🎭 Mask Expand",
    "OllamaNodesMaskPreview": "🎭 Mask Preview",
}
