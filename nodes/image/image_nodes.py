"""
OllamaNodes - Image Processing Nodes
Comprehensive image manipulation without external dependencies.
Uses only torch (provided by ComfyUI), numpy, and Pillow.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import json


def tensor_to_pil(tensor):
    """Convert ComfyUI IMAGE tensor [B,H,W,C] to list of PIL Images."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    images = []
    for i in range(tensor.shape[0]):
        img_np = (tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))
    return images


def pil_to_tensor(images):
    """Convert list of PIL Images to ComfyUI IMAGE tensor [B,H,W,C]."""
    if not isinstance(images, list):
        images = [images]
    tensors = []
    for img in images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))
    return torch.stack(tensors)


class OllamaNodesImageResize:
    """Smart image resize with multiple fitting modes."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "mode": (["stretch", "fit_contain", "fit_cover", "pad", "crop_center"],),
                "interpolation": (["lanczos", "bicubic", "bilinear", "nearest"],),
                "pad_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Resize images with multiple fitting modes: stretch, fit (contain/cover), pad, or center crop."

    def resize(self, image, width, height, mode, interpolation, pad_color):
        resample_map = {
            "lanczos": Image.LANCZOS,
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
        }
        resample = resample_map.get(interpolation, Image.LANCZOS)
        
        pil_images = tensor_to_pil(image)
        results = []
        
        for img in pil_images:
            if mode == "stretch":
                result = img.resize((width, height), resample)
            elif mode == "fit_contain":
                result = ImageOps.contain(img, (width, height), resample)
                bg = Image.new('RGB', (width, height), pad_color)
                offset = ((width - result.width) // 2, (height - result.height) // 2)
                bg.paste(result, offset)
                result = bg
            elif mode == "fit_cover":
                result = ImageOps.cover(img, (width, height), resample)
                left = (result.width - width) // 2
                top = (result.height - height) // 2
                result = result.crop((left, top, left + width, top + height))
            elif mode == "pad":
                result = ImageOps.pad(img, (width, height), resample, color=pad_color)
            elif mode == "crop_center":
                result = ImageOps.fit(img, (width, height), resample)
            else:
                result = img.resize((width, height), resample)
            results.append(result)
        
        return (pil_to_tensor(results),)


class OllamaNodesImageCrop:
    """Crop images with specified coordinates. JS widget provides interactive crop handles."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "width": ("INT", {"default": 256, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 256, "min": 1, "max": 16384, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Crop an image to specified coordinates and dimensions."

    def crop(self, image, x, y, width, height):
        pil_images = tensor_to_pil(image)
        results = []
        for img in pil_images:
            x2 = min(x + width, img.width)
            y2 = min(y + height, img.height)
            x1 = min(x, img.width)
            y1 = min(y, img.height)
            cropped = img.crop((x1, y1, x2, y2))
            results.append(cropped)
        return (pil_to_tensor(results),)


class OllamaNodesImageFlip:
    """Flip or rotate images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": (["flip_horizontal", "flip_vertical", "rotate_90_cw", "rotate_90_ccw", "rotate_180"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "flip"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Flip or rotate images (horizontal, vertical, 90°, 180°)."

    def flip(self, image, operation):
        pil_images = tensor_to_pil(image)
        results = []
        for img in pil_images:
            if operation == "flip_horizontal":
                result = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif operation == "flip_vertical":
                result = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif operation == "rotate_90_cw":
                result = img.transpose(Image.ROTATE_270)
            elif operation == "rotate_90_ccw":
                result = img.transpose(Image.ROTATE_90)
            elif operation == "rotate_180":
                result = img.transpose(Image.ROTATE_180)
            else:
                result = img
            results.append(result)
        return (pil_to_tensor(results),)


class OllamaNodesImageColorAdjust:
    """Adjust image colors: brightness, contrast, saturation, hue, gamma."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Adjust image brightness, contrast, saturation, hue, and gamma."

    def adjust(self, image, brightness, contrast, saturation, hue_shift, gamma):
        pil_images = tensor_to_pil(image)
        results = []
        for img in pil_images:
            if brightness != 1.0:
                img = ImageEnhance.Brightness(img).enhance(brightness)
            if contrast != 1.0:
                img = ImageEnhance.Contrast(img).enhance(contrast)
            if saturation != 1.0:
                img = ImageEnhance.Color(img).enhance(saturation)
            if hue_shift != 0.0:
                hsv = img.convert('HSV')
                h, s, v = hsv.split()
                h_np = np.array(h, dtype=np.int16)
                h_np = ((h_np + int(hue_shift * 255 / 360)) % 256).astype(np.uint8)
                h = Image.fromarray(h_np, 'L')
                img = Image.merge('HSV', (h, s, v)).convert('RGB')
            if gamma != 1.0:
                img_np = np.array(img).astype(np.float32) / 255.0
                img_np = np.power(img_np, 1.0 / gamma)
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
            results.append(img)
        return (pil_to_tensor(results),)


class OllamaNodesImageBlend:
    """Blend two images with various blend modes."""
    
    BLEND_MODES = [
        "normal", "multiply", "screen", "overlay", "darken", "lighten",
        "color_dodge", "color_burn", "hard_light", "soft_light",
        "difference", "exclusion", "add", "subtract", "divide"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "blend_mode": (cls.BLEND_MODES,),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Blend two images using various blend modes (multiply, screen, overlay, etc.)."

    def _blend_op(self, a, b, mode):
        eps = 1e-7
        if mode == "normal":
            return b
        elif mode == "multiply":
            return a * b
        elif mode == "screen":
            return 1.0 - (1.0 - a) * (1.0 - b)
        elif mode == "overlay":
            mask = a < 0.5
            result = torch.where(mask, 2 * a * b, 1.0 - 2 * (1.0 - a) * (1.0 - b))
            return result
        elif mode == "darken":
            return torch.min(a, b)
        elif mode == "lighten":
            return torch.max(a, b)
        elif mode == "color_dodge":
            return torch.clamp(a / (1.0 - b + eps), 0, 1)
        elif mode == "color_burn":
            return torch.clamp(1.0 - (1.0 - a) / (b + eps), 0, 1)
        elif mode == "hard_light":
            mask = b < 0.5
            return torch.where(mask, 2 * a * b, 1.0 - 2 * (1.0 - a) * (1.0 - b))
        elif mode == "soft_light":
            return torch.where(b < 0.5,
                a * (2 * b + a * (1 - 2 * b)),
                a + (2 * b - 1) * (torch.sqrt(a) - a))
        elif mode == "difference":
            return torch.abs(a - b)
        elif mode == "exclusion":
            return a + b - 2 * a * b
        elif mode == "add":
            return torch.clamp(a + b, 0, 1)
        elif mode == "subtract":
            return torch.clamp(a - b, 0, 1)
        elif mode == "divide":
            return torch.clamp(a / (b + eps), 0, 1)
        return b

    def blend(self, image_a, image_b, blend_mode, opacity):
        if image_a.shape[1:3] != image_b.shape[1:3]:
            pil_b = tensor_to_pil(image_b)
            h, w = image_a.shape[1], image_a.shape[2]
            pil_b = [img.resize((w, h), Image.LANCZOS) for img in pil_b]
            image_b = pil_to_tensor(pil_b)
        
        blended = self._blend_op(image_a, image_b, blend_mode)
        result = image_a * (1.0 - opacity) + blended * opacity
        result = torch.clamp(result, 0, 1)
        return (result,)


class OllamaNodesImageSharpen:
    """Sharpen images using unsharp mask."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "radius": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "threshold": ("INT", {"default": 3, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sharpen"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Sharpen images using unsharp mask with adjustable strength, radius, and threshold."

    def sharpen(self, image, strength, radius, threshold):
        pil_images = tensor_to_pil(image)
        results = []
        for img in pil_images:
            sharpened = img.filter(ImageFilter.UnsharpMask(
                radius=radius,
                percent=int(strength * 100),
                threshold=threshold
            ))
            results.append(sharpened)
        return (pil_to_tensor(results),)


class OllamaNodesImageBlur:
    """Blur images with different blur types."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (["gaussian", "box", "median"],),
                "radius": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blur"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Apply Gaussian, box, or median blur to images."

    def blur(self, image, blur_type, radius):
        pil_images = tensor_to_pil(image)
        results = []
        for img in pil_images:
            if blur_type == "gaussian":
                result = img.filter(ImageFilter.GaussianBlur(radius=radius))
            elif blur_type == "box":
                result = img.filter(ImageFilter.BoxBlur(radius=radius))
            elif blur_type == "median":
                size = max(3, int(radius) * 2 + 1)
                if size % 2 == 0:
                    size += 1
                result = img.filter(ImageFilter.MedianFilter(size=size))
            else:
                result = img
            results.append(result)
        return (pil_to_tensor(results),)


class OllamaNodesImageNoise:
    """Add procedural noise to images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_type": (["gaussian", "uniform", "salt_pepper"],),
                "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_noise"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Add Gaussian, uniform, or salt & pepper noise to images."

    def add_noise(self, image, noise_type, strength, seed):
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        
        result = image.clone()
        
        if noise_type == "gaussian":
            noise = torch.randn_like(result) * strength
            result = torch.clamp(result + noise, 0, 1)
        elif noise_type == "uniform":
            noise = (torch.rand_like(result) - 0.5) * 2 * strength
            result = torch.clamp(result + noise, 0, 1)
        elif noise_type == "salt_pepper":
            mask = torch.rand_like(result[:, :, :, 0:1])
            salt = mask < (strength / 2)
            pepper = mask > (1 - strength / 2)
            result[salt.expand_as(result)] = 1.0
            result[pepper.expand_as(result)] = 0.0
        
        return (result,)


class OllamaNodesImageToGrayscale:
    """Convert image to grayscale with channel weighting."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["luminance", "average", "red_channel", "green_channel", "blue_channel", "custom"],),
                "red_weight": ("FLOAT", {"default": 0.299, "min": 0.0, "max": 1.0, "step": 0.001}),
                "green_weight": ("FLOAT", {"default": 0.587, "min": 0.0, "max": 1.0, "step": 0.001}),
                "blue_weight": ("FLOAT", {"default": 0.114, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "to_grayscale"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Convert to grayscale using various methods (luminance, average, per-channel, or custom weights)."

    def to_grayscale(self, image, method, red_weight, green_weight, blue_weight):
        if method == "luminance":
            weights = torch.tensor([0.299, 0.587, 0.114])
        elif method == "average":
            weights = torch.tensor([1/3, 1/3, 1/3])
        elif method == "red_channel":
            weights = torch.tensor([1.0, 0.0, 0.0])
        elif method == "green_channel":
            weights = torch.tensor([0.0, 1.0, 0.0])
        elif method == "blue_channel":
            weights = torch.tensor([0.0, 0.0, 1.0])
        else:
            total = red_weight + green_weight + blue_weight
            if total > 0:
                weights = torch.tensor([red_weight / total, green_weight / total, blue_weight / total])
            else:
                weights = torch.tensor([1/3, 1/3, 1/3])
        
        gray = torch.sum(image[..., :3] * weights.to(image.device), dim=-1, keepdim=True)
        result = gray.repeat(1, 1, 1, 3)
        return (result,)


class OllamaNodesImageChannelSplit:
    """Split image into individual R, G, B channels."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("red", "green", "blue")
    FUNCTION = "split"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Split an image into its Red, Green, and Blue channels (each output as grayscale)."

    def split(self, image):
        r = image[..., 0:1].repeat(1, 1, 1, 3)
        g = image[..., 1:2].repeat(1, 1, 1, 3)
        b = image[..., 2:3].repeat(1, 1, 1, 3)
        return (r, g, b)


class OllamaNodesImageChannelMerge:
    """Merge R, G, B channels into a single image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "red": ("IMAGE",),
                "green": ("IMAGE",),
                "blue": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Merge separate R, G, B channel images into a single RGB image."

    def merge(self, red, green, blue):
        r = red[..., 0:1]
        g = green[..., 1:2] if green.shape[-1] > 1 else green[..., 0:1]
        b = blue[..., 2:3] if blue.shape[-1] > 2 else blue[..., 0:1]
        result = torch.cat([r, g, b], dim=-1)
        return (result,)


class OllamaNodesImageBatch:
    """Create, split, or manipulate image batches."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["join", "split_first", "split_last", "get_index", "count"],),
            },
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "count")
    FUNCTION = "batch_op"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Batch operations: join images into batch, split first/last, get by index, count batch size."

    def batch_op(self, operation, image_a=None, image_b=None, index=0):
        if operation == "join" and image_a is not None and image_b is not None:
            if image_a.shape[1:] != image_b.shape[1:]:
                pil_b = tensor_to_pil(image_b)
                h, w = image_a.shape[1], image_a.shape[2]
                pil_b = [img.resize((w, h), Image.LANCZOS) for img in pil_b]
                image_b = pil_to_tensor(pil_b).to(image_a.device)
            result = torch.cat([image_a, image_b], dim=0)
            return (result, result.shape[0])
        elif operation == "split_first" and image_a is not None:
            return (image_a[0:1], image_a.shape[0])
        elif operation == "split_last" and image_a is not None:
            return (image_a[-1:], image_a.shape[0])
        elif operation == "get_index" and image_a is not None:
            idx = min(index, image_a.shape[0] - 1)
            return (image_a[idx:idx+1], image_a.shape[0])
        elif operation == "count" and image_a is not None:
            return (image_a, image_a.shape[0])
        
        empty = torch.zeros(1, 64, 64, 3)
        return (empty, 0)


class OllamaNodesImageGrid:
    """Create a contact sheet / image grid from a batch."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "columns": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "spacing": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "background_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid_image",)
    FUNCTION = "create_grid"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Arrange batch images into a grid/contact sheet with configurable columns and spacing."

    def create_grid(self, images, columns, spacing, background_color):
        pil_images = tensor_to_pil(images)
        count = len(pil_images)
        rows = (count + columns - 1) // columns
        
        cell_w = max(img.width for img in pil_images)
        cell_h = max(img.height for img in pil_images)
        
        grid_w = columns * cell_w + (columns + 1) * spacing
        grid_h = rows * cell_h + (rows + 1) * spacing
        
        grid = Image.new('RGB', (grid_w, grid_h), background_color)
        
        for i, img in enumerate(pil_images):
            row = i // columns
            col = i % columns
            x = spacing + col * (cell_w + spacing) + (cell_w - img.width) // 2
            y = spacing + row * (cell_h + spacing) + (cell_h - img.height) // 2
            grid.paste(img, (x, y))
        
        return (pil_to_tensor([grid]),)


class OllamaNodesImageCompare:
    """Compare two images side-by-side with interactive slider (JS widget)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mode": (["side_by_side", "overlay", "difference", "split"],),
                "split_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "compare"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Compare two images: side-by-side, overlay, difference, or split view with interactive slider."

    def compare(self, image_a, image_b, mode, split_position, overlay_opacity):
        if image_a.shape[1:3] != image_b.shape[1:3]:
            pil_b = tensor_to_pil(image_b)
            h, w = image_a.shape[1], image_a.shape[2]
            pil_b = [img.resize((w, h), Image.LANCZOS) for img in pil_b]
            image_b = pil_to_tensor(pil_b).to(image_a.device)
        
        a = image_a[0:1]
        b = image_b[0:1]
        
        if mode == "side_by_side":
            result = torch.cat([a, b], dim=2)
        elif mode == "overlay":
            result = a * (1 - overlay_opacity) + b * overlay_opacity
        elif mode == "difference":
            result = torch.abs(a - b)
        elif mode == "split":
            w = a.shape[2]
            split_x = int(w * split_position)
            result = a.clone()
            result[:, :, split_x:, :] = b[:, :, split_x:, :]
        else:
            result = a
        
        return (torch.clamp(result, 0, 1),)


class OllamaNodesColorPalette:
    """Extract dominant color palette from image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("palette_image", "colors_hex")
    FUNCTION = "extract_palette"
    CATEGORY = "Ollama-Magic-Nodes/Image"
    DESCRIPTION = "Extract the dominant color palette from an image. Outputs a visual palette strip and hex color codes."

    def extract_palette(self, image, num_colors):
        pil_img = tensor_to_pil(image)[0]
        small = pil_img.resize((64, 64), Image.LANCZOS)
        pixels = np.array(small).reshape(-1, 3).astype(np.float64)
        
        # Simple k-means clustering
        np.random.seed(42)
        indices = np.random.choice(len(pixels), num_colors, replace=False)
        centers = pixels[indices].copy()
        
        for _ in range(20):
            distances = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([
                pixels[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centers[k]
                for k in range(num_colors)
            ])
            if np.allclose(centers, new_centers, atol=1):
                break
            centers = new_centers
        
        colors = centers.astype(np.uint8)
        
        # Sort by brightness
        brightness = np.sum(colors, axis=1)
        sort_idx = np.argsort(brightness)
        colors = colors[sort_idx]
        
        swatch_w = 64
        swatch_h = 64
        palette_w = swatch_w * num_colors
        palette_img = Image.new('RGB', (palette_w, swatch_h))
        
        hex_colors = []
        for i, color in enumerate(colors):
            swatch = Image.new('RGB', (swatch_w, swatch_h), tuple(color))
            palette_img.paste(swatch, (i * swatch_w, 0))
            hex_colors.append(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
        
        return (pil_to_tensor([palette_img]), ", ".join(hex_colors))


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesImageResize": OllamaNodesImageResize,
    "OllamaNodesImageCrop": OllamaNodesImageCrop,
    "OllamaNodesImageFlip": OllamaNodesImageFlip,
    "OllamaNodesImageColorAdjust": OllamaNodesImageColorAdjust,
    "OllamaNodesImageBlend": OllamaNodesImageBlend,
    "OllamaNodesImageSharpen": OllamaNodesImageSharpen,
    "OllamaNodesImageBlur": OllamaNodesImageBlur,
    "OllamaNodesImageNoise": OllamaNodesImageNoise,
    "OllamaNodesImageToGrayscale": OllamaNodesImageToGrayscale,
    "OllamaNodesImageChannelSplit": OllamaNodesImageChannelSplit,
    "OllamaNodesImageChannelMerge": OllamaNodesImageChannelMerge,
    "OllamaNodesImageBatch": OllamaNodesImageBatch,
    "OllamaNodesImageGrid": OllamaNodesImageGrid,
    "OllamaNodesImageCompare": OllamaNodesImageCompare,
    "OllamaNodesColorPalette": OllamaNodesColorPalette,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesImageResize": "🎨 Image Resize",
    "OllamaNodesImageCrop": "🎨 Image Crop",
    "OllamaNodesImageFlip": "🎨 Image Flip/Rotate",
    "OllamaNodesImageColorAdjust": "🎨 Color Adjust",
    "OllamaNodesImageBlend": "🎨 Image Blend",
    "OllamaNodesImageSharpen": "🎨 Image Sharpen",
    "OllamaNodesImageBlur": "🎨 Image Blur",
    "OllamaNodesImageNoise": "🎨 Image Noise",
    "OllamaNodesImageToGrayscale": "🎨 Grayscale",
    "OllamaNodesImageChannelSplit": "🎨 Channel Split",
    "OllamaNodesImageChannelMerge": "🎨 Channel Merge",
    "OllamaNodesImageBatch": "🎨 Image Batch",
    "OllamaNodesImageGrid": "🎨 Image Grid",
    "OllamaNodesImageCompare": "🎨 Image Compare",
    "OllamaNodesColorPalette": "🎨 Color Palette",
}
