"""
OllamaNodes - Video/Image Composition Nodes
Layer-based compositing with text overlays, transparent images, and effects.
Similar to After Effects / DaVinci Resolve compositing workflow.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import math
import json


def tensor_to_pil(tensor):
    """Convert IMAGE tensor [B,H,W,C] to list of PIL Images."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    images = []
    for i in range(tensor.shape[0]):
        img_np = (tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))
    return images


def pil_to_tensor(images):
    """Convert list of PIL Images to IMAGE tensor [B,H,W,C]."""
    if not isinstance(images, list):
        images = [images]
    tensors = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGB')
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))
    return torch.stack(tensors)


def load_font(font_size=24, font_path=None):
    """Load font with fallback."""
    font_paths = [
        font_path,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    
    for path in font_paths:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except:
                continue
    
    return ImageFont.load_default()


class OllamaNodesTextOverlay:
    """Add text overlay to images/video frames with full styling control."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "Your Text Here", "multiline": True}),
                "position_x": ("INT", {"default": 50, "min": -4096, "max": 4096, "step": 1}),
                "position_y": ("INT", {"default": 50, "min": -4096, "max": 4096, "step": 1}),
            },
            "optional": {
                "font_size": ("INT", {"default": 48, "min": 8, "max": 500, "step": 1}),
                "text_color": ("STRING", {"default": "#ffffff"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alignment": (["left", "center", "right"],),
                "position_mode": (["pixel", "percentage", "center", "top_left", "top_center", 
                                   "top_right", "middle_left", "middle_center", "middle_right",
                                   "bottom_left", "bottom_center", "bottom_right"],),
                "bold": ("BOOLEAN", {"default": False}),
                "italic": ("BOOLEAN", {"default": False}),
                "shadow": ("BOOLEAN", {"default": False}),
                "shadow_color": ("STRING", {"default": "#000000"}),
                "shadow_offset_x": ("INT", {"default": 2, "min": -20, "max": 20}),
                "shadow_offset_y": ("INT", {"default": 2, "min": -20, "max": 20}),
                "outline": ("BOOLEAN", {"default": False}),
                "outline_color": ("STRING", {"default": "#000000"}),
                "outline_width": ("INT", {"default": 2, "min": 1, "max": 10}),
                "font_path": ("STRING", {"default": "", "placeholder": "Custom font path (optional)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited",)
    FUNCTION = "add_text_overlay"
    CATEGORY = "Ollama-Magic-Nodes/Composition"
    DESCRIPTION = """Add styled text overlay to images.
    
Position Modes:
- pixel: Exact pixel coordinates
- percentage: % of image dimensions (0-100)
- center: Centered on canvas
- Named positions: top_left, middle_center, bottom_right, etc.

Features:
- Shadow with configurable offset
- Outline/stroke effect
- Opacity control
- Custom font support"""

    def add_text_overlay(self, images, text, position_x, position_y, font_size=48,
                        text_color="#ffffff", opacity=1.0, alignment="left",
                        position_mode="pixel", bold=False, italic=False,
                        shadow=False, shadow_color="#000000", shadow_offset_x=2, shadow_offset_y=2,
                        outline=False, outline_color="#000000", outline_width=2,
                        font_path=""):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        for img in pil_images:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create text layer
            txt_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(txt_layer)
            
            # Load font
            font = load_font(font_size, font_path)
            
            # Parse colors
            text_color_rgb = self._parse_color(text_color)
            shadow_color_rgb = self._parse_color(shadow_color)
            outline_color_rgb = self._parse_color(outline_color)
            
            # Calculate position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x, y = self._calculate_position(
                position_x, position_y, position_mode,
                img.width, img.height, text_width, text_height, alignment
            )
            
            alpha = int(opacity * 255)
            
            # Draw shadow
            if shadow:
                shadow_x = x + shadow_offset_x
                shadow_y = y + shadow_offset_y
                draw.text((shadow_x, shadow_y), text, font=font, 
                         fill=(shadow_color_rgb[0], shadow_color_rgb[1], shadow_color_rgb[2], alpha))
            
            # Draw outline
            if outline:
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx*dx + dy*dy <= outline_width*outline_width:
                            draw.text((x + dx, y + dy), text, font=font,
                                     fill=(outline_color_rgb[0], outline_color_rgb[1], outline_color_rgb[2], alpha))
            
            # Draw main text
            draw.text((x, y), text, font=font,
                     fill=(text_color_rgb[0], text_color_rgb[1], text_color_rgb[2], alpha))
            
            # Composite
            composited = Image.alpha_composite(img, txt_layer)
            results.append(composited)
        
        return (pil_to_tensor(results),)
    
    def _parse_color(self, color_str):
        """Parse hex color to RGB tuple."""
        color_str = color_str.strip().lstrip('#')
        if len(color_str) == 3:
            color_str = ''.join([c*2 for c in color_str])
        try:
            return (int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16))
        except:
            return (255, 255, 255)
    
    def _calculate_position(self, px, py, mode, img_w, img_h, text_w, text_h, alignment):
        """Calculate text position based on mode."""
        if mode == "pixel":
            x, y = px, py
        elif mode == "percentage":
            x = int(px / 100 * img_w)
            y = int(py / 100 * img_h)
        elif mode == "center":
            x = (img_w - text_w) // 2
            y = (img_h - text_h) // 2
        elif mode == "top_left":
            x, y = 10, 10
        elif mode == "top_center":
            x = (img_w - text_w) // 2
            y = 10
        elif mode == "top_right":
            x = img_w - text_w - 10
            y = 10
        elif mode == "middle_left":
            x = 10
            y = (img_h - text_h) // 2
        elif mode == "middle_center":
            x = (img_w - text_w) // 2
            y = (img_h - text_h) // 2
        elif mode == "middle_right":
            x = img_w - text_w - 10
            y = (img_h - text_h) // 2
        elif mode == "bottom_left":
            x = 10
            y = img_h - text_h - 10
        elif mode == "bottom_center":
            x = (img_w - text_w) // 2
            y = img_h - text_h - 10
        elif mode == "bottom_right":
            x = img_w - text_w - 10
            y = img_h - text_h - 10
        else:
            x, y = px, py
        
        # Apply alignment for text position
        if alignment == "center":
            x = x - text_w // 2
        elif alignment == "right":
            x = x - text_w
        
        return max(0, min(x, img_w - text_w)), max(0, min(y, img_h - text_h))


class OllamaNodesImageOverlay:
    """Overlay a transparent image (PNG/RGBA) onto another image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "position_mode": (["pixel", "percentage", "center", "top_left", "top_right",
                                   "bottom_left", "bottom_right"],),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                "use_alpha_from_overlay": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited",)
    FUNCTION = "overlay_image"
    CATEGORY = "Ollama-Magic-Nodes/Composition"
    DESCRIPTION = """Overlay a transparent image (PNG) onto background.
    
Useful for:
- Watermarks and logos
- Character sprites/overlays
- UI elements
- Visual effects
- Decorative elements

Tip: Use RGBA PNG images for transparency support."""
    
    def overlay_image(self, background, overlay_image, position_x, position_y,
                     scale=1.0, opacity=1.0, rotation=0.0, position_mode="pixel",
                     flip_horizontal=False, flip_vertical=False, use_alpha_from_overlay=True):
        
        bg_images = tensor_to_pil(background)
        ol_images = tensor_to_pil(overlay_image)
        results = []
        
        for i, bg in enumerate(bg_images):
            if bg.mode != 'RGBA':
                bg = bg.convert('RGBA')
            
            ol_idx = min(i, len(ol_images) - 1)
            ol = ol_images[ol_idx]
            
            # Apply transformations to overlay
            if flip_horizontal:
                ol = ol.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_vertical:
                ol = ol.transpose(Image.FLIP_TOP_BOTTOM)
            
            if scale != 1.0:
                new_w = int(ol.width * scale)
                new_h = int(ol.height * scale)
                ol = ol.resize((new_w, new_h), Image.LANCZOS)
            
            if rotation != 0.0:
                ol = ol.rotate(-rotation, expand=True, resample=Image.BICUBIC)
            
            if opacity < 1.0:
                ol_arr = np.array(ol)
                if ol_arr.shape[2] == 4:
                    ol_arr[:, :, 3] = (ol_arr[:, :, 3] * opacity).astype(np.uint8)
                ol = Image.fromarray(ol_arr)
            
            if ol.mode != 'RGBA':
                ol = ol.convert('RGBA')
            
            # Calculate position
            x, y = self._calculate_position(
                position_x, position_y, position_mode,
                bg.width, bg.height, ol.width, ol.height
            )
            
            # Create paste mask from alpha channel
            if use_alpha_from_overlay and ol.mode == 'RGBA':
                mask = ol.split()[3]
            else:
                mask = None
            
            # Create layer and paste
            layer = Image.new('RGBA', bg.size, (0, 0, 0, 0))
            layer.paste(ol, (x, y), mask)
            
            composited = Image.alpha_composite(bg, layer)
            results.append(composited)
        
        return (pil_to_tensor(results),)
    
    def _calculate_position(self, px, py, mode, bg_w, bg_h, ol_w, ol_h):
        if mode == "pixel":
            return px, py
        elif mode == "percentage":
            return int(px / 100 * bg_w), int(py / 100 * bg_h)
        elif mode == "center":
            return (bg_w - ol_w) // 2, (bg_h - ol_h) // 2
        elif mode == "top_left":
            return 0, 0
        elif mode == "top_right":
            return bg_w - ol_w, 0
        elif mode == "bottom_left":
            return 0, bg_h - ol_h
        elif mode == "bottom_right":
            return bg_w - ol_w, bg_h - ol_h
        return px, py


class OllamaNodesComposeLayers:
    """Multi-layer compositing with control over stacking order."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "layer_1": ("IMAGE",),
            },
            "optional": {
                "layer_2": ("IMAGE",),
                "layer_3": ("IMAGE",),
                "layer_4": ("IMAGE",),
                "layer_5": ("IMAGE",),
                "layer_1_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "layer_2_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "layer_3_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "layer_4_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "layer_5_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "blend_mode_1": (["normal", "multiply", "screen", "overlay", "add"],),
                "blend_mode_2": (["normal", "multiply", "screen", "overlay", "add"],),
                "blend_mode_3": (["normal", "multiply", "screen", "overlay", "add"],),
                "blend_mode_4": (["normal", "multiply", "screen", "overlay", "add"],),
                "blend_mode_5": (["normal", "multiply", "screen", "overlay", "add"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited",)
    FUNCTION = "compose"
    CATEGORY = "Ollama-Magic-Nodes/Composition"
    DESCRIPTION = """Compose up to 6 layers with blend modes and opacity.
    
Blend Modes:
- normal: Standard alpha compositing
- multiply: Darken
- screen: Lighten
- overlay: Contrast
- add: Additive (good for glow effects)
"""
    
    def compose(self, background, layer_1, 
                layer_2=None, layer_3=None, layer_4=None, layer_5=None,
                layer_1_opacity=1.0, layer_2_opacity=1.0, layer_3_opacity=1.0,
                layer_4_opacity=1.0, layer_5_opacity=1.0,
                blend_mode_1="normal", blend_mode_2="normal", blend_mode_3="normal",
                blend_mode_4="normal", blend_mode_5="normal"):
        
        bg_images = tensor_to_pil(background)
        l1_images = tensor_to_pil(layer_1) if layer_1 is not None else []
        l2_images = tensor_to_pil(layer_2) if layer_2 is not None else []
        l3_images = tensor_to_pil(layer_3) if layer_3 is not None else []
        l4_images = tensor_to_pil(layer_4) if layer_4 is not None else []
        l5_images = tensor_to_pil(layer_5) if layer_5 is not None else []
        
        results = []
        
        for i, bg in enumerate(bg_images):
            if bg.mode != 'RGBA':
                bg = bg.convert('RGBA')
            
            result = bg.copy()
            
            layers = [
                (l1_images, layer_1_opacity, blend_mode_1),
                (l2_images, layer_2_opacity, blend_mode_2),
                (l3_images, layer_3_opacity, blend_mode_3),
                (l4_images, layer_4_opacity, blend_mode_4),
                (l5_images, layer_5_opacity, blend_mode_5),
            ]
            
            for layer_images, opacity, blend_mode in layers:
                if not layer_images:
                    continue
                
                l_idx = min(i, len(layer_images) - 1)
                layer = layer_images[l_idx].copy()
                
                # Resize layer to match background if needed
                if layer.size != result.size:
                    layer = layer.resize(result.size, Image.LANCZOS)
                
                if layer.mode != 'RGBA':
                    layer = layer.convert('RGBA')
                
                # Apply opacity
                if opacity < 1.0:
                    layer_arr = np.array(layer)
                    layer_arr[:, :, 3] = (layer_arr[:, :, 3] * opacity).astype(np.uint8)
                    layer = Image.fromarray(layer_arr)
                
                # Apply blend mode
                if blend_mode == "normal":
                    result = Image.alpha_composite(result, layer)
                elif blend_mode == "multiply":
                    result = self._blend_multiply(result, layer)
                elif blend_mode == "screen":
                    result = self._blend_screen(result, layer)
                elif blend_mode == "overlay":
                    result = self._blend_overlay(result, layer)
                elif blend_mode == "add":
                    result = self._blend_add(result, layer)
            
            results.append(result)
        
        return (pil_to_tensor(results),)
    
    def _blend_multiply(self, base, overlay):
        base_arr = np.array(base).astype(float)
        ol_arr = np.array(overlay).astype(float)
        alpha = ol_arr[:, :, 3:4] / 255.0
        blended = (base_arr[:, :, :3] * ol_arr[:, :, :3] / 255.0) * alpha + base_arr[:, :, :3] * (1 - alpha)
        result = np.clip(blended, 0, 255).astype(np.uint8)
        result = np.concatenate([result, base_arr[:, :, 3:4].astype(np.uint8)], axis=2)
        return Image.fromarray(result)
    
    def _blend_screen(self, base, overlay):
        base_arr = np.array(base).astype(float)
        ol_arr = np.array(overlay).astype(float)
        alpha = ol_arr[:, :, 3:4] / 255.0
        blended = (255 - (255 - base_arr[:, :, :3]) * (255 - ol_arr[:, :, :3]) / 255.0) * alpha + base_arr[:, :, :3] * (1 - alpha)
        result = np.clip(blended, 0, 255).astype(np.uint8)
        result = np.concatenate([result, base_arr[:, :, 3:4].astype(np.uint8)], axis=2)
        return Image.fromarray(result)
    
    def _blend_overlay(self, base, overlay):
        base_arr = np.array(base).astype(float)
        ol_arr = np.array(overlay).astype(float)
        alpha = ol_arr[:, :, 3:4] / 255.0
        
        mask = base_arr[:, :, :3] < 128
        blended = np.where(mask, 
            2 * base_arr[:, :, :3] * ol_arr[:, :, :3] / 255.0,
            255 - 2 * (255 - base_arr[:, :, :3]) * (255 - ol_arr[:, :, :3]) / 255.0)
        blended = blended * alpha + base_arr[:, :, :3] * (1 - alpha)
        result = np.clip(blended, 0, 255).astype(np.uint8)
        result = np.concatenate([result, base_arr[:, :, 3:4].astype(np.uint8)], axis=2)
        return Image.fromarray(result)
    
    def _blend_add(self, base, overlay):
        base_arr = np.array(base).astype(float)
        ol_arr = np.array(overlay).astype(float)
        alpha = ol_arr[:, :, 3:4] / 255.0
        blended = np.clip(base_arr[:, :, :3] + ol_arr[:, :, :3] * alpha, 0, 255)
        result = blended.astype(np.uint8)
        result = np.concatenate([result, base_arr[:, :, 3:4].astype(np.uint8)], axis=2)
        return Image.fromarray(result)


class OllamaNodesTextBadge:
    """Create text badges with background, commonly used for titles, labels, etc."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "BADGE"}),
                "position_x": ("INT", {"default": 50, "min": 0, "max": 4096}),
                "position_y": ("INT", {"default": 50, "min": 0, "max": 4096}),
            },
            "optional": {
                "font_size": ("INT", {"default": 24, "min": 8, "max": 200}),
                "text_color": ("STRING", {"default": "#ffffff"}),
                "badge_color": ("STRING", {"default": "#ff6600"}),
                "badge_padding_x": ("INT", {"default": 20, "min": 0, "max": 100}),
                "badge_padding_y": ("INT", {"default": 10, "min": 0, "max": 50}),
                "badge_radius": ("INT", {"default": 8, "min": 0, "max": 50}),
                "opacity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "position_mode": (["top_left", "top_center", "top_right", 
                                   "middle_left", "center", "middle_right",
                                   "bottom_left", "bottom_center", "bottom_right"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited",)
    FUNCTION = "add_badge"
    CATEGORY = "Ollama-Magic-Nodes/Composition"
    DESCRIPTION = """Add a text badge/label with colored background.
    
Great for:
- Episode numbers
- Scene titles
- Section labels
- Corner watermarks with background
"""
    
    def add_badge(self, images, text, position_x, position_y, font_size=24,
                 text_color="#ffffff", badge_color="#ff6600", badge_padding_x=20,
                 badge_padding_y=10, badge_radius=8, opacity=0.9, position_mode="top_left"):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        for img in pil_images:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            font = load_font(font_size)
            text_rgb = self._parse_color(text_color)
            badge_rgb = self._parse_color(badge_color)
            
            # Measure text
            temp_draw = ImageDraw.Draw(img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Badge dimensions
            badge_w = text_w + badge_padding_x * 2
            badge_h = text_h + badge_padding_y * 2
            
            # Calculate position
            x, y = self._calculate_position(
                position_x, position_y, position_mode,
                img.width, img.height, badge_w, badge_h
            )
            
            # Create badge layer
            badge_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            badge_draw = ImageDraw.Draw(badge_layer)
            
            # Draw rounded rectangle badge
            alpha = int(opacity * 255)
            badge_color_rgba = (badge_rgb[0], badge_rgb[1], badge_rgb[2], alpha)
            badge_draw.rounded_rectangle(
                [x, y, x + badge_w, y + badge_h],
                radius=badge_radius,
                fill=badge_color_rgba
            )
            
            # Draw text
            text_x = x + badge_padding_x
            text_y = y + badge_padding_y
            badge_draw.text((text_x, text_y), text, font=font,
                          fill=(text_rgb[0], text_rgb[1], text_rgb[2], alpha))
            
            # Composite
            composited = Image.alpha_composite(img, badge_layer)
            results.append(composited)
        
        return (pil_to_tensor(results),)
    
    def _parse_color(self, color_str):
        color_str = color_str.strip().lstrip('#')
        if len(color_str) == 3:
            color_str = ''.join([c*2 for c in color_str])
        try:
            return (int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16))
        except:
            return (255, 255, 255)
    
    def _calculate_position(self, px, py, mode, img_w, img_h, badge_w, badge_h):
        margin = 10
        if mode == "top_left":
            return margin, margin
        elif mode == "top_center":
            return (img_w - badge_w) // 2, margin
        elif mode == "top_right":
            return img_w - badge_w - margin, margin
        elif mode == "middle_left":
            return margin, (img_h - badge_h) // 2
        elif mode == "center":
            return (img_w - badge_w) // 2, (img_h - badge_h) // 2
        elif mode == "middle_right":
            return img_w - badge_w - margin, (img_h - badge_h) // 2
        elif mode == "bottom_left":
            return margin, img_h - badge_h - margin
        elif mode == "bottom_center":
            return (img_w - badge_w) // 2, img_h - badge_h - margin
        elif mode == "bottom_right":
            return img_w - badge_w - margin, img_h - badge_h - margin
        return px, py


class OllamaNodesVideoProgress:
    """Add video progress bar/indicator to frames."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "total_frames": ("INT", {"default": 100, "min": 1, "max": 100000}),
            },
            "optional": {
                "bar_position": (["bottom", "top"],),
                "bar_height": ("INT", {"default": 4, "min": 1, "max": 20}),
                "bar_color": ("STRING", {"default": "#ff6600"}),
                "bar_background": ("STRING", {"default": "#333333"}),
                "show_percentage": ("BOOLEAN", {"default": False}),
                "show_timecodes": ("BOOLEAN", {"default": False}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
                "font_size": ("INT", {"default": 16, "min": 8, "max": 48}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("with_progress",)
    FUNCTION = "add_progress"
    CATEGORY = "Ollama-Magic-Nodes/Composition"
    DESCRIPTION = """Add video progress bar overlay to frames.
    
Features:
- Progress bar at top or bottom
- Percentage display
- Timecode display (current/total)
- Customizable colors"""
    
    def add_progress(self, images, current_frame, total_frames, bar_position="bottom",
                    bar_height=4, bar_color="#ff6600", bar_background="#333333",
                    show_percentage=False, show_timecodes=False, fps=30.0, font_size=16):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        bar_color_rgb = self._parse_color(bar_color)
        bg_color_rgb = self._parse_color(bar_background)
        progress = current_frame / max(total_frames - 1, 1)
        
        font = load_font(font_size)
        
        for i, img in enumerate(pil_images):
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Draw progress bar
            bar_y = 0 if bar_position == "top" else img.height - bar_height
            
            # Background
            draw.rectangle([0, bar_y, img.width, bar_y + bar_height], 
                         fill=(bg_color_rgb[0], bg_color_rgb[1], bg_color_rgb[2], 200))
            
            # Progress
            progress_width = int(img.width * progress)
            draw.rectangle([0, bar_y, progress_width, bar_y + bar_height],
                         fill=(bar_color_rgb[0], bar_color_rgb[1], bar_color_rgb[2], 255))
            
            # Text overlays
            text_y = bar_y + bar_height + 2 if bar_position == "bottom" else bar_y - font_size - 2
            
            if show_percentage:
                pct_text = f"{int(progress * 100)}%"
                draw.text((10, text_y), pct_text, font=font, fill=(255, 255, 255, 230))
            
            if show_timecodes:
                current_time = current_frame / fps
                total_time = total_frames / fps
                time_text = f"{current_time:.1f}s / {total_time:.1f}s"
                bbox = draw.textbbox((0, 0), time_text, font=font)
                text_w = bbox[2] - bbox[0]
                draw.text((img.width - text_w - 10, text_y), time_text, font=font, 
                         fill=(255, 255, 255, 230))
            
            composited = Image.alpha_composite(img, overlay)
            results.append(composited)
        
        return (pil_to_tensor(results),)
    
    def _parse_color(self, color_str):
        color_str = color_str.strip().lstrip('#')
        if len(color_str) == 3:
            color_str = ''.join([c*2 for c in color_str])
        try:
            return (int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16))
        except:
            return (255, 255, 255)


class OllamaNodesVignette:
    """Add vignette effect to images/video frames."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "vignette_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "vignette_radius": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "vignette_softness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "vignette_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("vignetted",)
    FUNCTION = "apply_vignette"
    CATEGORY = "Ollama-Magic-Nodes/Composition"
    DESCRIPTION = """Add vignette (darkened edges) effect.
    
Common in cinematic/storyboard work:
- Draws focus to center
- Adds dramatic mood
- Simulates lens effects"""
    
    def apply_vignette(self, images, vignette_strength=0.5, vignette_radius=0.7,
                      vignette_softness=0.5, vignette_color="#000000"):
        
        pil_images = tensor_to_pil(images)
        results = []
        color_rgb = self._parse_color(vignette_color)
        
        for img in pil_images:
            w, h = img.size
            
            # Create vignette mask
            X = np.arange(0, w) - w / 2
            Y = np.arange(0, h) - h / 2
            X, Y = np.meshgrid(X, Y)
            
            # Distance from center normalized
            dist = np.sqrt((X / w)**2 + (Y / h)**2) * 2
            
            # Create smooth falloff
            vignette = 1.0 - np.clip((dist - vignette_radius) / vignette_softness, 0, 1) * vignette_strength
            vignette = np.clip(vignette, 0, 1)
            
            # Apply to image
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            img_arr = np.array(img)
            
            # Apply vignette to RGB channels
            for c in range(3):
                img_arr[:, :, c] = (img_arr[:, :, c] * vignette + 
                                   color_rgb[c] * (1 - vignette)).astype(np.uint8)
            
            results.append(Image.fromarray(img_arr))
        
        return (pil_to_tensor(results),)
    
    def _parse_color(self, color_str):
        color_str = color_str.strip().lstrip('#')
        if len(color_str) == 3:
            color_str = ''.join([c*2 for c in color_str])
        try:
            return (int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16))
        except:
            return (0, 0, 0)


class OllamaNodesAnimatedText:
    """Create text with animation effects for video frames."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "Animated Text"}),
                "animation_type": (["typewriter", "fade_in", "fade_out", "slide_in_left",
                                   "slide_in_right", "slide_in_top", "slide_in_bottom",
                                   "scale_in", "bounce_in"],),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "total_frames": ("INT", {"default": 30, "min": 1, "max": 1000}),
            },
            "optional": {
                "font_size": ("INT", {"default": 48, "min": 8, "max": 500}),
                "text_color": ("STRING", {"default": "#ffffff"}),
                "position_x": ("INT", {"default": 50, "min": 0, "max": 4096}),
                "position_y": ("INT", {"default": 50, "min": 0, "max": 4096}),
                "position_mode": (["center", "top_left", "top_center", "top_right",
                                   "bottom_left", "bottom_center", "bottom_right"],),
                "shadow": ("BOOLEAN", {"default": True}),
                "typewriter_speed": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("animated_text",)
    FUNCTION = "animate_text"
    CATEGORY = "Ollama-Magic-Nodes/Composition"
    DESCRIPTION = """Add animated text effects to video frames.
    
Animation Types:
- typewriter: Characters appear one by one
- fade_in/fade_out: Opacity animation
- slide_in_*: Slide from edge
- scale_in: Grow from center
- bounce_in: Elastic bounce effect
    
Connect to OllamaNodesCounter or time-based values for smooth animation."""
    
    def animate_text(self, images, text, animation_type, current_frame, total_frames,
                    font_size=48, text_color="#ffffff", position_x=50, position_y=50,
                    position_mode="center", shadow=True, typewriter_speed=3.0):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        font = load_font(font_size)
        text_rgb = self._parse_color(text_color)
        
        # Calculate animation progress (0-1)
        progress = min(1.0, current_frame / max(total_frames - 1, 1))
        
        for img in pil_images:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Determine what text to show
            if animation_type == "typewriter":
                chars_to_show = int(progress * len(text) * typewriter_speed / 3)
                display_text = text[:min(chars_to_show, len(text))]
            else:
                display_text = text
            
            if not display_text:
                results.append(img)
                continue
            
            txt_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(txt_layer)
            
            # Calculate text size
            bbox = draw.textbbox((0, 0), display_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Base position
            x, y = self._calculate_position(
                position_x, position_y, position_mode,
                img.width, img.height, text_w, text_h
            )
            
            # Apply animation transform
            alpha = 1.0
            offset_x, offset_y = 0, 0
            scale = 1.0
            
            if animation_type == "fade_in":
                alpha = progress
            elif animation_type == "fade_out":
                alpha = 1 - progress
            elif animation_type == "slide_in_left":
                offset_x = int((1 - progress) * -img.width)
            elif animation_type == "slide_in_right":
                offset_x = int((1 - progress) * img.width)
            elif animation_type == "slide_in_top":
                offset_y = int((1 - progress) * -img.height)
            elif animation_type == "slide_in_bottom":
                offset_y = int((1 - progress) * img.height)
            elif animation_type == "scale_in":
                scale = 0.1 + progress * 0.9
                # Adjust position for scaling
                x = x + text_w // 2
                y = y + text_h // 2
            elif animation_type == "bounce_in":
                if progress < 0.6:
                    scale = progress / 0.6 * 1.1
                else:
                    scale = 1.1 - (progress - 0.6) / 0.4 * 0.1
                x = x + text_w // 2
                y = y + text_h // 2
            
            alpha_int = int(alpha * 255)
            
            # Draw shadow
            if shadow and scale == 1.0:
                draw.text((x + 2 + offset_x, y + 2 + offset_y), display_text, font=font,
                         fill=(0, 0, 0, alpha_int // 2))
            
            # Draw text
            if scale != 1.0:
                # Scale text by drawing on a temp image
                temp_size = (int(img.width * 2), int(img.height * 2))
                temp_img = Image.new('RGBA', temp_size, (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_img)
                temp_font = load_font(int(font_size * scale))
                temp_bbox = temp_draw.textbbox((0, 0), display_text, font=temp_font)
                tw, th = temp_bbox[2] - temp_bbox[0], temp_bbox[3] - temp_bbox[1]
                temp_draw.text(((temp_size[0] - tw) // 2, (temp_size[1] - th) // 2), 
                             display_text, font=temp_font,
                             fill=(text_rgb[0], text_rgb[1], text_rgb[2], alpha_int))
                
                # Paste centered
                txt_layer.paste(temp_img, (int(img.width * 0.5 - temp_size[0] * 0.5 + offset_x),
                                          int(img.height * 0.5 - temp_size[1] * 0.5 + offset_y)))
            else:
                draw.text((x + offset_x, y + offset_y), display_text, font=font,
                         fill=(text_rgb[0], text_rgb[1], text_rgb[2], alpha_int))
            
            composited = Image.alpha_composite(img, txt_layer)
            results.append(composited)
        
        return (pil_to_tensor(results),)
    
    def _parse_color(self, color_str):
        color_str = color_str.strip().lstrip('#')
        if len(color_str) == 3:
            color_str = ''.join([c*2 for c in color_str])
        try:
            return (int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16))
        except:
            return (255, 255, 255)
    
    def _calculate_position(self, px, py, mode, img_w, img_h, text_w, text_h):
        if mode == "center":
            return (img_w - text_w) // 2, (img_h - text_h) // 2
        elif mode == "top_left":
            return 20, 20
        elif mode == "top_center":
            return (img_w - text_w) // 2, 20
        elif mode == "top_right":
            return img_w - text_w - 20, 20
        elif mode == "bottom_left":
            return 20, img_h - text_h - 20
        elif mode == "bottom_center":
            return (img_w - text_w) // 2, img_h - text_h - 20
        elif mode == "bottom_right":
            return img_w - text_w - 20, img_h - text_h - 20
        return px, py


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesTextOverlay": OllamaNodesTextOverlay,
    "OllamaNodesImageOverlay": OllamaNodesImageOverlay,
    "OllamaNodesComposeLayers": OllamaNodesComposeLayers,
    "OllamaNodesTextBadge": OllamaNodesTextBadge,
    "OllamaNodesVideoProgress": OllamaNodesVideoProgress,
    "OllamaNodesVignette": OllamaNodesVignette,
    "OllamaNodesAnimatedText": OllamaNodesAnimatedText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesTextOverlay": "🖌 Text Overlay",
    "OllamaNodesImageOverlay": "🖌 Image Overlay",
    "OllamaNodesComposeLayers": "🖌 Compose Layers",
    "OllamaNodesTextBadge": "🖌 Text Badge",
    "OllamaNodesVideoProgress": "🖌 Video Progress Bar",
    "OllamaNodesVignette": "🖌 Vignette Effect",
    "OllamaNodesAnimatedText": "🖌 Animated Text",
}
