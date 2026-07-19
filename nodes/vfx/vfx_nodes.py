"""
OllamaNodes - VFX & Visual Effects Nodes
Particle systems, shape animations, screen effects, light effects, and procedural patterns.
Generates effects on transparent or solid backgrounds for compositing.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import random
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
        if img.mode == 'RGBA':
            # Composite onto black for RGB output
            bg = Image.new('RGB', img.size, (0, 0, 0))
            bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))
    return torch.stack(tensors)


def pil_to_tensor_with_alpha(images):
    """Convert RGBA PIL Images to tensor, keeping alpha as 3rd channel concept."""
    if not isinstance(images, list):
        images = [images]
    tensors = []
    for img in images:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))
    return torch.stack(tensors)


def parse_color(hex_str):
    """Parse hex color to RGB tuple."""
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c*2 for c in hex_str])
    try:
        return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
    except:
        return (255, 255, 255)


# ============================================================
# PARTICLE SYSTEMS
# ============================================================

class OllamaNodesParticleRain:
    """Generate rain particle effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "density": ("INT", {"default": 100, "min": 10, "max": 500}),
                "drop_length": ("INT", {"default": 15, "min": 2, "max": 50}),
                "drop_width": ("INT", {"default": 1, "min": 1, "max": 4}),
                "speed": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 50.0}),
                "wind": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0}),
                "color": ("STRING", {"default": "#aaaaff"}),
                "opacity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "angle": ("FLOAT", {"default": 15.0, "min": -45.0, "max": 45.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rain_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Particles"
    DESCRIPTION = """Generate rain particle effect.
    
Parameters:
- density: Number of rain drops
- speed: How fast drops fall
- wind: Horizontal wind effect
- angle: Rain angle from vertical"""

    def generate(self, width, height, frame, density=100, drop_length=15, 
                 drop_width=1, speed=15.0, wind=0.0, color="#aaaaff", opacity=0.7,
                 angle=15.0, seed=42):
        
        rng = random.Random(seed)
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        color_rgb = parse_color(color)
        alpha = int(opacity * 255)
        
        angle_rad = math.radians(angle)
        
        for i in range(density):
            # Each drop has unique random properties
            rng.seed(seed + i)
            
            # Initial position (distributed across screen)
            base_x = rng.random() * width * 1.5 - width * 0.25
            base_y = rng.random() * height * 2 - height
            
            # Calculate current position based on frame
            current_y = (base_y + frame * speed + i * 17) % (height + drop_length * 2)
            current_x = base_x + frame * wind + math.sin(frame * 0.05 + i) * 2
            
            # Apply rain angle
            current_x += current_y * math.tan(angle_rad) * 0.3
            
            # Draw drop
            end_x = current_x + math.sin(angle_rad) * drop_length
            end_y = current_y - math.cos(angle_rad) * drop_length
            
            draw.line([(current_x, current_y), (end_x, end_y)],
                     fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha),
                     width=drop_width)
        
        return (pil_to_tensor([img]),)


class OllamaNodesParticleSnow:
    """Generate snow particle effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "density": ("INT", {"default": 150, "min": 10, "max": 500}),
                "flake_size": ("INT", {"default": 3, "min": 1, "max": 8}),
                "speed": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0}),
                "wobble": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0}),
                "color": ("STRING", {"default": "#ffffff"}),
                "opacity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("snow_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Particles"
    DESCRIPTION = """Generate snow particle effect with wobble motion."""

    def generate(self, width, height, frame, density=150, flake_size=3,
                 speed=2.0, wobble=30.0, color="#ffffff", opacity=0.9, seed=123):
        
        rng = random.Random(seed)
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        color_rgb = parse_color(color)
        alpha = int(opacity * 255)
        
        for i in range(density):
            rng.seed(seed + i)
            
            base_x = rng.random() * width
            base_y = rng.random() * height * 2 - height * 0.5
            wobble_speed = rng.uniform(0.02, 0.08)
            wobble_amp = rng.uniform(wobble * 0.5, wobble)
            
            # Calculate position with wobble
            current_y = (base_y + frame * speed) % (height + 20)
            current_x = base_x + math.sin(frame * wobble_speed + i * 0.5) * wobble_amp
            
            # Flake size variation
            size = rng.randint(max(1, flake_size - 1), flake_size + 1)
            
            # Draw flake (circle)
            draw.ellipse([current_x - size, current_y - size, 
                         current_x + size, current_y + size],
                        fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha))
        
        return (pil_to_tensor([img]),)


class OllamaNodesParticleFire:
    """Generate fire/flame particle effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "base_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "base_y": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "density": ("INT", {"default": 80, "min": 20, "max": 300}),
                "flame_height": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.9}),
                "flame_width": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5}),
                "speed": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0}),
                "turbulence": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 50.0}),
                "color_inner": ("STRING", {"default": "#ffff00"}),
                "color_mid": ("STRING", {"default": "#ff6600"}),
                "color_outer": ("STRING", {"default": "#ff2200"}),
                "opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 456, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("fire_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Particles"
    DESCRIPTION = """Generate fire/flame particle effect with color gradient."""

    def generate(self, width, height, frame, base_x=0.5, base_y=0.9,
                 density=80, flame_height=0.4, flame_width=0.15, speed=3.0,
                 turbulence=20.0, color_inner="#ffff00", color_mid="#ff6600",
                 color_outer="#ff2200", opacity=0.8, seed=456):
        
        rng = random.Random(seed)
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        color_in = parse_color(color_inner)
        color_mid = parse_color(color_mid)
        color_out = parse_color(color_outer)
        
        for i in range(density):
            rng.seed(seed + i)
            
            # Fire originates from base point
            origin_x = base_x * width + rng.uniform(-flame_width * width / 2, flame_width * width / 2)
            origin_y = base_y * height
            
            # Particle rises upward
            lifetime = rng.uniform(0.5, 1.0)
            progress = ((frame * speed * 0.01 + rng.random()) % 1.0)
            
            # Position
            current_y = origin_y - progress * flame_height * height
            turb_x = math.sin(frame * 0.1 + i * 0.3) * turbulence + rng.uniform(-5, 5)
            current_x = origin_x + turb_x * (1 - progress)
            
            # Size (shrinks as it rises)
            size = (1 - progress * 0.7) * flame_width * width * rng.uniform(0.3, 0.8)
            
            # Color based on height
            if progress < 0.3:
                c = color_in
            elif progress < 0.6:
                t = (progress - 0.3) / 0.3
                c = tuple(int(color_in[j] * (1-t) + color_mid[j] * t) for j in range(3))
            else:
                t = (progress - 0.6) / 0.4
                c = tuple(int(color_mid[j] * (1-t) + color_out[j] * t) for j in range(3))
            
            # Alpha fades with height
            alpha = int(opacity * 255 * (1 - progress) * (1 - progress))
            
            draw = ImageDraw.Draw(img)
            draw.ellipse([current_x - size, current_y - size,
                         current_x + size, current_y + size * 0.6],
                        fill=(c[0], c[1], c[2], alpha))
        
        # Blur for glow effect
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        return (pil_to_tensor([img]),)


class OllamaNodesParticleSparks:
    """Generate sparks/embers particle effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "base_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "base_y": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "density": ("INT", {"default": 50, "min": 10, "max": 200}),
                "spread": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 300.0}),
                "speed": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 5.0}),
                "gravity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "color": ("STRING", {"default": "#ffaa00"}),
                "glow_color": ("STRING", {"default": "#ff6600"}),
                "spark_size": ("INT", {"default": 2, "min": 1, "max": 5}),
                "trail_length": ("INT", {"default": 5, "min": 0, "max": 15}),
                "seed": ("INT", {"default": 789, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sparks_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Particles"
    DESCRIPTION = """Generate sparks/embers particle effect with trails."""

    def generate(self, width, height, frame, base_x=0.5, base_y=0.9,
                 density=50, spread=100.0, speed=2.0, gravity=0.5,
                 color="#ffaa00", glow_color="#ff6600", spark_size=2,
                 trail_length=5, seed=789):
        
        rng = random.Random(seed)
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        spark_rgb = parse_color(color)
        glow_rgb = parse_color(glow_color)
        
        for i in range(density):
            rng.seed(seed + i)
            
            # Random velocity
            vx = rng.uniform(-spread, spread) * 0.01
            vy = rng.uniform(-spread, -spread * 0.3) * 0.01
            
            # Loop animation
            cycle_length = rng.uniform(30, 60)
            t = (frame * speed * 0.02 + rng.random() * cycle_length) % cycle_length
            progress = t / cycle_length
            
            # Position with gravity
            current_x = base_x * width + vx * t * 10
            current_y = base_y * height + vy * t * 10 + gravity * t * t * 0.5
            
            # Skip if off screen
            if current_x < 0 or current_x > width or current_y < 0 or current_y > height:
                continue
            
            # Draw trail
            if trail_length > 0:
                for j in range(trail_length):
                    trail_progress = max(0, progress - j * 0.01)
                    trail_y = base_y * height + vy * (t - j * 2) * 10 + gravity * (t - j * 2) * (t - j * 2) * 0.5
                    trail_x = current_x - vx * j * 2
                    trail_alpha = int(255 * (1 - j / trail_length) * 0.5)
                    draw.point([trail_x, trail_y], fill=(glow_rgb[0], glow_rgb[1], glow_rgb[2], trail_alpha))
            
            # Draw spark
            alpha = int(255 * (1 - progress))
            draw.ellipse([current_x - spark_size, current_y - spark_size,
                         current_x + spark_size, current_y + spark_size],
                        fill=(spark_rgb[0], spark_rgb[1], spark_rgb[2], alpha))
        
        # Add glow
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        return (pil_to_tensor([img]),)


class OllamaNodesParticleConfetti:
    """Generate confetti particle effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "density": ("INT", {"default": 100, "min": 20, "max": 300}),
                "colors": ("STRING", {"default": "#ff0000,#00ff00,#0000ff,#ffff00,#ff00ff"}),
                "size": ("INT", {"default": 8, "min": 3, "max": 20}),
                "speed": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0}),
                "wobble": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 60.0}),
                "rotation_speed": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.3}),
                "gravity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "shape": (["rectangle", "circle", "strip"],),
                "seed": ("INT", {"default": 321, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("confetti_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Particles"
    DESCRIPTION = """Generate colorful confetti particle effect."""

    def generate(self, width, height, frame, density=100,
                 colors="#ff0000,#00ff00,#0000ff,#ffff00,#ff00ff",
                 size=8, speed=3.0, wobble=30.0, rotation_speed=0.1,
                 gravity=0.3, shape="rectangle", seed=321):
        
        rng = random.Random(seed)
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        color_list = [parse_color(c.strip()) for c in colors.split(',')]
        
        for i in range(density):
            rng.seed(seed + i)
            
            # Initial random position (from top)
            start_x = rng.random() * width
            start_y = rng.random() * height * 0.3 - height * 0.3
            
            # Movement
            vx = rng.uniform(-1, 1)
            fall_speed = rng.uniform(speed * 0.5, speed * 1.5)
            wobble_phase = rng.random() * math.pi * 2
            rot_speed = rng.uniform(-rotation_speed, rotation_speed)
            
            # Current position
            t = frame * 0.1
            current_x = start_x + vx * t + math.sin(t * 2 + wobble_phase) * wobble
            current_y = (start_y + fall_speed * t + gravity * t * t * 0.5) % (height + 50) - 25
            
            # Rotation
            rotation = t * rot_speed * 180 / math.pi
            
            # Choose color
            color = color_list[i % len(color_list)]
            
            # Create confetti piece
            confetti = Image.new('RGBA', (size * 2, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(confetti)
            
            if shape == "rectangle":
                draw.rectangle([0, 0, size * 2, size], fill=(*color, 220))
            elif shape == "circle":
                draw.ellipse([0, 0, size, size], fill=(*color, 220))
            elif shape == "strip":
                draw.rectangle([0, 0, size * 3, size // 2], fill=(*color, 220))
            
            # Rotate
            confetti = confetti.rotate(rotation, expand=True, resample=Image.BICUBIC)
            
            # Paste onto main image
            paste_x = int(current_x - confetti.width // 2)
            paste_y = int(current_y - confetti.height // 2)
            
            img.paste(confetti, (paste_x, paste_y), confetti)
        
        return (pil_to_tensor([img]),)


class OllamaNodesParticleDust:
    """Generate floating dust/bokeh particle effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "density": ("INT", {"default": 50, "min": 10, "max": 200}),
                "min_size": ("INT", {"default": 2, "min": 1, "max": 10}),
                "max_size": ("INT", {"default": 10, "min": 3, "max": 30}),
                "speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0}),
                "float_amplitude": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 50.0}),
                "color": ("STRING", {"default": "#ffffff"}),
                "opacity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0}),
                "glow": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 654, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("dust_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Particles"
    DESCRIPTION = """Generate floating dust/bokeh particle effect for atmosphere."""

    def generate(self, width, height, frame, density=50, min_size=2, max_size=10,
                 speed=0.5, float_amplitude=20.0, color="#ffffff", opacity=0.4,
                 glow=True, seed=654):
        
        rng = random.Random(seed)
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        color_rgb = parse_color(color)
        
        for i in range(density):
            rng.seed(seed + i)
            
            base_x = rng.random() * width
            base_y = rng.random() * height
            float_speed = rng.uniform(0.01, 0.05)
            float_phase = rng.random() * math.pi * 2
            size = rng.randint(min_size, max_size)
            
            # Floating motion
            current_x = base_x + math.sin(frame * float_speed + float_phase) * float_amplitude
            current_y = base_y + math.cos(frame * float_speed * 0.7 + float_phase) * float_amplitude * 0.5
            current_y -= frame * speed * 0.1  # Slow upward drift
            
            # Wrap around
            current_x = current_x % width
            current_y = current_y % height
            
            # Alpha varies
            alpha = int(opacity * 255 * rng.uniform(0.5, 1.0))
            
            if glow:
                # Draw glow first
                glow_size = size * 3
                glow_alpha = alpha // 3
                draw.ellipse([current_x - glow_size, current_y - glow_size,
                             current_x + glow_size, current_y + glow_size],
                            fill=(color_rgb[0], color_rgb[1], color_rgb[2], glow_alpha))
            
            # Draw particle
            draw.ellipse([current_x - size, current_y - size,
                         current_x + size, current_y + size],
                        fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha))
        
        if glow:
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        return (pil_to_tensor([img]),)


# ============================================================
# SHAPE ANIMATIONS
# ============================================================

class OllamaNodesShapePulse:
    """Generate pulsing circle/ring animation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "shape": (["circle", "ring", "square", "diamond"],),
                "color": ("STRING", {"default": "#ff6600"}),
                "base_size": ("INT", {"default": 50, "min": 10, "max": 300}),
                "pulse_amount": ("INT", {"default": 30, "min": 0, "max": 100}),
                "pulse_speed": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5}),
                "ring_width": ("INT", {"default": 5, "min": 1, "max": 30}),
                "opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "glow": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("shape_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Shapes"
    DESCRIPTION = """Generate pulsing shape animation (circle, ring, square, diamond)."""

    def generate(self, width, height, frame, center_x=0.5, center_y=0.5,
                 shape="circle", color="#ff6600", base_size=50, pulse_amount=30,
                 pulse_speed=0.1, ring_width=5, opacity=0.8, glow=False):
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        color_rgb = parse_color(color)
        alpha = int(opacity * 255)
        
        # Pulsing size
        pulse = math.sin(frame * pulse_speed) * pulse_amount
        size = base_size + pulse
        
        cx = center_x * width
        cy = center_y * height
        
        bbox = [cx - size, cy - size, cx + size, cy + size]
        
        if glow:
            # Draw glow
            glow_size = size * 1.3
            glow_bbox = [cx - glow_size, cy - glow_size, cx + glow_size, cy + glow_size]
            draw.ellipse(glow_bbox, fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha // 3))
        
        if shape == "circle":
            draw.ellipse(bbox, fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha))
        elif shape == "ring":
            # Outer circle
            draw.ellipse(bbox, fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha))
            # Inner circle (transparent)
            inner_size = size - ring_width
            inner_bbox = [cx - inner_size, cy - inner_size, cx + inner_size, cy + inner_size]
            draw.ellipse(inner_bbox, fill=(0, 0, 0, 0))
        elif shape == "square":
            draw.rectangle(bbox, fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha))
        elif shape == "diamond":
            points = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
            draw.polygon(points, fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha))
        
        if glow:
            img = img.filter(ImageFilter.GaussianBlur(radius=3))
        
        return (pil_to_tensor([img]),)


class OllamaNodesShapeWave:
    """Generate animated wave/sine pattern."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "color": ("STRING", {"default": "#00aaff"}),
                "wave_type": (["sine", "triangle", "square"],),
                "amplitude": ("INT", {"default": 30, "min": 5, "max": 100}),
                "frequency": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.2}),
                "speed": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5}),
                "wave_width": ("INT", {"default": 4, "min": 1, "max": 20}),
                "num_waves": ("INT", {"default": 3, "min": 1, "max": 10}),
                "wave_spacing": ("INT", {"default": 30, "min": 10, "max": 100}),
                "opacity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("wave_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Shapes"
    DESCRIPTION = """Generate animated wave pattern."""

    def generate(self, width, height, frame, color="#00aaff", wave_type="sine",
                 amplitude=30, frequency=0.05, speed=0.1, wave_width=4,
                 num_waves=3, wave_spacing=30, opacity=0.7):
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        color_rgb = parse_color(color)
        alpha = int(opacity * 255)
        
        center_y = height // 2
        
        for wave_idx in range(num_waves):
            wave_offset = (wave_idx - num_waves // 2) * wave_spacing
            
            points = []
            for x in range(0, width, 2):
                # Calculate y based on wave type
                base_y = center_y + wave_offset
                
                if wave_type == "sine":
                    y = base_y + amplitude * math.sin(x * frequency + frame * speed + wave_idx * 0.5)
                elif wave_type == "triangle":
                    t = (x * frequency + frame * speed + wave_idx * 0.5) % (2 * math.pi)
                    y = base_y + amplitude * (2 * abs(t / math.pi - 1) - 1)
                elif wave_type == "square":
                    y = base_y + amplitude * (1 if math.sin(x * frequency + frame * speed + wave_idx * 0.5) > 0 else -1)
                
                points.append((x, y))
            
            # Draw wave line
            if len(points) > 1:
                # Fade alpha based on wave index
                wave_alpha = int(alpha * (1 - abs(wave_idx - num_waves // 2) / (num_waves // 2 + 1)))
                draw.line(points, fill=(color_rgb[0], color_rgb[1], color_rgb[2], wave_alpha),
                         width=wave_width)
        
        if wave_width > 3:
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        return (pil_to_tensor([img]),)


class OllamaNodesShapeRings:
    """Generate expanding ring animation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "color": ("STRING", {"default": "#ffffff"}),
                "num_rings": ("INT", {"default": 3, "min": 1, "max": 10}),
                "max_radius": ("INT", {"default": 200, "min": 50, "max": 400}),
                "ring_width": ("INT", {"default": 3, "min": 1, "max": 15}),
                "speed": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 5.0}),
                "fade_out": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rings_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Shapes"
    DESCRIPTION = """Generate expanding ring animation (radar/sonar style)."""

    def generate(self, width, height, frame, center_x=0.5, center_y=0.5,
                 color="#ffffff", num_rings=3, max_radius=200, ring_width=3,
                 speed=2.0, fade_out=True):
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        color_rgb = parse_color(color)
        
        cx = center_x * width
        cy = center_y * height
        
        cycle_length = max_radius / speed
        
        for i in range(num_rings):
            # Each ring is offset in time
            ring_time = (frame + i * cycle_length / num_rings) % cycle_length
            radius = ring_time * speed
            
            if radius > max_radius:
                continue
            
            # Alpha fades as ring expands
            if fade_out:
                alpha = int(255 * (1 - radius / max_radius))
            else:
                alpha = 255
            
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(bbox, outline=(color_rgb[0], color_rgb[1], color_rgb[2], alpha),
                        width=ring_width)
        
        return (pil_to_tensor([img]),)


class OllamaNodesShapeGrid:
    """Generate animated grid pattern."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "color": ("STRING", {"default": "#00ff88"}),
                "grid_spacing": ("INT", {"default": 40, "min": 10, "max": 100}),
                "line_width": ("INT", {"default": 1, "min": 1, "max": 5}),
                "animation_type": (["scroll_x", "scroll_y", "pulse", "fade_pulse"],),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0}),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "perspective": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid_effect",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Shapes"
    DESCRIPTION = """Generate animated grid pattern (sci-fi/cyberpunk style)."""

    def generate(self, width, height, frame, color="#00ff88", grid_spacing=40,
                 line_width=1, animation_type="scroll_x", speed=1.0, opacity=0.5,
                 perspective=False):
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        color_rgb = parse_color(color)
        alpha = int(opacity * 255)
        
        if animation_type == "scroll_x":
            offset = int(frame * speed) % grid_spacing
        elif animation_type == "scroll_y":
            offset = int(frame * speed) % grid_spacing
        else:
            offset = 0
        
        # Calculate alpha for pulse animations
        if animation_type == "pulse":
            pulse = (math.sin(frame * 0.05 * speed) + 1) / 2
            alpha = int(alpha * pulse)
        elif animation_type == "fade_pulse":
            pulse = (math.sin(frame * 0.1 * speed) + 1) / 2
            alpha = int(alpha * pulse)
        
        # Draw vertical lines
        for x in range(-grid_spacing, width + grid_spacing, grid_spacing):
            actual_x = (x + offset) if animation_type == "scroll_x" else x
            
            if perspective:
                # Perspective lines converge to center
                draw.line([(actual_x, 0), (width // 2, height // 2)],
                         fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha // 2),
                         width=line_width)
                draw.line([(actual_x, height), (width // 2, height // 2)],
                         fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha // 2),
                         width=line_width)
            else:
                draw.line([(actual_x, 0), (actual_x, height)],
                         fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha),
                         width=line_width)
        
        # Draw horizontal lines
        for y in range(-grid_spacing, height + grid_spacing, grid_spacing):
            actual_y = (y + offset) if animation_type == "scroll_y" else y
            
            draw.line([(0, actual_y), (width, actual_y)],
                     fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha),
                     width=line_width)
        
        return (pil_to_tensor([img]),)


# ============================================================
# SCREEN EFFECTS
# ============================================================

class OllamaNodesEffectGlitch:
    """Apply glitch/distortion effect to images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "num_slices": ("INT", {"default": 10, "min": 1, "max": 30}),
                "color_shift": ("BOOLEAN", {"default": True}),
                "color_shift_amount": ("INT", {"default": 10, "min": 1, "max": 30}),
                "block_glitch": ("BOOLEAN", {"default": True}),
                "block_size": ("INT", {"default": 20, "min": 5, "max": 50}),
                "scanlines": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("glitched",)
    FUNCTION = "apply_glitch"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Screen"
    DESCRIPTION = """Apply glitch/distortion effect to images/video frames."""

    def apply_glitch(self, images, frame, intensity=0.3, num_slices=10,
                    color_shift=True, color_shift_amount=10,
                    block_glitch=True, block_size=20,
                    scanlines=False, seed=42):
        
        pil_images = tensor_to_pil(images)
        results = []
        rng = random.Random(seed + frame)
        
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_arr = np.array(img)
            h, w, c = img_arr.shape
            
            # Slice displacement
            slice_height = h // num_slices
            for i in range(num_slices):
                if rng.random() < intensity * 0.5:
                    y_start = i * slice_height
                    y_end = min((i + 1) * slice_height, h)
                    displacement = rng.randint(-20, 20) * int(intensity * 3)
                    img_arr[y_start:y_end] = np.roll(img_arr[y_start:y_end], displacement, axis=1)
            
            # Block glitch
            if block_glitch:
                num_blocks = int(intensity * 10)
                for _ in range(num_blocks):
                    bx = rng.randint(0, w - block_size)
                    by = rng.randint(0, h - block_size)
                    if rng.random() < intensity:
                        # Replace with random color block
                        color = rng.randint(0, 2, size=(block_size, block_size, 3)) * 255
                        img_arr[by:by+block_size, bx:bx+block_size] = color
            
            # Color shift (RGB channel offset)
            if color_shift and intensity > 0.1:
                shift = int(color_shift_amount * intensity)
                img_arr[:, :, 0] = np.roll(img_arr[:, :, 0], shift, axis=1)  # Red shift
                img_arr[:, :, 2] = np.roll(img_arr[:, :, 2], -shift, axis=1)  # Blue shift
            
            result = Image.fromarray(img_arr)
            
            # Scanlines
            if scanlines:
                draw = ImageDraw.Draw(result)
                for y in range(0, h, 2):
                    draw.line([(0, y), (w, y)], fill=(0, 0, 0, 40), width=1)
            
            results.append(result)
        
        return (pil_to_tensor(results),)


class OllamaNodesEffectChromatic:
    """Apply chromatic aberration effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "strength": ("INT", {"default": 5, "min": 0, "max": 30}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("chromatic_aberration",)
    FUNCTION = "apply"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Screen"
    DESCRIPTION = """Apply chromatic aberration (color fringing) effect."""

    def apply(self, images, strength=5, center_x=0.5, center_y=0.5):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_arr = np.array(img).astype(float)
            h, w, _ = img_arr.shape
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # Distance from center
            cx, cy = center_x * w, center_y * h
            dist_x = (x_coords - cx) / (w / 2)
            dist_y = (y_coords - cy) / (h / 2)
            dist = np.sqrt(dist_x**2 + dist_y**2)
            
            # Channel offsets
            red_shift = strength * dist
            blue_shift = -strength * dist
            
            # Apply shifts
            result = np.zeros_like(img_arr)
            for y in range(h):
                for x in range(w):
                    # Red channel (shift outward)
                    rx = min(max(0, int(x + dist_x[y, x] * strength)), w-1)
                    ry = min(max(0, int(y + dist_y[y, x] * strength)), h-1)
                    result[y, x, 0] = img_arr[ry, rx, 0]
                    
                    # Green channel (no shift)
                    result[y, x, 1] = img_arr[y, x, 1]
                    
                    # Blue channel (shift inward)
                    bx = min(max(0, int(x - dist_x[y, x] * strength)), w-1)
                    by = min(max(0, int(y - dist_y[y, x] * strength)), h-1)
                    result[y, x, 2] = img_arr[by, bx, 2]
            
            results.append(Image.fromarray(result.astype(np.uint8)))
        
        return (pil_to_tensor(results),)


class OllamaNodesEffectScanlines:
    """Apply scanline effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "scanline_spacing": ("INT", {"default": 4, "min": 2, "max": 20}),
                "opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "scroll": ("BOOLEAN", {"default": True}),
                "speed": ("INT", {"default": 1, "min": 0, "max": 5}),
                "horizontal": ("BOOLEAN", {"default": False}),
                "color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scanlined",)
    FUNCTION = "apply"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Screen"
    DESCRIPTION = """Apply CRT/TV scanline effect."""

    def apply(self, images, frame, scanline_spacing=4, opacity=0.3,
             scroll=True, speed=1, horizontal=False, color="#000000"):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        color_rgb = parse_color(color)
        alpha = int(opacity * 255)
        
        for img in pil_images:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            offset = (frame * speed) % scanline_spacing if scroll else 0
            
            if horizontal:
                for y in range(-offset, img.height, scanline_spacing):
                    draw.line([(0, y), (img.width, y)],
                             fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha))
            else:
                for x in range(-offset, img.width, scanline_spacing):
                    draw.line([(x, 0), (x, img.height)],
                             fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha))
            
            results.append(Image.alpha_composite(img, overlay))
        
        return (pil_to_tensor(results),)


class OllamaNodesEffectNoise:
    """Apply film grain/noise effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5}),
                "monochrome": ("BOOLEAN", {"default": False}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("noisy",)
    FUNCTION = "apply"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Screen"
    DESCRIPTION = """Apply film grain/noise effect."""

    def apply(self, images, frame, amount=0.1, monochrome=False,
             speed=1.0, seed=42):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_arr = np.array(img).astype(float)
            h, w, c = img_arr.shape
            
            # Generate noise
            np.random.seed(int(seed + frame * speed))
            
            if monochrome:
                noise = np.random.normal(0, amount * 128, (h, w, 1))
                noise = np.repeat(noise, 3, axis=2)
            else:
                noise = np.random.normal(0, amount * 128, (h, w, c))
            
            # Apply noise
            result = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            results.append(Image.fromarray(result))
        
        return (pil_to_tensor(results),)


class OllamaNodesEffectVHS:
    """Apply VHS/retro tape effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "tracking_lines": ("BOOLEAN", {"default": True}),
                "color_bleed": ("BOOLEAN", {"default": True}),
                "desaturation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "noise_amount": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.2}),
                "warp_amount": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("vhs_effect",)
    FUNCTION = "apply"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Screen"
    DESCRIPTION = """Apply VHS/retro tape effect with tracking and color bleed."""

    def apply(self, images, frame, tracking_lines=True, color_bleed=True,
             desaturation=0.2, noise_amount=0.05, warp_amount=0.001, seed=42):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_arr = np.array(img).astype(float)
            h, w, c = img_arr.shape
            
            # Tracking distortion (horizontal offset at certain rows)
            if tracking_lines:
                num_glitches = 3
                for _ in range(num_glitches):
                    y = (frame * 7 + _ * 50) % h
                    height = 5 + (frame * 3 + _) % 10
                    offset = (10 + (frame * 2) % 20) * (1 if _ % 2 == 0 else -1)
                    y_end = min(y + height, h)
                    img_arr[y:y_end] = np.roll(img_arr[y:y_end], int(offset * w * warp_amount), axis=1)
            
            # Color bleed (horizontal blur on color channels)
            if color_bleed:
                for c_idx in [0, 2]:  # R and B channels
                    shifted = np.roll(img_arr[:, :, c_idx], 2, axis=1)
                    img_arr[:, :, c_idx] = img_arr[:, :, c_idx] * 0.7 + shifted * 0.3
            
            # Desaturation
            if desaturation > 0:
                gray = np.mean(img_arr, axis=2, keepdims=True)
                img_arr = img_arr * (1 - desaturation) + gray * desaturation
            
            # Noise
            if noise_amount > 0:
                np.random.seed(seed + frame)
                noise = np.random.normal(0, noise_amount * 128, img_arr.shape)
                img_arr = np.clip(img_arr + noise, 0, 255)
            
            # Slight color shift towards warm
            img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] * 1.05, 0, 255)
            img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] * 0.95, 0, 255)
            
            results.append(Image.fromarray(img_arr.astype(np.uint8)))
        
        return (pil_to_tensor(results),)


class OllamaNodesEffectSpeedLines:
    """Generate anime-style speed lines effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": {
                "color": ("STRING", {"default": "#ffffff"}),
                "num_lines": ("INT", {"default": 30, "min": 5, "max": 100}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "min_length": ("INT", {"default": 50, "min": 10, "max": 200}),
                "max_length": ("INT", {"default": 200, "min": 50, "max": 400}),
                "speed": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 5.0}),
                "fade": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("speed_lines",)
    FUNCTION = "generate"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Screen"
    DESCRIPTION = """Generate anime-style speed lines for action scenes."""

    def generate(self, width, height, frame, color="#ffffff", num_lines=30,
                center_x=0.5, center_y=0.5, min_length=50, max_length=200,
                speed=2.0, fade=True, seed=42):
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        color_rgb = parse_color(color)
        
        cx = center_x * width
        cy = center_y * height
        
        rng = random.Random(seed)
        
        for i in range(num_lines):
            rng.seed(seed + i)
            
            # Random angle
            angle = rng.uniform(0, math.pi * 2)
            
            # Random position along radius
            start_dist = rng.uniform(0, min_length)
            end_dist = rng.uniform(min_length, max_length)
            
            # Animate
            anim_offset = (frame * speed * 2) % (max_length - min_length)
            start_dist = (start_dist + anim_offset) % max_length
            end_dist = start_dist + rng.uniform(min_length, max_length - min_length)
            
            # Calculate positions
            x1 = cx + math.cos(angle) * start_dist
            y1 = cy + math.sin(angle) * start_dist
            x2 = cx + math.cos(angle) * end_dist
            y2 = cy + math.sin(angle) * end_dist
            
            # Alpha based on distance
            if fade:
                alpha = int(255 * (1 - start_dist / max_length))
            else:
                alpha = 255
            
            draw.line([(x1, y1), (x2, y2)],
                     fill=(color_rgb[0], color_rgb[1], color_rgb[2], alpha),
                     width=rng.randint(1, 3))
        
        # Blur for smoother look
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        return (pil_to_tensor([img]),)


class OllamaNodesEffectLensFlare:
    """Generate lens flare effect."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "position_x": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "position_y": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "flares": ("INT", {"default": 5, "min": 1, "max": 10}),
                "main_color": ("STRING", {"default": "#ffffcc"}),
                "secondary_color": ("STRING", {"default": "#ff6600"}),
                "intensity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5}),
                "spread": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("with_lens_flare",)
    FUNCTION = "apply"
    CATEGORY = "Ollama-Magic-Nodes/VFX/Light"
    DESCRIPTION = """Add lens flare effect to image."""

    def apply(self, images, position_x=0.3, position_y=0.3, flares=5,
             main_color="#ffffcc", secondary_color="#ff6600",
             intensity=0.8, spread=0.3):
        
        pil_images = tensor_to_pil(images)
        results = []
        
        main_rgb = parse_color(main_color)
        sec_rgb = parse_color(secondary_color)
        
        for img in pil_images:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            w, h = img.size
            cx = position_x * w
            cy = position_y * h
            
            # Main glow
            glow_size = min(w, h) * 0.15 * intensity
            for r in range(int(glow_size), 0, -2):
                alpha = int(100 * intensity * (1 - r / glow_size))
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                           fill=(main_rgb[0], main_rgb[1], main_rgb[2], alpha))
            
            # Secondary flares along line through center
            for i in range(flares):
                t = (i + 1) / (flares + 1)
                fx = cx + (cx - w/2) * t * spread * 5
                fy = cy + (cy - h/2) * t * spread * 5
                
                flare_size = 5 + i * 3
                alpha = int(150 * intensity * (1 - t * 0.5))
                
                color = main_rgb if i % 2 == 0 else sec_rgb
                draw.ellipse([fx - flare_size, fy - flare_size,
                            fx + flare_size, fy + flare_size],
                           fill=(color[0], color[1], color[2], alpha))
            
            # Rainbow ring
            ring_size = min(w, h) * spread
            for angle in range(0, 360, 30):
                rx = cx + math.cos(math.radians(angle)) * ring_size
                ry = cy + math.sin(math.radians(angle)) * ring_size
                draw.ellipse([rx - 3, ry - 3, rx + 3, ry + 3],
                           fill=(sec_rgb[0], sec_rgb[1], sec_rgb[2], 100))
            
            # Blur for glow
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=5))
            
            results.append(Image.alpha_composite(img, overlay))
        
        return (pil_to_tensor(results),)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesParticleRain": OllamaNodesParticleRain,
    "OllamaNodesParticleSnow": OllamaNodesParticleSnow,
    "OllamaNodesParticleFire": OllamaNodesParticleFire,
    "OllamaNodesParticleSparks": OllamaNodesParticleSparks,
    "OllamaNodesParticleConfetti": OllamaNodesParticleConfetti,
    "OllamaNodesParticleDust": OllamaNodesParticleDust,
    "OllamaNodesShapePulse": OllamaNodesShapePulse,
    "OllamaNodesShapeWave": OllamaNodesShapeWave,
    "OllamaNodesShapeRings": OllamaNodesShapeRings,
    "OllamaNodesShapeGrid": OllamaNodesShapeGrid,
    "OllamaNodesEffectGlitch": OllamaNodesEffectGlitch,
    "OllamaNodesEffectChromatic": OllamaNodesEffectChromatic,
    "OllamaNodesEffectScanlines": OllamaNodesEffectScanlines,
    "OllamaNodesEffectNoise": OllamaNodesEffectNoise,
    "OllamaNodesEffectVHS": OllamaNodesEffectVHS,
    "OllamaNodesEffectSpeedLines": OllamaNodesEffectSpeedLines,
    "OllamaNodesEffectLensFlare": OllamaNodesEffectLensFlare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesParticleRain": "✨ Particle Rain",
    "OllamaNodesParticleSnow": "✨ Particle Snow",
    "OllamaNodesParticleFire": "✨ Particle Fire",
    "OllamaNodesParticleSparks": "✨ Particle Sparks",
    "OllamaNodesParticleConfetti": "✨ Particle Confetti",
    "OllamaNodesParticleDust": "✨ Particle Dust/Bokeh",
    "OllamaNodesShapePulse": "✨ Shape Pulse",
    "OllamaNodesShapeWave": "✨ Shape Wave",
    "OllamaNodesShapeRings": "✨ Shape Rings",
    "OllamaNodesShapeGrid": "✨ Shape Grid",
    "OllamaNodesEffectGlitch": "✨ Effect Glitch",
    "OllamaNodesEffectChromatic": "✨ Effect Chromatic",
    "OllamaNodesEffectScanlines": "✨ Effect Scanlines",
    "OllamaNodesEffectNoise": "✨ Effect Film Noise",
    "OllamaNodesEffectVHS": "✨ Effect VHS Retro",
    "OllamaNodesEffectSpeedLines": "✨ Effect Speed Lines",
    "OllamaNodesEffectLensFlare": "✨ Effect Lens Flare",
}
