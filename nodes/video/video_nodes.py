"""
OllamaNodes - Video Processing Nodes
Loading, editing, and manipulating video files with performance optimization.
Handles long videos via chunked loading, frame extraction, and memory management.
"""

import torch
import numpy as np
from PIL import Image
import os
import math
import json
import hashlib
import tempfile


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


class OllamaNodesVideoLoader:
    """
    Load video files with performance optimization for long videos.
    Supports chunked loading, frame skipping, and resolution scaling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "placeholder": "path/to/video.mp4"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "max_frames": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1}),
                "frame_skip": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "force_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("frames", "frame_count", "total_video_frames", "fps", "width", "height")
    FUNCTION = "load_video"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Load video with performance optimization.
    
Parameters:
- video_path: Path to video file (mp4, avi, mov, webm, mkv)
- start_frame: First frame to load
- max_frames: Maximum frames to load (memory safety)
- frame_skip: Skip N frames between each loaded frame (e.g., 1 = every other frame)
- scale_factor: Resize frames (0.5 = half size, faster loading)
- force_fps: Override video FPS (0 = use original)"""

    def load_video(self, video_path, start_frame, max_frames, frame_skip, scale_factor, force_fps):
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV (cv2) required for video loading. Install with: pip install opencv-python")
        
        if not os.path.exists(video_path):
            # Try looking in input directory
            import folder_paths
            video_path = os.path.join(folder_paths.get_input_directory(), video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_fps = force_fps if force_fps > 0 else original_fps
        
        if scale_factor != 1.0:
            width = int(width * scale_factor)
            height = int(height * scale_factor)
        
        # Calculate actual frame indices to load
        frame_indices = []
        current_frame = start_frame
        skip_counter = 0
        
        while len(frame_indices) < max_frames and current_frame < total_frames:
            if skip_counter == 0:
                frame_indices.append(current_frame)
                skip_counter = frame_skip
            else:
                skip_counter -= 1
            current_frame += 1
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to float32 [0,1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame))
        
        cap.release()
        
        if not frames:
            # Return empty frame
            frames = [torch.zeros(64, 64, 3)]
        
        frames_tensor = torch.stack(frames)
        
        return (frames_tensor, len(frames), total_frames, output_fps, width, height)


class OllamaNodesVideoSaver:
    """Save image batch as video file with configurable codec and quality."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "output_video"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "codec": (["mp4v", "avc1", "XVID", "MJPG"],),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    OUTPUT_NODE = True
    DESCRIPTION = """Save image batch as video file.
    
Codecs:
- mp4v: Standard MP4 (most compatible)
- avc1: H.264 (better compression)
- XVID: AVI format
- MJPG: Motion JPEG (large files, fast encoding)"""

    def save_video(self, images, filename, fps, codec, quality):
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV (cv2) required. Install with: pip install opencv-python")
        
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        
        ext = ".mp4" if codec in ["mp4v", "avc1"] else ".avi"
        video_path = os.path.join(output_dir, f"{filename}{ext}")
        
        # Handle duplicate filenames
        counter = 1
        base_path = video_path
        while os.path.exists(video_path):
            name, extension = os.path.splitext(base_path)
            video_path = f"{name}_{counter}{extension}"
            counter += 1
        
        frames = tensor_to_pil(images)
        h, w = frames[0].height, frames[0].width
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        for frame in frames:
            frame_np = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        return (video_path,)


class OllamaNodesVideoInfo:
    """Get detailed video file information without loading all frames."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("frame_count", "width", "height", "fps", "duration_seconds", "info_text")
    FUNCTION = "get_info"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = "Get video metadata without loading frames into memory."

    def get_info(self, video_path):
        try:
            import cv2
        except ImportError:
            return (0, 0, 0, 0.0, 0.0, "OpenCV not installed")
        
        if not os.path.exists(video_path):
            import folder_paths
            video_path = os.path.join(folder_paths.get_input_directory(), video_path)
        
        if not os.path.exists(video_path):
            return (0, 0, 0, 0.0, 0.0, f"File not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (0, 0, 0, 0.0, 0.0, f"Cannot open: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        info = f"{width}x{height} | {frame_count} frames | {fps:.2f} fps | {duration:.2f}s"
        
        return (frame_count, width, height, fps, duration, info)


class OllamaNodesVideoFrameExtract:
    """Extract specific frames from a video by index or time."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "extraction_mode": (["by_index", "by_time", "uniform_sample", "first_last"],),
                "frame_indices": ("STRING", {"default": "0, 10, 20, 30", "placeholder": "0, 10, 20 or 0-30"}),
                "time_stamps": ("STRING", {"default": "0.0, 1.0, 2.0", "placeholder": "seconds"}),
                "sample_count": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "count", "extracted_indices")
    FUNCTION = "extract_frames"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Extract specific frames from video.
    
Modes:
- by_index: Specify frame numbers (e.g., "0, 10, 20" or "0-30")
- by_time: Specify timestamps in seconds
- uniform_sample: Extract N evenly-spaced frames
- first_last: Get only first and last frames"""

    def extract_frames(self, video_path, extraction_mode, frame_indices, time_stamps, sample_count, scale_factor):
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV required")
        
        if not os.path.exists(video_path):
            import folder_paths
            video_path = os.path.join(folder_paths.get_input_directory(), video_path)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        indices = []
        
        if extraction_mode == "by_index":
            # Parse indices like "0, 10, 20" or "0-30"
            for part in frame_indices.split(','):
                part = part.strip()
                if '-' in part and not part.startswith('-'):
                    start, end = part.split('-')
                    indices.extend(range(int(start), int(end) + 1))
                else:
                    try:
                        indices.append(int(part))
                    except ValueError:
                        pass
        elif extraction_mode == "by_time":
            for t in time_stamps.split(','):
                try:
                    t_float = float(t.strip())
                    idx = int(t_float * fps)
                    indices.append(idx)
                except ValueError:
                    pass
        elif extraction_mode == "uniform_sample":
            if total_frames > 0:
                step = max(1, total_frames // sample_count)
                indices = list(range(0, total_frames, step))[:sample_count]
        elif extraction_mode == "first_last":
            indices = [0, max(0, total_frames - 1)]
        
        # Clamp and deduplicate
        indices = sorted(set(max(0, min(i, total_frames - 1)) for i in indices))
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if scale_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
        
        cap.release()
        
        if not frames:
            frames = [torch.zeros(64, 64, 3)]
        
        return (torch.stack(frames), len(frames), ", ".join(map(str, indices)))


class OllamaNodesVideoBatchSplit:
    """Split video/image batch into multiple segments."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "split_mode": (["equal_parts", "by_count", "by_indices"],),
                "num_parts": ("INT", {"default": 2, "min": 2, "max": 100}),
                "frames_per_part": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "split_indices": ("STRING", {"default": "0, 30, 60", "placeholder": "frame indices for splits"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("part_1", "part_2", "part_3", "part_4", "num_parts")
    FUNCTION = "split_batch"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Split frame batch into parts.
    
Modes:
- equal_parts: Divide evenly into N parts
- by_count: Each part has fixed frame count
- by_indices: Split at specified frame indices"""

    def split_batch(self, images, split_mode, num_parts, frames_per_part, split_indices):
        total = images.shape[0]
        
        if split_mode == "equal_parts":
            part_size = total // num_parts
            indices = [i * part_size for i in range(num_parts)] + [total]
        elif split_mode == "by_count":
            indices = list(range(0, total, frames_per_part)) + [total]
        elif split_mode == "by_indices":
            indices = [0]
            for idx in split_indices.split(','):
                try:
                    i = int(idx.strip())
                    if 0 < i < total:
                        indices.append(i)
                except ValueError:
                    pass
            indices.append(total)
            indices = sorted(set(indices))
        else:
            indices = [0, total]
        
        parts = []
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            parts.append(images[start:end])
        
        # Pad to 4 outputs
        while len(parts) < 4:
            parts.append(torch.zeros(1, 64, 64, 3))
        
        return (parts[0], parts[1], parts[2], parts[3], len([p for p in parts if p.shape[0] > 1 or not torch.all(p == 0)]))


class OllamaNodesVideoBatchJoin:
    """Join multiple image batches into a single video sequence."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "batch_1": ("IMAGE",),
                "batch_2": ("IMAGE",),
                "batch_3": ("IMAGE",),
                "batch_4": ("IMAGE",),
                "transition_frames": ("INT", {"default": 0, "min": 0, "max": 60}),
                "transition_type": (["none", "crossfade", "fade_black"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("combined", "total_frames")
    FUNCTION = "join_batches"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Join multiple frame batches into one sequence.
    
Transitions:
- none: Direct concatenation
- crossfade: Blend between segments
- fade_black: Fade out to black, then in"""

    def join_batches(self, batch_1=None, batch_2=None, batch_3=None, batch_4=None, 
                     transition_frames=0, transition_type="none"):
        batches = [b for b in [batch_1, batch_2, batch_3, batch_4] if b is not None]
        
        if not batches:
            return (torch.zeros(1, 64, 64, 3), 0)
        
        if len(batches) == 1:
            return (batches[0], batches[0].shape[0])
        
        # Ensure all batches have same spatial dimensions
        target_h, target_w = batches[0].shape[1], batches[0].shape[2]
        
        result_frames = []
        
        for i, batch in enumerate(batches):
            # Resize if needed
            if batch.shape[1] != target_h or batch.shape[2] != target_w:
                pil_frames = tensor_to_pil(batch)
                pil_frames = [f.resize((target_w, target_h), Image.LANCZOS) for f in pil_frames]
                batch = pil_to_tensor(pil_frames)
            
            if i > 0 and transition_frames > 0 and transition_type != "none":
                prev_batch = batches[i - 1]
                
                if transition_type == "crossfade":
                    # Create crossfade
                    for t in range(transition_frames):
                        alpha = t / transition_frames
                        if len(result_frames) > 0 and batch.shape[0] > 0:
                            blended = result_frames[-1] * (1 - alpha) + batch[0] * alpha
                            result_frames.append(blended)
                
                elif transition_type == "fade_black":
                    # Fade out
                    for t in range(transition_frames // 2):
                        alpha = 1 - t / (transition_frames // 2)
                        if len(result_frames) > 0:
                            faded = result_frames[-1] * alpha
                            result_frames.append(faded)
                    # Fade in
                    for t in range(transition_frames - transition_frames // 2):
                        alpha = t / (transition_frames - transition_frames // 2)
                        if batch.shape[0] > 0:
                            faded = batch[0] * alpha
                            result_frames.append(faded)
            
            for j in range(batch.shape[0]):
                result_frames.append(batch[j])
        
        result = torch.stack(result_frames)
        return (result, result.shape[0])


class OllamaNodesVideoReverse:
    """Reverse video playback order."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reversed",)
    FUNCTION = "reverse"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = "Reverse the order of frames in a batch."

    def reverse(self, images):
        return (torch.flip(images, [0]),)


class OllamaNodesVideoLoop:
    """Loop video by repeating frames."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "loop_count": ("INT", {"default": 2, "min": 1, "max": 100}),
                "mode": (["repeat", "ping_pong", "reverse_loop"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("looped", "total_frames")
    FUNCTION = "loop_video"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Loop/repeat video frames.
    
Modes:
- repeat: A, A, A... (simple repeat)
- ping_pong: A, reverse(A), A, reverse(A)...
- reverse_loop: A, reverse(A)..."""

    def loop_video(self, images, loop_count, mode):
        frames = []
        
        for i in range(loop_count):
            if mode == "repeat":
                frames.append(images)
            elif mode == "ping_pong":
                if i % 2 == 0:
                    frames.append(images)
                else:
                    frames.append(torch.flip(images, [0]))
            elif mode == "reverse_loop":
                frames.append(images)
                frames.append(torch.flip(images, [0]))
        
        result = torch.cat(frames, dim=0)
        return (result, result.shape[0])


class OllamaNodesVideoSpeed:
    """Change video playback speed by frame interpolation/skipping."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "interpolation": (["nearest", "linear"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "change_speed"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Change video speed.
    
- speed_factor > 1: Faster (fewer frames)
- speed_factor < 1: Slower (more frames via interpolation)

Interpolation (for slowdown):
- nearest: Duplicate frames
- linear: Blend between frames"""

    def change_speed(self, images, speed_factor, interpolation):
        n = images.shape[0]
        new_n = max(1, int(n / speed_factor))
        
        if speed_factor >= 1.0:
            # Speed up: sample frames
            indices = np.linspace(0, n - 1, new_n).astype(int)
            result = images[indices]
        else:
            # Slow down: interpolate
            frames = []
            for i in range(new_n):
                t = i * speed_factor
                idx = int(t)
                frac = t - idx
                
                if interpolation == "nearest" or idx >= n - 1:
                    frames.append(images[min(idx, n - 1)])
                else:
                    # Linear blend
                    blended = images[idx] * (1 - frac) + images[idx + 1] * frac
                    frames.append(blended)
            result = torch.stack(frames)
        
        return (result, result.shape[0])


class OllamaNodesVideoTrim:
    """Trim video to specific start/end frames or times."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_mode": (["by_frames", "by_percentage"],),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 100000}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_percent": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("trimmed", "frame_count")
    FUNCTION = "trim"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Trim video to specific range.
    
- by_frames: Use frame indices (end_frame=-1 means last frame)
- by_percentage: Use percentage of total length"""

    def trim(self, images, trim_mode, start_frame, end_frame, start_percent, end_percent):
        n = images.shape[0]
        
        if trim_mode == "by_frames":
            start = max(0, min(start_frame, n - 1))
            end = n if end_frame < 0 else min(end_frame + 1, n)
        else:
            start = int(n * start_percent / 100)
            end = int(n * end_percent / 100)
        
        start = max(0, start)
        end = max(start + 1, min(end, n))
        
        result = images[start:end]
        return (result, result.shape[0])


class OllamaNodesVideoFade:
    """Add fade in/out effects to video."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fade_in_frames": ("INT", {"default": 0, "min": 0, "max": 120}),
                "fade_out_frames": ("INT", {"default": 0, "min": 0, "max": 120}),
                "fade_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("faded",)
    FUNCTION = "apply_fade"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = "Apply fade in/out effects. Fades to/from specified color (default black)."

    def apply_fade(self, images, fade_in_frames, fade_out_frames, fade_color):
        n = images.shape[0]
        result = images.clone()
        
        # Parse color
        color = fade_color.lstrip('#')
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0
        fade_tensor = torch.tensor([r, g, b]).view(1, 1, 1, 3).to(images.device)
        
        # Fade in
        for i in range(min(fade_in_frames, n)):
            alpha = i / fade_in_frames
            result[i] = result[i] * alpha + fade_tensor * (1 - alpha)
        
        # Fade out
        for i in range(min(fade_out_frames, n)):
            frame_idx = n - 1 - i
            alpha = i / fade_out_frames
            result[frame_idx] = result[frame_idx] * (1 - alpha) + fade_tensor * alpha
        
        return (result,)


class OllamaNodesVideoOverlay:
    """Overlay one video/image on another with position and blend control."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "overlay": ("IMAGE",),
                "x_position": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "y_position": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 2.0, "step": 0.01}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited",)
    FUNCTION = "overlay"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Overlay video/image on background.
    
- x_position/y_position: -1 to 1 (0 = center)
- scale: Size of overlay relative to background
- opacity: Transparency of overlay
- mask: Optional mask for overlay (white = visible)"""

    def overlay(self, background, overlay, x_position, y_position, scale, opacity, mask=None):
        bg_h, bg_w = background.shape[1], background.shape[2]
        
        # Resize overlay
        pil_overlays = tensor_to_pil(overlay)
        new_w = int(bg_w * scale)
        new_h = int(bg_h * scale)
        pil_overlays = [o.resize((new_w, new_h), Image.LANCZOS) for o in pil_overlays]
        
        # Calculate position
        x = int((x_position + 1) / 2 * bg_w - new_w / 2)
        y = int((y_position + 1) / 2 * bg_h - new_h / 2)
        
        results = []
        for i in range(background.shape[0]):
            bg_frame = background[i].clone()
            ov_idx = min(i, len(pil_overlays) - 1)
            ov_frame = pil_overlays[ov_idx]
            ov_tensor = pil_to_tensor([ov_frame])[0]
            
            # Calculate valid overlay region
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(bg_w, x + new_w)
            y2 = min(bg_h, y + new_h)
            
            ox1 = max(0, -x)
            oy1 = max(0, -y)
            ox2 = ox1 + (x2 - x1)
            oy2 = oy1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                ov_crop = ov_tensor[oy1:oy2, ox1:ox2, :]
                
                if mask is not None:
                    m_idx = min(i, mask.shape[0] - 1)
                    m = mask[m_idx]
                    # Resize mask
                    m_pil = Image.fromarray((m.cpu().numpy() * 255).astype(np.uint8), 'L')
                    m_pil = m_pil.resize((new_w, new_h), Image.LANCZOS)
                    m_np = np.array(m_pil).astype(np.float32) / 255.0
                    m_tensor = torch.from_numpy(m_np[oy1:oy2, ox1:ox2]).unsqueeze(-1)
                    alpha = m_tensor * opacity
                else:
                    alpha = opacity
                
                bg_frame[y1:y2, x1:x2, :] = bg_frame[y1:y2, x1:x2, :] * (1 - alpha) + ov_crop * alpha
            
            results.append(bg_frame)
        
        return (torch.stack(results),)


class OllamaNodesVideoTransition:
    """Create transitions between two video clips."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("IMAGE",),
                "video_b": ("IMAGE",),
                "transition_frames": ("INT", {"default": 30, "min": 1, "max": 120}),
                "transition_type": (["crossfade", "wipe_left", "wipe_right", "wipe_up", "wipe_down", 
                                    "fade_black", "fade_white", "dissolve", "zoom_in", "zoom_out"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("transition", "frame_count")
    FUNCTION = "create_transition"
    CATEGORY = "Ollama-Magic-Nodes/Video"
    DESCRIPTION = """Create transitions between clips.
    
Types: crossfade, wipe (4 directions), fade to color, dissolve, zoom"""

    def create_transition(self, video_a, video_b, transition_frames, transition_type):
        # Take last frame of A and first frame of B
        frame_a = video_a[-1:]
        frame_b = video_b[0:1]
        
        # Ensure same size
        if frame_a.shape[1:3] != frame_b.shape[1:3]:
            h, w = frame_a.shape[1], frame_a.shape[2]
            pil_b = tensor_to_pil(frame_b)
            pil_b = [f.resize((w, h), Image.LANCZOS) for f in pil_b]
            frame_b = pil_to_tensor(pil_b)
        
        h, w = frame_a.shape[1], frame_a.shape[2]
        frames = []
        
        for t in range(transition_frames):
            progress = t / (transition_frames - 1) if transition_frames > 1 else 1
            
            if transition_type == "crossfade":
                blended = frame_a[0] * (1 - progress) + frame_b[0] * progress
            
            elif transition_type == "wipe_left":
                split = int(w * progress)
                blended = frame_a[0].clone()
                blended[:, :split, :] = frame_b[0, :, :split, :]
            
            elif transition_type == "wipe_right":
                split = int(w * (1 - progress))
                blended = frame_a[0].clone()
                blended[:, split:, :] = frame_b[0, :, split:, :]
            
            elif transition_type == "wipe_up":
                split = int(h * progress)
                blended = frame_a[0].clone()
                blended[:split, :, :] = frame_b[0, :split, :, :]
            
            elif transition_type == "wipe_down":
                split = int(h * (1 - progress))
                blended = frame_a[0].clone()
                blended[split:, :, :] = frame_b[0, split:, :, :]
            
            elif transition_type == "fade_black":
                if progress < 0.5:
                    alpha = 1 - progress * 2
                    blended = frame_a[0] * alpha
                else:
                    alpha = (progress - 0.5) * 2
                    blended = frame_b[0] * alpha
            
            elif transition_type == "fade_white":
                if progress < 0.5:
                    alpha = progress * 2
                    blended = frame_a[0] * (1 - alpha) + torch.ones_like(frame_a[0]) * alpha
                else:
                    alpha = (progress - 0.5) * 2
                    blended = torch.ones_like(frame_b[0]) * (1 - alpha) + frame_b[0] * alpha
            
            elif transition_type == "dissolve":
                noise = torch.rand_like(frame_a[0, :, :, 0:1])
                mask = (noise < progress).float()
                blended = frame_a[0] * (1 - mask) + frame_b[0] * mask
            
            elif transition_type in ["zoom_in", "zoom_out"]:
                if transition_type == "zoom_in":
                    scale = 1 + progress * 0.5
                else:
                    scale = 1.5 - progress * 0.5
                
                # Scale frame A
                pil_a = tensor_to_pil(frame_a)[0]
                new_size = (int(w * scale), int(h * scale))
                pil_a = pil_a.resize(new_size, Image.LANCZOS)
                
                # Crop center
                left = (new_size[0] - w) // 2
                top = (new_size[1] - h) // 2
                pil_a = pil_a.crop((left, top, left + w, top + h))
                
                scaled_a = pil_to_tensor([pil_a])[0]
                blended = scaled_a * (1 - progress) + frame_b[0] * progress
            
            else:
                blended = frame_a[0] * (1 - progress) + frame_b[0] * progress
            
            frames.append(blended)
        
        result = torch.stack(frames)
        return (result, result.shape[0])


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesVideoLoader": OllamaNodesVideoLoader,
    "OllamaNodesVideoSaver": OllamaNodesVideoSaver,
    "OllamaNodesVideoInfo": OllamaNodesVideoInfo,
    "OllamaNodesVideoFrameExtract": OllamaNodesVideoFrameExtract,
    "OllamaNodesVideoBatchSplit": OllamaNodesVideoBatchSplit,
    "OllamaNodesVideoBatchJoin": OllamaNodesVideoBatchJoin,
    "OllamaNodesVideoReverse": OllamaNodesVideoReverse,
    "OllamaNodesVideoLoop": OllamaNodesVideoLoop,
    "OllamaNodesVideoSpeed": OllamaNodesVideoSpeed,
    "OllamaNodesVideoTrim": OllamaNodesVideoTrim,
    "OllamaNodesVideoFade": OllamaNodesVideoFade,
    "OllamaNodesVideoOverlay": OllamaNodesVideoOverlay,
    "OllamaNodesVideoTransition": OllamaNodesVideoTransition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesVideoLoader": "🎬 Video Loader",
    "OllamaNodesVideoSaver": "🎬 Video Saver",
    "OllamaNodesVideoInfo": "🎬 Video Info",
    "OllamaNodesVideoFrameExtract": "🎬 Frame Extract",
    "OllamaNodesVideoBatchSplit": "🎬 Batch Split",
    "OllamaNodesVideoBatchJoin": "🎬 Batch Join",
    "OllamaNodesVideoReverse": "🎬 Video Reverse",
    "OllamaNodesVideoLoop": "🎬 Video Loop",
    "OllamaNodesVideoSpeed": "🎬 Video Speed",
    "OllamaNodesVideoTrim": "🎬 Video Trim",
    "OllamaNodesVideoFade": "🎬 Video Fade",
    "OllamaNodesVideoOverlay": "🎬 Video Overlay",
    "OllamaNodesVideoTransition": "🎬 Video Transition",
}
