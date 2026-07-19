"""
OllamaNodes - Storyboard Creation Nodes
Generate storyboards, shot lists, and scene descriptions using LLM-powered prompts.
Works standalone or with Ollama integration for automated generation.
"""

import json


# ============================================================
# SHOT TYPES & CAMERA DATABASE
# ============================================================

SHOT_TYPES = {
    "WS": "Wide Shot - Establishes the scene, shows full environment",
    "LS": "Long Shot - Subject full body, environmental context",
    "MLS": "Medium Long Shot - Subject from knees up",
    "MS": "Medium Shot - Subject from waist up",
    "MCU": "Medium Close-Up - Subject from chest up",
    "CU": "Close-Up - Subject face fills frame",
    "ECU": "Extreme Close-Up - Detail (eyes, hands, object)",
    "OTS": "Over-The-Shoulder - Looking past someone's shoulder",
    "POV": "Point of View - Character's perspective",
    "BIRD": "Bird's Eye View - Directly above",
    "HIGH": "High Angle - Looking down on subject",
    "EYE": "Eye Level - Normal perspective",
    "LOW": "Low Angle - Looking up at subject",
    "DUTCH": "Dutch Angle - Tilted for disorientation",
    "WIPE": "Insert/Cutaway - Detail shot",
}

CAMERA_MOVEMENTS = [
    "Static", "Pan", "Tilt", "Dolly In", "Dolly Out",
    "Tracking", "Steadicam", "Handheld", "Crane", "Drone",
    "Zoom In", "Zoom Out", "Whip Pan", "Arc",
]

STORYBEAT_TEMPLATES = {
    "three_act": [
        {"act": "Act 1", "beats": ["Opening Image", "Setup", "Theme Stated", "Catalyst", "Debate"]},
        {"act": "Act 2A", "beats": ["Break into Two", "B Story", "Fun and Games", "Midpoint"]},
        {"act": "Act 2B", "beats": ["Bad Guys Close In", "All Is Lost", "Dark Night"]},
        {"act": "Act 3", "beats": ["Break into Three", "Finale", "Final Image"]},
    ],
    "heros_journey": [
        {"act": "Ordinary World", "beats": ["Hero in normal life", "Call to adventure"]},
        {"act": "Refusal", "beats": ["Hero hesitates", "Meeting mentor"]},
        {"act": "Journey Begins", "beats": ["Crossing threshold", "Tests and allies"]},
        {"act": "Ordeal", "beats": ["Approach cave", "Innermost cave", "Ordeal"]},
        {"act": "Return", "beats": ["Reward", "Road back", "Resurrection"]},
    ],
    "five_act": [
        {"act": "Exposition", "beats": ["Introduce characters", "Establish world"]},
        {"act": "Rising Action", "beats": ["Inciting incident", "Rising stakes"]},
        {"act": "Climax", "beats": ["Confrontation", "Turning point"]},
        {"act": "Falling Action", "beats": ["Resolution begins", "Aftermath"]},
        {"act": "Dénouement", "beats": ["Final resolution", "New normal"]},
    ],
}

MOOD_PRESETS = {
    "dramatic": {"lighting": "high contrast, dramatic shadows", "color": "deep blues, golds", "music": "orchestral tension"},
    "comedy": {"lighting": "bright, even", "color": "warm, saturated", "music": "upbeat, playful"},
    "horror": {"lighting": "low key, harsh shadows", "color": "desaturated, sickly greens", "music": "dissonant, sparse"},
    "romance": {"lighting": "soft, warm backlighting", "color": "golden hour tones", "music": "sweeping, emotional"},
    "action": {"lighting": "dynamic, motion blur", "color": "high contrast, orange/teal", "music": "intense, rhythmic"},
    "sci_fi": {"lighting": "neon, ambient glow", "color": "cool blues, cyan, magenta", "music": "electronic, ambient"},
    "fantasy": {"lighting": "ethereal, magical glow", "color": "rich jewel tones", "music": "orchestral, mystical"},
    "noir": {"lighting": "venetian blinds, deep shadows", "color": "black & white or muted", "music": "jazz, smoky"},
    "melancholy": {"lighting": "diffused, grey", "color": "muted, blue-grey", "music": "minimal piano"},
    "whimsical": {"lighting": "soft, magical", "color": "pastels, whimsical", "music": "light, magical chimes"},
}


class OllamaNodesStoryScene:
    """Define a single storyboard scene with all metadata."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_number": ("INT", {"default": 1, "min": 1, "max": 999}),
                "scene_title": ("STRING", {"default": "Scene Title", "placeholder": "Scene name..."}),
                "scene_description": ("STRING", {"default": "", "multiline": True, "placeholder": "What happens in this scene..."}),
                "location": ("STRING", {"default": "Interior - Office"}),
                "time_of_day": (["Day", "Night", "Dawn", "Dusk", "Golden Hour", "Timeless"],),
                "mood": (list(MOOD_PRESETS.keys()),),
            },
            "optional": {
                "characters": ("STRING", {"default": "", "multiline": True, "placeholder": "Character 1, Character 2"}),
                "dialogue_summary": ("STRING", {"default": "", "multiline": True}),
                "sound_effects": ("STRING", {"default": ""}),
                "vfx_notes": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scene_data_json", "scene_summary")
    FUNCTION = "create_scene"
    CATEGORY = "Ollama-Magic-Nodes/Storyboard"
    DESCRIPTION = """Define a storyboard scene with full metadata.
    
This node structures scene data that can be connected to:
- OllamaNodesShotListGenerator: To break scene into shots
- OllamaNodesImagePromptGenerator: To create image generation prompts
- OllamaNodesStoryboardExport: To export the final storyboard"""

    def create_scene(self, scene_number, scene_title, scene_description, location, 
                     time_of_day, mood, characters="", dialogue_summary="", 
                     sound_effects="", vfx_notes=""):
        scene_data = {
            "scene_number": scene_number,
            "scene_title": scene_title,
            "description": scene_description,
            "location": location,
            "time_of_day": time_of_day,
            "mood": mood,
            "mood_details": MOOD_PRESETS.get(mood, {}),
            "characters": [c.strip() for c in characters.split(",") if c.strip()],
            "dialogue": dialogue_summary,
            "sound_effects": sound_effects,
            "vfx": vfx_notes,
        }
        
        summary = f"""SCENE {scene_number}: {scene_title}
Location: {location} | Time: {time_of_day} | Mood: {mood}
Characters: {characters or "None specified"}
Description: {scene_description}
"""
        if dialogue_summary:
            summary += f"Dialogue: {dialogue_summary}\n"
        if vfx_notes:
            summary += f"VFX: {vfx_notes}\n"
        
        return (json.dumps(scene_data, indent=2), summary)


class OllamaNodesShotDefinition:
    """Define a single shot in the storyboard."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_id": ("STRING", {"default": "1A", "placeholder": "e.g., 1A, 1B, 2A"}),
                "shot_type": (list(SHOT_TYPES.keys()),),
                "camera_angle": (["Eye Level", "High Angle", "Low Angle", "Bird's Eye", "Dutch Angle", "Overhead"],),
                "camera_movement": (CAMERA_MOVEMENTS,),
                "description": ("STRING", {"default": "", "multiline": True, "placeholder": "What happens in this shot..."}),
            },
            "optional": {
                "duration_seconds": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "focal_length": ("STRING", {"default": "35mm"}),
                "character_action": ("STRING", {"default": ""}),
                "transition_in": (["Cut", "Dissolve", "Fade In", "Fade from Black", "Wipe", "None"],),
                "transition_out": (["Cut", "Dissolve", "Fade Out", "Fade to Black", "Wipe", "None"],),
                "dialogue": ("STRING", {"default": ""}),
                "notes": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("shot_data_json", "shot_summary")
    FUNCTION = "create_shot"
    CATEGORY = "Ollama-Magic-Nodes/Storyboard"
    DESCRIPTION = """Define a single shot with camera specifications.
    
Shot types define framing:
- WS/LS: Wide/Long shot for establishing
- MS/MCU: Medium shots for dialogue
- CU/ECU: Close-ups for emotion"""

    def create_shot(self, shot_id, shot_type, camera_angle, camera_movement, description,
                   duration_seconds=3.0, focal_length="35mm", character_action="",
                   transition_in="Cut", transition_out="Cut", dialogue="", notes=""):
        shot_data = {
            "shot_id": shot_id,
            "shot_type": shot_type,
            "shot_type_description": SHOT_TYPES.get(shot_type, ""),
            "camera_angle": camera_angle,
            "camera_movement": camera_movement,
            "description": description,
            "duration_seconds": duration_seconds,
            "focal_length": focal_length,
            "character_action": character_action,
            "transition_in": transition_in,
            "transition_out": transition_out,
            "dialogue": dialogue,
            "notes": notes,
        }
        
        summary = f"""SHOT {shot_id}: {SHOT_TYPES.get(shot_type, shot_type)}
Camera: {camera_angle}, {camera_movement}
Duration: {duration_seconds}s | Lens: {focal_length}
{f"Action: {character_action}" if character_action else ""}
{f"Dialogue: {dialogue}" if dialogue else ""}
Description: {description}
"""
        if notes:
            summary += f"Notes: {notes}\n"
        
        return (json.dumps(shot_data, indent=2), summary)


class OllamaNodesImagePromptGenerator:
    """Generate image generation prompts from scene/shot descriptions.
    
    Uses templates to create consistent prompts for AI image generation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "description": ("STRING", {"default": "", "multiline": True}),
                "style_preset": (["photorealistic", "cinematic", "anime", "digital_art", 
                                  "oil_painting", "concept_art", "storyboard_sketch", 
                                  "graphic_novel", "noir", "watercolor"],),
                "shot_type": (list(SHOT_TYPES.keys()),),
                "mood": (list(MOOD_PRESETS.keys()),),
            },
            "optional": {
                "subject": ("STRING", {"default": "", "placeholder": "Main subject/character..."}),
                "art_style_details": ("STRING", {"default": "", "placeholder": "Additional style notes..."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "quality_boost": ("BOOLEAN", {"default": True}),
                "aspect_ratio": (["16:9", "4:3", "1:1", "9:16", "21:9"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "prompt_info")
    FUNCTION = "generate_prompt"
    CATEGORY = "Ollama-Magic-Nodes/Storyboard"
    DESCRIPTION = """Generate image generation prompts from scene descriptions.
    
Creates consistent prompts optimized for:
- Stable Diffusion / SDXL / Flux
- DALL-E / Midjourney style
- ComfyUI workflows with IPAdapter"""

    def generate_prompt(self, description, style_preset, shot_type, mood,
                        subject="", art_style_details="", negative_prompt="",
                        quality_boost=True, aspect_ratio="16:9"):
        
        # Build positive prompt
        parts = []
        
        # Quality prefix
        if quality_boost:
            parts.append("masterpiece, best quality, highly detailed")
        
        # Shot type visual description
        shot_desc = SHOT_TYPES.get(shot_type, "")
        if shot_desc:
            # Extract visual hints from shot type
            if "Wide" in shot_type or "Long" in shot_type:
                parts.append("wide shot, full scene, environmental")
            elif "Close" in shot_type:
                parts.append("close-up shot, detailed face")
            elif "Medium" in shot_type:
                parts.append("medium shot, waist up")
        
        # Style presets
        style_map = {
            "photorealistic": "photorealistic, 8k uhd, dslr, film grain, sharp focus",
            "cinematic": "cinematic shot, dramatic lighting, depth of field, movie still, anamorphic lens",
            "anime": "anime style, detailed anime illustration, vibrant colors, cel shading",
            "digital_art": "digital art, trending on artstation, concept art, highly detailed",
            "oil_painting": "oil painting, masterful brushstrokes, canvas texture, fine art",
            "concept_art": "concept art, matte painting, environmental design, epic composition",
            "storyboard_sketch": "storyboard sketch, pencil drawing, line art, composition guide",
            "graphic_novel": "graphic novel style, bold outlines, dramatic shadows, ink work",
            "noir": "film noir style, high contrast, venetian blind shadows, moody",
            "watercolor": "watercolor painting, soft washes, paper texture, delicate",
        }
        parts.append(style_map.get(style_preset, style_preset))
        
        # Mood details
        mood_data = MOOD_PRESETS.get(mood, {})
        if mood_data:
            parts.append(f"{mood_data.get('lighting', '')} lighting")
        
        # Subject
        if subject:
            parts.append(subject)
        
        # Main description
        if description:
            parts.append(description)
        
        # Art style details
        if art_style_details:
            parts.append(art_style_details)
        
        positive_prompt = ", ".join(parts)
        
        # Build negative prompt
        default_negative = "worst quality, low quality, blurry, deformed, ugly, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, watermark, text, signature"
        if negative_prompt:
            final_negative = f"{negative_prompt}, {default_negative}"
        else:
            final_negative = default_negative
        
        info = f"""Prompt Generated:
Style: {style_preset}
Shot: {shot_type}
Mood: {mood}
Aspect: {aspect_ratio}
Quality boost: {'Yes' if quality_boost else 'No'}"""
        
        return (positive_prompt, final_negative, info)


class OllamaNodesShotListGenerator:
    """Generate a complete shot list from scene description using templates."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_description": ("STRING", {"default": "", "multiline": True}),
                "scene_type": (["dialogue", "action", "establishing", "montage", "chase", "emotional", "transition"],),
                "number_of_shots": ("INT", {"default": 5, "min": 1, "max": 20}),
                "duration_per_shot": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
            },
            "optional": {
                "characters": ("STRING", {"default": ""}),
                "include_transitions": ("BOOLEAN", {"default": True}),
                "camera_movement_preference": (["static_only", "mixed", "dynamic"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("shot_list_json", "shot_list_text", "estimated_duration")
    FUNCTION = "generate_shot_list"
    CATEGORY = "Ollama-Magic-Nodes/Storyboard"
    DESCRIPTION = """Generate a shot list from scene description.
    
Scene types determine shot patterns:
- Dialogue: Shot-reverse-shot, over-shoulder
- Action: Wide establishing, dynamic tracking
- Establishing: Wide to medium progression
- Emotional: Close-ups, tight framing"""

    def generate_shot_list(self, scene_description, scene_type, number_of_shots, 
                          duration_per_shot, characters="", include_transitions=True,
                          camera_movement_preference="mixed"):
        
        # Define shot patterns based on scene type
        patterns = {
            "dialogue": ["WS", "MS", "CU", "OTS", "MS", "CU", "OTS", "MS"],
            "action": ["WS", "LS", "MLS", "MS", "WS", "MLS", "CU", "WS"],
            "establishing": ["WS", "LS", "MLS", "MS", "CU", "WS"],
            "montage": ["CU", "MS", "WS", "ECU", "MLS", "CU", "WS", "MS"],
            "chase": ["WS", "LS", "MS", "WS", "MLS", "LS", "CU", "WS"],
            "emotional": ["CU", "ECU", "MS", "CU", "ECU", "MCU", "CU"],
            "transition": ["WS", "MLS", "MS", "LS", "WS"],
        }
        
        pattern = patterns.get(scene_type, patterns["dialogue"])
        
        # Generate shots
        shots = []
        total_duration = 0
        
        for i in range(number_of_shots):
            shot_type = pattern[i % len(pattern)]
            
            # Determine camera movement
            if camera_movement_preference == "static_only":
                movement = "Static"
            elif camera_movement_preference == "dynamic":
                movements = ["Pan", "Tilt", "Tracking", "Dolly In", "Dolly Out", "Steadicam"]
                movement = movements[i % len(movements)]
            else:
                if i % 3 == 0:
                    movement = "Static"
                else:
                    movement = ["Pan", "Tilt", "Dolly In"][i % 3]
            
            # Determine transitions
            trans_in = "Cut" if i > 0 and include_transitions else "Fade In" if i == 0 else "None"
            trans_out = "Cut" if i < number_of_shots - 1 and include_transitions else "Fade Out" if i == number_of_shots - 1 else "None"
            
            shot = {
                "shot_id": f"{chr(65 + i // 8)}{i + 1}",
                "shot_type": shot_type,
                "shot_type_description": SHOT_TYPES.get(shot_type, ""),
                "camera_movement": movement,
                "duration_seconds": duration_per_shot,
                "transition_in": trans_in,
                "transition_out": trans_out,
                "description": f"Shot {i + 1}: {scene_type} beat",
            }
            
            shots.append(shot)
            total_duration += duration_per_shot
        
        shot_list_data = {
            "scene_description": scene_description,
            "scene_type": scene_type,
            "shots": shots,
            "total_shots": len(shots),
            "total_duration": total_duration,
        }
        
        # Generate readable text
        text = f"""SHOT LIST - {scene_type.upper()} SCENE
Scene: {scene_description[:100]}...
Total Shots: {len(shots)} | Est. Duration: {total_duration:.1f}s

"""
        for shot in shots:
            text += f"""{shot['shot_id']}: {shot['shot_type']} - {SHOT_TYPES.get(shot['shot_type'], '')}
  Movement: {shot['camera_movement']} | Duration: {shot['duration_seconds']}s
  {f"In: {shot['transition_in']}, Out: {shot['transition_out']}" if include_transitions else ''}
  
"""
        
        return (json.dumps(shot_list_data, indent=2), text, int(total_duration))


class OllamaNodesStoryboardPage:
    """Combine scene and shots into a storyboard page layout."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_data": ("STRING", {"default": "{}", "multiline": True}),
                "panel_count": ("INT", {"default": 4, "min": 1, "max": 12}),
                "layout": (["2x2", "3x2", "4x3", "horizontal_strip", "vertical_strip", "custom"],),
            },
            "optional": {
                "panel_1": ("STRING", {"default": "", "multiline": True}),
                "panel_2": ("STRING", {"default": "", "multiline": True}),
                "panel_3": ("STRING", {"default": "", "multiline": True}),
                "panel_4": ("STRING", {"default": "", "multiline": True}),
                "panel_descriptions": ("STRING", {"default": "", "multiline": True, 
                    "placeholder": "One description per line..."}),
                "include_camera_notes": ("BOOLEAN", {"default": True}),
                "page_title": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("storyboard_json", "storyboard_text")
    FUNCTION = "create_page"
    CATEGORY = "Ollama-Magic-Nodes/Storyboard"
    DESCRIPTION = """Create a storyboard page combining scene info with panel descriptions.
    
Layout options:
- 2x2: 4 panels (standard)
- 3x2: 6 panels (detailed)
- 4x3: 12 panels (comprehensive)
- Strip: Horizontal or vertical scrolling"""

    def create_page(self, scene_data, panel_count, layout,
                   panel_1="", panel_2="", panel_3="", panel_4="",
                   panel_descriptions="", include_camera_notes=True, page_title=""):
        
        # Parse scene data
        try:
            scene = json.loads(scene_data)
        except:
            scene = {"scene_title": "Untitled Scene"}
        
        # Collect panel descriptions
        panels = []
        manual_panels = [panel_1, panel_2, panel_3, panel_4]
        for i, desc in enumerate(manual_panels):
            if desc:
                panels.append({"panel": i + 1, "description": desc, "shot_type": "MS"})
        
        # Add from panel_descriptions if provided
        if panel_descriptions:
            for i, line in enumerate(panel_descriptions.strip().split("\n")):
                if line.strip():
                    panels.append({
                        "panel": len(panels) + 1,
                        "description": line.strip(),
                        "shot_type": "MS"
                    })
        
        # Pad to panel_count
        while len(panels) < panel_count:
            panels.append({
                "panel": len(panels) + 1,
                "description": f"[Panel {len(panels) + 1} - Add description]",
                "shot_type": "MS"
            })
        
        panels = panels[:panel_count]
        
        storyboard_data = {
            "page_title": page_title or scene.get("scene_title", "Storyboard Page"),
            "scene": scene,
            "layout": layout,
            "panels": panels,
            "include_camera_notes": include_camera_notes,
        }
        
        # Generate readable text
        title = storyboard_data["page_title"]
        text = f"""{'='*50}
STORYBOARD: {title}
Scene: {scene.get('scene_title', 'N/A')}
Location: {scene.get('location', 'N/A')}
Mood: {scene.get('mood', 'N/A')}
Layout: {layout} ({panel_count} panels)
{'='*50}

"""
        for panel in panels:
            text += f"""[Panel {panel['panel']}] {panel['shot_type']}
{panel['description']}
{'─'*30}
"""
        
        return (json.dumps(storyboard_data, indent=2), text)


class OllamaNodesCharacterSheet:
    """Define character descriptions for consistent storyboard generation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_name": ("STRING", {"default": "Character", "placeholder": "Name..."}),
                "physical_description": ("STRING", {"default": "", "multiline": True, 
                    "placeholder": "Age, build, hair, eyes, distinguishing features..."}),
                "outfit": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "Clothing description..."}),
                "personality_traits": ("STRING", {"default": "", "placeholder": "brave, curious, etc."}),
            },
            "optional": {
                "age_range": ("STRING", {"default": "30s"}),
                "style_keywords": ("STRING", {"default": "", "placeholder": "cinematic, realistic, anime..."}),
                "reference_notes": ("STRING", {"default": "", "multiline": True}),
                "character_arc": ("STRING", {"default": "", "multiline": True, "placeholder": "Character development..."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("character_data_json", "character_prompt", "character_summary")
    FUNCTION = "create_character"
    CATEGORY = "Ollama-Magic-Nodes/Storyboard"
    DESCRIPTION = """Define a character for consistent representation across storyboard panels.
    
Provides:
- JSON data for storage
- Image generation prompt
- Human-readable summary"""

    def create_character(self, character_name, physical_description, outfit,
                        personality_traits, age_range="30s", style_keywords="",
                        reference_notes="", character_arc=""):
        
        character_data = {
            "name": character_name,
            "age_range": age_range,
            "physical_description": physical_description,
            "outfit": outfit,
            "personality_traits": personality_traits,
            "style_keywords": style_keywords,
            "reference_notes": reference_notes,
            "character_arc": character_arc,
        }
        
        # Build image prompt
        prompt_parts = [character_name]
        if age_range:
            prompt_parts.append(f"{age_range} years old")
        if physical_description:
            prompt_parts.append(physical_description)
        if outfit:
            prompt_parts.append(f"wearing {outfit}")
        if style_keywords:
            prompt_parts.append(style_keywords)
        
        character_prompt = ", ".join(prompt_parts)
        
        summary = f"""CHARACTER: {character_name}
Age: {age_range}
Description: {physical_description}
Outfit: {outfit}
Personality: {personality_traits}
{"Arc: " + character_arc if character_arc else ""}
"""
        
        return (json.dumps(character_data, indent=2), character_prompt, summary)


class OllamaNodesSceneToPrompt:
    """Convert a complete scene description into an image prompt.
    
    Combines scene data, character data, and shot info into a coherent prompt.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_description": ("STRING", {"default": "", "multiline": True}),
                "shot_type": (list(SHOT_TYPES.keys()),),
                "style": (["photorealistic", "cinematic", "anime", "concept_art", 
                          "graphic_novel", "storyboard_sketch"],),
            },
            "optional": {
                "characters": ("STRING", {"default": "", "multiline": True}),
                "setting": ("STRING", {"default": ""}),
                "time_of_day": (["Day", "Night", "Dawn", "Dusk", "Golden Hour"],),
                "mood": (list(MOOD_PRESETS.keys()),),
                "additional_details": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "prompt_breakdown")
    FUNCTION = "generate_scene_prompt"
    CATEGORY = "Ollama-Magic-Nodes/Storyboard"
    DESCRIPTION = """Convert scene description to optimized image generation prompt.
    
Combines all scene elements into a single coherent prompt
for consistent visual style across storyboard panels."""

    def generate_scene_prompt(self, scene_description, shot_type, style,
                             characters="", setting="", time_of_day="Day",
                             mood="cinematic", additional_details=""):
        
        parts = []
        
        # Style
        style_map = {
            "photorealistic": "photorealistic, 8k, detailed",
            "cinematic": "cinematic shot, movie still, dramatic lighting",
            "anime": "anime style, detailed illustration",
            "concept_art": "concept art, digital painting",
            "graphic_novel": "graphic novel illustration, bold lines",
            "storyboard_sketch": "storyboard sketch, pencil drawing",
        }
        parts.append(style_map.get(style, style))
        
        # Shot framing
        shot_visuals = {
            "WS": "wide shot, establishing shot",
            "MS": "medium shot",
            "CU": "close-up shot",
            "ECU": "extreme close-up",
            "LS": "long shot",
        }
        parts.append(shot_visuals.get(shot_type, "medium shot"))
        
        # Mood/lighting
        mood_data = MOOD_PRESETS.get(mood, {})
        if mood_data:
            parts.append(mood_data.get("lighting", ""))
        
        # Time of day lighting
        time_lighting = {
            "Day": "bright daylight",
            "Night": "nighttime, artificial lighting",
            "Dawn": "soft dawn light",
            "Dusk": "golden dusk lighting",
            "Golden Hour": "warm golden hour light",
        }
        parts.append(time_lighting.get(time_of_day, ""))
        
        # Characters
        if characters:
            parts.append(characters)
        
        # Setting
        if setting:
            parts.append(f"set in {setting}")
        
        # Main description
        if scene_description:
            parts.append(scene_description)
        
        # Additional
        if additional_details:
            parts.append(additional_details)
        
        prompt = ", ".join([p for p in parts if p])
        
        breakdown = f"""PROMPT BREAKDOWN:
Style: {style}
Framing: {shot_type}
Mood: {mood}
Time: {time_of_day}
Characters: {characters or 'None specified'}
Setting: {setting or 'Not specified'}
"""
        
        return (prompt, breakdown)


class OllamaNodesStoryboardExport:
    """Export storyboard to various formats."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard_data": ("STRING", {"default": "{}", "multiline": True}),
                "export_format": (["markdown", "json", "csv", "plain_text"],),
            },
            "optional": {
                "title": ("STRING", {"default": "My Storyboard"}),
                "author": ("STRING", {"default": ""}),
                "include_timestamps": ("BOOLEAN", {"default": True}),
                "filename": ("STRING", {"default": "storyboard"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("export_content", "metadata", "filename_suggestion")
    FUNCTION = "export_storyboard"
    CATEGORY = "Ollama-Magic-Nodes/Storyboard"
    OUTPUT_NODE = True
    DESCRIPTION = """Export storyboard to different formats.
    
Formats:
- Markdown: Human-readable with formatting
- JSON: Structured data for further processing
- CSV: Spreadsheet-compatible
- Plain text: Simple text output"""

    def export_storyboard(self, storyboard_data, export_format, 
                         title="My Storyboard", author="", 
                         include_timestamps=True, filename="storyboard"):
        try:
            data = json.loads(storyboard_data)
        except:
            data = {"title": title}
        
        scenes = data.get("scenes", [data])
        if not isinstance(scenes, list):
            scenes = [scenes]
        
        if export_format == "markdown":
            content = f"""# {title}

**Author:** {author or 'Not specified'}
**Scenes:** {len(scenes)}

---

"""
            for i, scene in enumerate(scenes):
                content += f"""## Scene {scene.get('scene_number', i + 1)}: {scene.get('scene_title', 'Untitled')}

**Location:** {scene.get('location', 'N/A')}
**Time:** {scene.get('time_of_day', 'N/A')}
**Mood:** {scene.get('mood', 'N/A')}

{scene.get('description', '')}

---

"""
            if include_timestamps:
                content += f"*Generated: {{}}*\n"
        
        elif export_format == "json":
            content = json.dumps(data, indent=2)
        
        elif export_format == "csv":
            content = "Scene,Shot,Type,Description,Duration,Notes\n"
            for i, scene in enumerate(scenes):
                for shot in scene.get("shots", []):
                    content += f"{i+1},{shot.get('shot_id','')},{shot.get('shot_type','')},\"{shot.get('description','')}\",{shot.get('duration_seconds',3)},{shot.get('notes','')}\n"
        
        else:  # plain text
            content = f"{title}\n{'='*len(title)}\n\n"
            for scene in scenes:
                content += f"Scene {scene.get('scene_number', '?')}: {scene.get('scene_title', 'Untitled')}\n"
                content += f"Location: {scene.get('location', 'N/A')}\n"
                content += f"{scene.get('description', '')}\n\n"
        
        metadata = json.dumps({
            "title": title,
            "author": author,
            "format": export_format,
            "scene_count": len(scenes),
        }, indent=2)
        
        safe_filename = "".join(c if c.isalnum() or c in "-_ " else "" for c in title).strip().replace(" ", "_")
        
        return (content, metadata, f"{safe_filename or filename}.{export_format[:3]}")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OllamaNodesStoryScene": OllamaNodesStoryScene,
    "OllamaNodesShotDefinition": OllamaNodesShotDefinition,
    "OllamaNodesImagePromptGenerator": OllamaNodesImagePromptGenerator,
    "OllamaNodesShotListGenerator": OllamaNodesShotListGenerator,
    "OllamaNodesStoryboardPage": OllamaNodesStoryboardPage,
    "OllamaNodesCharacterSheet": OllamaNodesCharacterSheet,
    "OllamaNodesSceneToPrompt": OllamaNodesSceneToPrompt,
    "OllamaNodesStoryboardExport": OllamaNodesStoryboardExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaNodesStoryScene": "🎬 Story Scene",
    "OllamaNodesShotDefinition": "🎬 Shot Definition",
    "OllamaNodesImagePromptGenerator": "🎬 Image Prompt Gen",
    "OllamaNodesShotListGenerator": "🎬 Shot List Gen",
    "OllamaNodesStoryboardPage": "🎬 Storyboard Page",
    "OllamaNodesCharacterSheet": "🎬 Character Sheet",
    "OllamaNodesSceneToPrompt": "🎬 Scene to Prompt",
    "OllamaNodesStoryboardExport": "🎬 Storyboard Export",
}
