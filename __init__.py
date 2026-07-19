"""
ComfyUI-OllamaNodes - Advanced Ollama node pack.

Place this folder inside ComfyUI/custom_nodes/ and restart ComfyUI.
Requires Ollama running locally, defaulting to http://localhost:11434.
"""

from .nodes.core.ollama_model_loader import OllamaModelLoader
from .nodes.core.ollama_text_generate import OllamaTextGenerate
from .nodes.core.ollama_chat import OllamaChat
from .nodes.core.ollama_vision import OllamaVision
from .nodes.core.ollama_embeddings import OllamaEmbeddings
from .nodes.core.ollama_prompt_builder import OllamaPromptBuilder, OllamaSinglePromptBuilder
from .nodes.core.ollama_json_extractor import OllamaJSONExtractor
from .nodes.core.ollama_model_manager import OllamaModelManager
from .nodes.prompt.prompt_enhancer import OllamaPromptEnhancer
from .nodes.scene.scene_director import OllamaSceneDirector
from .nodes.image.style_transfer import OllamaStyleTransfer
from .nodes.image.image_analyzer import OllamaImageAnalyzer
from .nodes.image.image_edit import OllamaImageEdit
from .nodes.image.image_sequence_analyzer import OllamaImageSequenceAnalyzer
from .nodes.image.photoset_folder_analyzer import OllamaPhotosetFolderAnalyzer
from .nodes.image.analyzer_prompt_saver import OllamaAnalyzerPromptSaver
from .nodes.image.web_image_tool import OllamaWebImageTool
from .nodes.image.image_merger import OllamaImageMerger
from .nodes.character.character_sheet import OllamaCharacterSheet

from .nodes.prompt.negative_prompt import OllamaNegativePrompt
from .nodes.prompt.prompt_variations import OllamaPromptVariations
from .nodes.image.inpaint_prompt import OllamaInpaintPrompt
from .nodes.character.pose_descriptor import OllamaPoseDescriptor
from .nodes.character.outfit_generator import OllamaOutfitGenerator
from .nodes.scene.environment_transform import OllamaEnvironmentTransform
from .nodes.image.image_comparator import OllamaImageComparator
from .nodes.scene.storyboard import OllamaStoryboardGenerator
from .nodes.scene.regional_prompts import OllamaRegionalPrompts
from .nodes.prompt.prompt_critic import OllamaPromptCritic
from .nodes.image.auto_tagger import OllamaAutoTagger
from .nodes.image.color_palette import OllamaColorPalette
from .nodes.image.style_identifier import OllamaStyleIdentifier
from .nodes.scene.aspect_optimizer import OllamaAspectOptimizer
from .nodes.scene.advanced_scene_director import OllamaAdvancedSceneDirector
from .nodes.character.subject_builder import OllamaSubjectBuilder
from .nodes.design.lighting_designer import OllamaLightingDesigner
from .nodes.prompt.prompt_combiner import OllamaPromptCombiner
from .nodes.prompt.prompt_translator import OllamaPromptTranslator
from .nodes.prompt.wildcard_generator import OllamaWildcardGenerator
from .nodes.video.video_prompt_generator import OllamaLTXVVideoPrompt, OllamaWanVideoPrompt
from .nodes.video.ltx_ingredients_prompt_node import LTXIngredientsPrompt
from .nodes.character.character_consistency_tools import (
    OllamaCharacterAnchorExtractor,
    OllamaExpressionSheetGenerator,
    OllamaOutfitSheetGenerator,
    OllamaPoseSheetGenerator,
    OllamaPromptContinuityChecker,
)
from .nodes.video.video_shot_director import OllamaVideoShotDirector
from .nodes.scene.action_choreographer import OllamaActionChoreographer
from .nodes.scene.background_generator import OllamaBackgroundGenerator
from .nodes.utility.controlnet_helper import OllamaControlNetHelper
from .nodes.character.creature_creator import OllamaCreatureCreator
from .nodes.prompt.detail_injector import OllamaDetailInjector
from .nodes.scene.emotion_director import OllamaEmotionDirector
from .nodes.character.hand_pose_helper import OllamaHandPoseHelper
from .nodes.image.image_to_story import OllamaImageToStory
from .nodes.utility.lora_suggester import OllamaLoraSuggester
from .nodes.design.texture_material import OllamaTextureMaterial
from .nodes.image.image_nodes import NODE_CLASS_MAPPINGS as IMAGE_NODES, NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DISPLAY
from .nodes.mask.mask_nodes import NODE_CLASS_MAPPINGS as MASK_NODES, NODE_DISPLAY_NAME_MAPPINGS as MASK_DISPLAY
from .nodes.text.text_nodes import NODE_CLASS_MAPPINGS as TEXT_NODES, NODE_DISPLAY_NAME_MAPPINGS as TEXT_DISPLAY
from .nodes.math.math_nodes import NODE_CLASS_MAPPINGS as MATH_NODES, NODE_DISPLAY_NAME_MAPPINGS as MATH_DISPLAY
from .nodes.flow.flow_nodes import NODE_CLASS_MAPPINGS as FLOW_NODES, NODE_DISPLAY_NAME_MAPPINGS as FLOW_DISPLAY
from .nodes.utility.utility_nodes import NODE_CLASS_MAPPINGS as UTIL_NODES, NODE_DISPLAY_NAME_MAPPINGS as UTIL_DISPLAY
from .nodes.animation.animation_nodes import NODE_CLASS_MAPPINGS as ANIM_NODES, NODE_DISPLAY_NAME_MAPPINGS as ANIM_DISPLAY
from .nodes.color.color_nodes import NODE_CLASS_MAPPINGS as COLOR_NODES, NODE_DISPLAY_NAME_MAPPINGS as COLOR_DISPLAY
from .nodes.video.video_nodes import NODE_CLASS_MAPPINGS as VIDEO_NODES, NODE_DISPLAY_NAME_MAPPINGS as VIDEO_DISPLAY
from .nodes.storyboard.storyboard_nodes import NODE_CLASS_MAPPINGS as STORY_NODES, NODE_DISPLAY_NAME_MAPPINGS as STORY_DISPLAY
from .nodes.composition.composition_nodes import NODE_CLASS_MAPPINGS as COMP_NODES, NODE_DISPLAY_NAME_MAPPINGS as COMP_DISPLAY
from .nodes.vfx.vfx_nodes import NODE_CLASS_MAPPINGS as VFX_NODES, NODE_DISPLAY_NAME_MAPPINGS as VFX_DISPLAY

NODE_CLASS_MAPPINGS = {
    # Core
    "OllamaModelLoader": OllamaModelLoader,
    "OllamaTextGenerate": OllamaTextGenerate,
    "OllamaChat": OllamaChat,
    "OllamaVision": OllamaVision,
    "OllamaEmbeddings": OllamaEmbeddings,
    "OllamaPromptBuilder": OllamaPromptBuilder,
    "OllamaSinglePromptBuilder": OllamaSinglePromptBuilder,
    "OllamaJSONExtractor": OllamaJSONExtractor,
    "OllamaModelManager": OllamaModelManager,
    # Existing creative nodes
    "OllamaPromptEnhancer": OllamaPromptEnhancer,
    "OllamaSceneDirector": OllamaSceneDirector,
    "OllamaStyleTransfer": OllamaStyleTransfer,
    "OllamaImageAnalyzer": OllamaImageAnalyzer,
    "OllamaImageEdit": OllamaImageEdit,
    "OllamaImageSequenceAnalyzer": OllamaImageSequenceAnalyzer,
    "OllamaPhotosetFolderAnalyzer": OllamaPhotosetFolderAnalyzer,
    "OllamaAnalyzerPromptSaver": OllamaAnalyzerPromptSaver,
    "OllamaWebImageTool": OllamaWebImageTool,
    "OllamaImageMerger": OllamaImageMerger,
    "OllamaCharacterSheet": OllamaCharacterSheet,
    # Newly merged extended nodes
    "OllamaNegativePrompt": OllamaNegativePrompt,
    "OllamaPromptVariations": OllamaPromptVariations,
    "OllamaInpaintPrompt": OllamaInpaintPrompt,
    "OllamaPoseDescriptor": OllamaPoseDescriptor,
    "OllamaOutfitGenerator": OllamaOutfitGenerator,
    "OllamaEnvironmentTransform": OllamaEnvironmentTransform,
    "OllamaImageComparator": OllamaImageComparator,
    "OllamaStoryboardGenerator": OllamaStoryboardGenerator,
    "OllamaRegionalPrompts": OllamaRegionalPrompts,
    "OllamaPromptCritic": OllamaPromptCritic,
    "OllamaAutoTagger": OllamaAutoTagger,
    "OllamaColorPalette": OllamaColorPalette,
    "OllamaStyleIdentifier": OllamaStyleIdentifier,
    "OllamaAspectOptimizer": OllamaAspectOptimizer,
    # Newly merged builder nodes
    "OllamaAdvancedSceneDirector": OllamaAdvancedSceneDirector,
    "OllamaSubjectBuilder": OllamaSubjectBuilder,
    "OllamaLightingDesigner": OllamaLightingDesigner,
    "OllamaPromptCombiner": OllamaPromptCombiner,
    "OllamaPromptTranslator": OllamaPromptTranslator,
    "OllamaWildcardGenerator": OllamaWildcardGenerator,
    # Video prompt nodes
    "OllamaLTXVVideoPrompt": OllamaLTXVVideoPrompt,
    "OllamaWanVideoPrompt": OllamaWanVideoPrompt,
    "LTXIngredientsPrompt": LTXIngredientsPrompt,
    "OllamaVideoShotDirector": OllamaVideoShotDirector,
    # Character continuity nodes
    "OllamaCharacterAnchorExtractor": OllamaCharacterAnchorExtractor,
    "OllamaOutfitSheetGenerator": OllamaOutfitSheetGenerator,
    "OllamaPoseSheetGenerator": OllamaPoseSheetGenerator,
    "OllamaExpressionSheetGenerator": OllamaExpressionSheetGenerator,
    "OllamaPromptContinuityChecker": OllamaPromptContinuityChecker,
    # Newly integrated creator/designer/helper nodes
    "OllamaActionChoreographer": OllamaActionChoreographer,
    "OllamaBackgroundGenerator": OllamaBackgroundGenerator,
    "OllamaControlNetHelper": OllamaControlNetHelper,
    "OllamaCreatureCreator": OllamaCreatureCreator,
    "OllamaDetailInjector": OllamaDetailInjector,
    "OllamaEmotionDirector": OllamaEmotionDirector,
    "OllamaHandPoseHelper": OllamaHandPoseHelper,
    "OllamaImageToStory": OllamaImageToStory,
    "OllamaLoraSuggester": OllamaLoraSuggester,
    "OllamaTextureMaterial": OllamaTextureMaterial,
    # Integrated utility/media/composition nodes
    **IMAGE_NODES,
    **MASK_NODES,
    **TEXT_NODES,
    **MATH_NODES,
    **FLOW_NODES,
    **UTIL_NODES,
    **ANIM_NODES,
    **COLOR_NODES,
    **VIDEO_NODES,
    **STORY_NODES,
    **COMP_NODES,
    **VFX_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaModelLoader": "Ollama Model Loader",
    "OllamaTextGenerate": "Ollama Text Generate",
    "OllamaChat": "Ollama Chat",
    "OllamaVision": "Ollama Vision (Multi-modal)",
    "OllamaEmbeddings": "Ollama Embeddings",
    "OllamaPromptBuilder": "Ollama Prompt Builder",
    "OllamaSinglePromptBuilder": "Ollama Single Prompt Builder",
    "OllamaJSONExtractor": "Ollama JSON Extractor",
    "OllamaModelManager": "Ollama Model Manager",
    "OllamaPromptEnhancer": "Ollama Prompt Enhancer",
    "OllamaSceneDirector": "Ollama Scene Director",
    "OllamaStyleTransfer": "Ollama Style Transfer",
    "OllamaImageAnalyzer": "Ollama Image Analyzer",
    "OllamaImageEdit": "Ollama Image Edit Prompt",
    "OllamaImageSequenceAnalyzer": "Ollama Image Sequence Analyzer",
    "OllamaPhotosetFolderAnalyzer": "Ollama Photoset Folder Analyzer",
    "OllamaAnalyzerPromptSaver": "Ollama Analyzer Prompt Saver",
    "OllamaWebImageTool": "Ollama Web Image Tool",
    "OllamaImageMerger": "Ollama Image Merger",
    "OllamaCharacterSheet": "Ollama Character Sheet",
    "OllamaNegativePrompt": "Negative Prompt Generator",
    "OllamaPromptVariations": "Batch Prompt Variations",
    "OllamaInpaintPrompt": "Inpainting Prompt Generator",
    "OllamaPoseDescriptor": "Pose Descriptor",
    "OllamaOutfitGenerator": "Outfit Generator",
    "OllamaEnvironmentTransform": "Environment Transformer",
    "OllamaImageComparator": "Image Comparator",
    "OllamaStoryboardGenerator": "Storyboard Generator",
    "OllamaRegionalPrompts": "Regional Prompt Generator",
    "OllamaPromptCritic": "Prompt Critic",
    "OllamaAutoTagger": "Auto Tagger",
    "OllamaColorPalette": "Color Palette Extractor",
    "OllamaStyleIdentifier": "Art Style Identifier",
    "OllamaAspectOptimizer": "Aspect Ratio Optimizer",
    "OllamaAdvancedSceneDirector": "Advanced Scene Director",
    "OllamaSubjectBuilder": "Subject Builder",
    "OllamaLightingDesigner": "Lighting Designer",
    "OllamaPromptCombiner": "Prompt Combiner",
    "OllamaPromptTranslator": "Prompt Translator",
    "OllamaWildcardGenerator": "Wildcard Generator",
    "OllamaLTXVVideoPrompt": "LTXV 2.3 Video Prompt Generator",
    "OllamaWanVideoPrompt": "Wan 2.2 Video Prompt Generator",
    "LTXIngredientsPrompt": "LTX Ingredients Prompt",
    "OllamaVideoShotDirector": "Video Shot Director",
    "OllamaCharacterAnchorExtractor": "Character Consistency Anchor Extractor",
    "OllamaOutfitSheetGenerator": "Outfit Sheet Generator",
    "OllamaPoseSheetGenerator": "Pose Sheet Generator",
    "OllamaExpressionSheetGenerator": "Expression Sheet Generator",
    "OllamaPromptContinuityChecker": "Prompt Continuity Checker",
    "OllamaActionChoreographer": "Action Choreographer",
    "OllamaBackgroundGenerator": "Background Generator",
    "OllamaControlNetHelper": "ControlNet Helper",
    "OllamaCreatureCreator": "Creature Creator",
    "OllamaDetailInjector": "Detail Injector",
    "OllamaEmotionDirector": "Emotion Director",
    "OllamaHandPoseHelper": "Hand Pose Helper",
    "OllamaImageToStory": "Image to Story",
    "OllamaLoraSuggester": "LoRA Suggester",
    "OllamaTextureMaterial": "Texture & Material Designer",
    **IMAGE_DISPLAY,
    **MASK_DISPLAY,
    **TEXT_DISPLAY,
    **MATH_DISPLAY,
    **FLOW_DISPLAY,
    **UTIL_DISPLAY,
    **ANIM_DISPLAY,
    **COLOR_DISPLAY,
    **VIDEO_DISPLAY,
    **STORY_DISPLAY,
    **COMP_DISPLAY,
    **VFX_DISPLAY,
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
