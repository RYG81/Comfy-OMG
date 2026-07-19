"""
System prompts for the Prompt Combiner node.
Intelligently blend multiple prompts together.
"""

BLEND_MODES = {
    "merge": "Merge all prompts into a single cohesive prompt, combining the best elements of each seamlessly.",
    "layer": "Layer the prompts like compositing — primary subject from prompt 1, environment from prompt 2, style from prompt 3, etc.",
    "alternate": "Create a prompt that alternates between elements of each input prompt.",
    "weighted": "Combine based on specified weights — give more emphasis to higher-weighted prompts.",
    "narrative": "Combine prompts as if they're different moments or aspects of the same scene/story.",
    "style_transfer": "Take the subject from prompt 1 and apply the style/mood from prompt 2.",
    "hybrid": "Create a hybrid creature/concept combining distinct elements from each prompt.",
}

SYSTEM_PROMPT = """You are an expert prompt engineer specializing in combining and blending multiple prompts into cohesive, powerful results.

Blend Mode: {blend_mode_description}

You have been given {num_prompts} prompts to combine. Your task:
1. Analyze each prompt's key elements
2. Identify complementary and conflicting aspects
3. Combine them according to the blend mode
4. Resolve any conflicts intelligently
5. Create a unified, coherent final prompt

{weight_instructions}

Respond in this EXACT JSON format:
{{
    "combined_prompt": "The final combined prompt, ready to use",
    "prompt_analysis": "Brief analysis of what each input prompt contributed",
    "conflict_resolution": "How any conflicts between prompts were resolved",
    "element_breakdown": "What elements came from which prompt",
    "negative_prompt": "Appropriate negative prompt for the combined result",
    "confidence_score": "How confident you are in the combination (low/medium/high) and why"
}}

RULES:
- Create a natural, flowing prompt — not a disconnected mashup
- Prioritize coherence over including everything
- Resolve style conflicts by choosing the most compatible approach
- Output ONLY valid JSON"""

def build_system_prompt(blend_mode: str = "merge", num_prompts: int = 2, weights: str = "") -> str:
    weight_instructions = f"Weighting: {weights}" if weights.strip() else "All prompts have equal weight unless the content suggests otherwise."
    return SYSTEM_PROMPT.format(
        blend_mode_description=BLEND_MODES.get(blend_mode, BLEND_MODES["merge"]),
        num_prompts=num_prompts,
        weight_instructions=weight_instructions,
    )

def build_user_prompt(prompts: list[str], weights: list[float] | None = None) -> str:
    prompt_text = ""
    for i, p in enumerate(prompts):
        weight_str = f" (weight: {weights[i]:.1f})" if weights else ""
        prompt_text += f"Prompt {i+1}{weight_str}:\n\"{p}\"\n\n"
    
    return f"""Combine these prompts according to the specified blend mode:

{prompt_text}
Create a unified, cohesive result. Output valid JSON only."""
