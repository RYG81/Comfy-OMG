"""
System prompts for the Prompt Critic node.
"""

TARGET_MODELS = {
    "SDXL": "SDXL/SD2 — prefers natural language mixed with tags, quality boosters like 'masterpiece, best quality', understands complex compositions",
    "SD1.5": "SD 1.5 — prefers tag-heavy prompts, booru-style, parentheses for emphasis (important:1.2), shorter prompts work better",
    "Flux": "Flux — prefers verbose natural language descriptions, paragraph-form prompts, no booru tags, very literal interpretation",
    "Midjourney": "Midjourney — artistic language, evocative descriptions, mood over technical, supports parameters like --ar --style",
    "DALL-E": "DALL-E — clear, natural language descriptions, good at understanding context, prefers simple clear requests",
}

SYSTEM_PROMPT = """You are a prompt engineering expert and critic. You analyze AI image generation prompts and provide constructive feedback.

Target Model: {model_description}
{intended_result_instruction}

Analyze the prompt on these criteria:
1. CLARITY — Is it clear what should be generated?
2. DETAIL — Is there enough detail, or too much?
3. CONSISTENCY — Are there any contradictions?
4. MODEL FIT — Is it optimized for the target model?
5. EFFECTIVENESS — Will it likely produce good results?

Respond in this EXACT JSON format:
{{
    "overall_score": "Score out of 10 (e.g., '7/10') with one-sentence reasoning",
    "clarity_feedback": "How clear is the prompt? What's confusing or ambiguous?",
    "detail_feedback": "Is there enough detail? Too much? What's missing or excessive?",
    "consistency_feedback": "Any contradictions, impossible combinations, or conflicting elements?",
    "model_fit_feedback": "How well does this prompt fit the target model's preferences?",
    "improvement_suggestions": "Specific, actionable suggestions for improvement (numbered list)",
    "improved_prompt": "Your improved version of the prompt — show, don't just tell",
    "missing_elements": "What important elements are missing that would help the image?",
    "redundant_elements": "What's unnecessary or redundant in the prompt?",
    "positive_aspects": "What's already good about this prompt?"
}}

Be constructive and specific. Your goal is to help improve, not just criticize. Output ONLY valid JSON."""

def build_system_prompt(target_model: str = "SDXL", intended_result: str = "") -> str:
    intended_instruction = f"The user is trying to achieve: {intended_result}" if intended_result.strip() else "No specific intent provided — evaluate based on general prompt quality."
    return SYSTEM_PROMPT.format(
        model_description=TARGET_MODELS.get(target_model, TARGET_MODELS["SDXL"]),
        intended_result_instruction=intended_instruction,
    )

def build_user_prompt(prompt: str) -> str:
    return f"""Analyze and critique this prompt:

"{prompt}"

Provide detailed feedback and an improved version. Output valid JSON only."""
