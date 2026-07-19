"""
System prompts for the Wildcard Generator node.
Generate prompts with randomizable variations.
"""

VARIATION_STYLE = [
    "subtle", "moderate", "dramatic", "experimental",
]

CATEGORIES_TO_VARY = [
    "all", "subject_only", "style_only", "environment_only",
    "colors_only", "mood_only", "details_only", "custom",
]

SYSTEM_PROMPT = """You are an expert at creating wildcard prompts with randomizable variations.

Wildcard syntax: {{option1|option2|option3}} — one option will be randomly selected at generation time.

Variation Style: {variation_style}
Categories to Vary: {categories}

Your task:
1. Analyze the base prompt
2. Identify elements that can be varied
3. Create wildcard options for those elements
4. Ensure all variations maintain coherence
5. Generate the requested number of options per wildcard

Number of options per wildcard: {num_options}

Respond in this EXACT JSON format:
{{
    "wildcard_prompt": "The prompt with {{wildcards|like|this}} syntax",
    "static_elements": "Elements that remain constant",
    "varied_elements": "What was made into wildcards",
    "total_combinations": "Approximate number of possible combinations",
    "sample_outputs": ["Example resolved prompt 1", "Example resolved prompt 2", "Example resolved prompt 3"],
    "variation_notes": "Notes on the variations created"
}}

RULES:
- Wildcards use curly braces: {{opt1|opt2|opt3}}
- Each option should be roughly equivalent quality
- Don't vary critical structural elements that would break coherence
- Keep variations within the same general concept
- Output ONLY valid JSON"""

def build_system_prompt(
    variation_style: str = "moderate",
    categories: str = "all",
    num_options: int = 3,
) -> str:
    style_desc = {
        "subtle": "Minor variations only — slight color changes, small detail differences",
        "moderate": "Noticeable variations — different but related options",
        "dramatic": "Significant variations — very different interpretations",
        "experimental": "Wild variations — push creative boundaries",
    }.get(variation_style, "Moderate variations")
    
    return SYSTEM_PROMPT.format(
        variation_style=style_desc,
        categories=categories,
        num_options=num_options,
    )

def build_user_prompt(base_prompt: str) -> str:
    return f"""Create a wildcard prompt from this base:

"{base_prompt}"

Add randomizable variations using {{option1|option2|option3}} syntax. Output valid JSON only."""
