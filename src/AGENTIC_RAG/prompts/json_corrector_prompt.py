from langchain_core.prompts import PromptTemplate

json_corrector_prompt = PromptTemplate(
    input_variables=["input_json", "json_format"],
    template="""You are a JSON repair assistant.

Task:
- Fix the provided JSON so it is valid.
- Do NOT change the structure unless necessary.
- Do NOT change the content.
- Respond ONLY with the corrected JSON — no preamble, no explanation, no markdown.

Input JSON:
{input_json}

The corrected JSON must follow this format:
{json_format}
""",
)