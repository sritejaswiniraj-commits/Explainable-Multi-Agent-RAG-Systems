from langchain_core.prompts import PromptTemplate

json_format = """{
    "answer_letter": "<the single best option letter that matches the provided answer and reasoning,
    or 'X' if no option matches>"
}"""


final_answer_agent_prompt = PromptTemplate(
    input_variables=["options", "answer", "reasoning"],
    template="""You are a medical expert.

Your task is to select the option that best matches the provided answer and reasoning.

Options:
{options}

Previously generated answer: {answer}
Reasoning behind the answer: {reasoning}

Instructions:
- Carefully compare the provided answer and reasoning against each option one by one.
- Think in detail about how well each option aligns with the provided answer.
- Select the ONE option that most closely matches.
- If the answer and reasoning do not clearly match any of the options, return "X".
- Do NOT use any outside knowledge — only match based on the answer and reasoning given.
- Respond ONLY with valid JSON — no explanation, no markdown:

{json_format}
"""
).partial(json_format=json_format)