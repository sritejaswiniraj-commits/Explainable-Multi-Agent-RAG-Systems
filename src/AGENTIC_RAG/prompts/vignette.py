from langchain_core.prompts import PromptTemplate

json_format = """{
    "patient_summary": "<age, sex, chief complaint>",
    "key_findings": ["<finding 1>", "<finding 2>"],
    "relevant_history": "<relevant past medical / drug history>",
    "core_problem": "<one sentence: what the question is really asking>"
}"""

vignette_prompt = PromptTemplate(
    input_variables=["question", "domain"],
    template="""You are a {domain} medical expert.

Analyze the clinical vignette below and extract ONLY the information essential to solve the question. Ignore distractors.

QUESTION: {question}

Respond ONLY with this JSON (no markdown, no explanation):
{json_format}
""",
).partial(json_format=json_format)