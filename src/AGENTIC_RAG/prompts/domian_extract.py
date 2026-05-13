from langchain_core.prompts import PromptTemplate

json_format = """{
    "domain": "<primary medical domain>"
}"""

domain_extract_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Classify this medical question into one primary domain and understand what is being asked.

QUESTION: {question}

Respond ONLY with this JSON (no markdown, no explanation):
{json_format}
""",
).partial(json_format=json_format)