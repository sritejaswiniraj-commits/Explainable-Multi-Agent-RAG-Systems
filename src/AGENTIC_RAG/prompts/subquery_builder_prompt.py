from langchain_core.prompts import PromptTemplate

json_format_options = """{
  "Question1": "<Sub-question 1>",
  "Question2": "<Sub-question 2>",
  "Question3": "<Sub-question 3>",
  "Question4": "<Sub-question 4>"
}"""

subquery_builder_prompt_options = PromptTemplate(
    input_variables=["question", "domain", "options", "vignette"],
    template="""You are a knowledgeable {domain} medical assistant.

Decompose the clinical question into 3–4 independent sub-questions.

Each sub-question must:
- Be self-contained and answerable independently.
- Target a single clinical concept.
- Be necessary to solve the question.
- Be designed to differentiate between the given answer options (focus on key differences between options).
- Be concise and optimized for retrieval.

Question: {question}
Options: {options}
Vignette: {vignette}

Rules:
- No overlap or redundancy.
- No explanations or reasoning.
- Ensure all key aspects needed to choose the correct option are covered.
- Use clinically relevant concepts only.

Output ONLY valid JSON:
{json_format_options}
""",
).partial(json_format_options=json_format_options)


from langchain_core.prompts import PromptTemplate

json_format_noptions = """{
  "Question1": "<Sub-question 1>",
  "Question2": "<Sub-question 2>",
  "Question3": "<Sub-question 3>",
  "Question4": "<Sub-question 4>"
}"""

subquery_builder_prompt_noptions = PromptTemplate(
    input_variables=["question", "domain", "vignette"],
    template="""You are a knowledgeable {domain} medical assistant.

Your task is to decompose the following clinical question into 3–4 independent sub-questions.

Each sub-question must:
- Be answerable on its own without needing the others.
- Target a single, specific clinical concept.
- Be necessary for solving the original question.
- Contribute a distinct piece of information that, when combined with the others, leads to the final answer.
- Be clear, concise, and optimized for search (self-contained, no ambiguity).

Think of this as breaking the problem into modular parts where each answer provides a building block toward solving the full question.

Question: {question}
Vignette Summary: {vignette}

Rules:
- Do NOT create overlapping or redundant sub-questions.
- Do NOT include reasoning or explanations.
- Ensure coverage of all key aspects required to answer the original question.
- Focus only on clinically relevant concepts using {domain} expertise.

Respond ONLY with valid JSON (no markdown, no explanation):
{json_format_noptions}
""",
).partial(json_format_noptions=json_format_noptions)
