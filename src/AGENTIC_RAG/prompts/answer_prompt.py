from langchain_core.prompts import PromptTemplate


json_format_open = """{
  "thinking": "<step by step reasoning: what the evidence reveals and how it leads to the answer>",
  "reasoning": "<Pro-claims for the answer, followed by Con-claims of alternative findings, citing all relevant evidence IDs (e.g., [E1], [E4])>",
  "answer": "<concise answer to the question>",
  "evidences": ["E1", "E4"]
}"""

json_format_mc = """{
  "thinking": "<step by step reasoning: what the insights reveal, how they map to options, which option is best supported>",
  "reasoning": "<Justify the correct option (Pro) and provide explicit rebuttals for why other distractors are less likely (Cons), citing all relevant evidences IDs (e.g., [E1], [E4])>",
  "answer_letter": "<single correct option letter: A/B/C/D/E>",
  "evidences": ["E1", "E4"]
}"""

answer_agent_prompt_open = PromptTemplate(
    input_variables=["question", "domain", "insights_str"],
    template="""You are an expert {domain} medical specialist.

QUESTION: {question}

Insights:
{insights_str}

Instructions:
- The Insights contains answers to subqueries generated from the question, each with an Answer, a Reasoning, and evidences used to generate the answer (e.g. [E1], [E2], [E3]).
- Synthesize a comprehensive clinical explanation for the correct answer to the original question using ONLY the information in the Insights.
- **Reasoning Structure:** In the "reasoning" field, provide an argumentation-style analysis. 
  - Explicitly state the primary claims (Pros) that support the answer, and address any conflicting findings or limitations (Cons) found in all relevant evidence (e.g., [E1], [E4]).
  - **Crucial:** Do NOT mention "subqueries," or "subquery indices" (e.g., avoid "According to Subquery 1"). Use only the evidence labels like [E1], [E2].
- In "thinking": reason step by step — what the insights reveal and how they map to the final answer. 
- In "answer": provide a concise, direct answer to the question.
- In "evidences": list only the IDs whose Evidences directly support your final answer.

Respond ONLY with valid JSON (no markdown, no extra text):
{json_format}
""",
).partial(json_format=json_format_open)

answer_agent_prompt_mc = PromptTemplate(
    input_variables=["question", "options", "domain", "insights_str"],
    template="""You are an expert {domain} medical specialist.

QUESTION: {question}
OPTIONS: {options}

Insights:
{insights_str}

Instructions:
- The Insights contains answers to subqueries generated from the question, each with an Answer, a Reasoning, and evidences used to generate the answer (labeled [E1], [E4], etc.).
- Synthesize a comprehensive clinical explanation for the correct answer to the original question using ONLY the information in the Insights.
- **Reasoning Structure:** In the "reasoning" field, use a differential-style argumentation. 
  - Justify the correct option (Pro) and provide explicit rebuttals for why other distractors are less likely based on all relevant evidence (e.g., [E1], [E4]).
  - **Crucial:** Do NOT mention "subqueries," or "subquery indices" (e.g., avoid "According to Subquery 1"). Use only the evidence labels like [E1], [E2].
- In "thinking": reason step by step — what the insights reveal, how they map to each option, which is best supported.
- In "answer_letter": return the single correct option letter (A/B/C/D/E). If the Insights are insufficient then return "X".
- In "evidences": list only the IDs whose Evidences directly support your final answer.

Respond ONLY with valid JSON (no markdown, no extra text):
{json_format}
""",
).partial(json_format=json_format_mc)


