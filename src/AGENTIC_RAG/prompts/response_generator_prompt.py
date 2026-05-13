from langchain_core.prompts import PromptTemplate

json_format = """{
  "thinking": "<step by step reasoning: what the subquery asks, what the evidence says, what answers are plausible>",
  "answer": "<concise answer to the subquery>",
  "reasoning": "<explanation of the answer, citing evidence numbers, e.g. [E6], [E8]>",
  "evidence_ids": ["E6", "E8"],
  "evidence_quotes": {
    "E1": "<exact all sentences from [E1] that supports the best answer>",
    "E3": "<exact all sentences from [E3] that supports the best answer>"
  }
}"""
 
response_prompt = PromptTemplate(
    input_variables=["Original Question", "question", "domain", "vignette", "retrieved_texts"],
    template="""You are an expert {domain} medical assistant.
 
ORIGINAL QUESTION: {Original Question}

SUBQUERY: {question}

VIGNETTE: {vignette}

EVIDENCE:
{retrieved_texts}
 
Instructions:
- Answer the subquery with the understanding that it is derived from the original question, citing evidence by its exact label (e.g., [E6], [E7]). Use only the provided evidence and do not rely on outside knowledge.
- Evidence items are labeled with their global IDs (e.g. [E6], [E7]) as shown above.
- In "thinking": reason step by step — what is the subquery asking, what does the evidence say, what answers are plausible. Cite evidence by its exact label (e.g. [E6], [E7]).
- "reasoning" must explain the correctness of the answers, citing evidence IDs (e.g. [E6], [E8]).
- "evidence_ids" must be a list of the exact global ID strings cited (e.g. ["E6", "E8"]).
- For each cited evidence ID, copy EXACT all sentences from that evidence item into "evidence_quotes" that directly support your answer. Do not paraphrase.
- Do NOT use outside knowledge — stay strictly within the provided evidence.
Respond ONLY with valid JSON (no markdown, no extra text):
{json_format}
""",
).partial(json_format=json_format)
 