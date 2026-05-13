import json

from src.AGENTIC_RAG.agents.base_agent import BaseAgent
from src.AGENTIC_RAG.state.agent_state import AgentState
from src.AGENTIC_RAG.prompts.answer_prompt import (
    answer_agent_prompt_open,
    answer_agent_prompt_mc,
    json_format_open,
    json_format_mc,
)


class AnswerAgent(BaseAgent):
    def __init__(self, llm, has_options):
        super().__init__("AnswerAgent", llm)
        self.has_options = has_options

    def run(self, state):
        format_insights = self._format_insights(state)

        if self.has_options:
            prompt = answer_agent_prompt_mc
            json_format = json_format_mc
            invoke_kwargs = {
                "question": state.question,
                "options": state.options,
                "domain": state.domain,
                "insights_str": format_insights,
            }
        else:
            prompt = answer_agent_prompt_open
            json_format = json_format_open
            invoke_kwargs = {
                "question": state.question,
                "domain": state.domain,
                "insights_str": format_insights,
            }

        chain = prompt | self.llm
        result = chain.invoke(invoke_kwargs)
        
        parsed = self.safe_parse(result, json_format)

        state.thinking = parsed.get("thinking", "")
        state.reasoning = parsed.get("reasoning", "")

        if self.has_options:
            state.answer_letter = parsed.get("answer_letter", "")
        else:
            state.answer = parsed.get("answer", "")

        id_to_text = {}
        for chunks in state.reranked_chunks.values():
            for item in chunks:
                id_to_text[item["id"]] = item["chunk"]

        id_to_quote = {}
        for quotes in state.subquery_evidence_quotes.values():
            for eid, quote in quotes.items():
                id_to_quote[eid] = quote

        selected_ids = parsed.get("evidences", [])
        valid_eids = [eid for eid in selected_ids if isinstance(eid, str) and eid in id_to_text]

        state.evidences_ids = valid_eids
        state.evidences = [id_to_text[eid] for eid in valid_eids]
        state.quotes = {eid: id_to_quote[eid] for eid in valid_eids if eid in id_to_quote}

        return state

    def _format_insights(self, state):
        lines = []
        for i in sorted(state.subquery_questions.keys()):
            lines.append(f"[Subquery {i+1}] {state.subquery_questions[i]}")
            lines.append(f"  Answer: {state.subquery_possible_answers[i]}")
            lines.append(f"  Reasoning: {state.subquery_reasonings[i]}")
            lines.append(f"  Evidences:")

            quotes = state.subquery_evidence_quotes.get(i, {})
            for eid, quote in quotes.items():
                lines.append(f"    [{eid}]: {quote}")

        return "\n".join(lines)