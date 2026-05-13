from src.AGENTIC_RAG.agents.base_agent import BaseAgent
from src.AGENTIC_RAG.state.agent_state import AgentState
from src.AGENTIC_RAG.prompts.response_generator_prompt import response_prompt, json_format


class ResponseGeneratorAgent(BaseAgent):
    def __init__(self, llm, subquery_index):
        super().__init__("ResponseGeneratorAgent", llm)
        self.prompt = response_prompt
        self.subquery_index = subquery_index

    def run(self, state):
        idx = self.subquery_index
        subquery = state.subquery_questions[idx]
      
        reranked = state.reranked_chunks[idx]
        numbered_evidence = "\n\n".join(
            f"[{item['id']}] {item['chunk']}"
            for item in reranked
        )
        chain = self.prompt | self.llm
        result = chain.invoke({
            "Original Question": state.question,
            "question":subquery,
            "domain":state.domain,
            "vignette":state.vignette,
            "retrieved_texts": numbered_evidence,
        })
        parsed = self.safe_parse(result, json_format)
        
        state.subquery_thinkings[idx] = parsed.get("thinking", "")
        state.subquery_possible_answers[idx] = parsed.get("answer", "")
        state.subquery_evidence_quotes[idx] = parsed.get("evidence_quotes", {})
        state.subquery_cited_evidence_ids[idx] = parsed.get("evidence_ids", [])
        state.subquery_reasonings[idx]= parsed.get("reasoning", "")

        return state