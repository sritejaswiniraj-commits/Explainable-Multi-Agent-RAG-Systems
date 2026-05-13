from src.AGENTIC_RAG.agents.base_agent import BaseAgent
from src.AGENTIC_RAG.state.agent_state import AgentState
from src.AGENTIC_RAG.prompts.match_options import final_answer_agent_prompt, json_format


class MatchOptionAgent(BaseAgent):

    def __init__(self, llm):
        super().__init__("MatchOptionAgent", llm)
        self.prompt = final_answer_agent_prompt

    def run(self, state):
        options = state.options
        answer = state.answer
        reasoning = state.reasoning

        chain  = self.prompt | self.llm
        result = chain.invoke({
            "options": options,
            "answer": answer,
            "reasoning": reasoning,
        })

        parsed = self.safe_parse(result, json_format)

        state.answer_letter = parsed.get("answer_letter", "")
        return state 