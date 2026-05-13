from src.AGENTIC_RAG.agents.base_agent import BaseAgent
from src.AGENTIC_RAG.state.agent_state import AgentState
from src.AGENTIC_RAG.prompts.domian_extract import domain_extract_prompt, json_format


class DomainExtractAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__("DomainExtractAgent", llm)
        self.prompt = domain_extract_prompt

    def run(self, state):
        chain = self.prompt | self.llm
        result = chain.invoke({"question": state.question})
        parsed = self.safe_parse(result, json_format)
        state.domain = parsed.get("domain", "")
        return state