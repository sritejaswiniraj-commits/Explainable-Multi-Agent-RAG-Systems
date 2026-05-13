from src.AGENTIC_RAG.agents.base_agent import BaseAgent
from src.AGENTIC_RAG.prompts.vignette import vignette_prompt, json_format


class VignetteAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__("VignetteAgent", llm)
        self.prompt = vignette_prompt

    def run(self, state):
        chain = self.prompt | self.llm
        result = chain.invoke({"question": state.question, "domain": state.domain})
        state.vignette = self.safe_parse(result, json_format)
        return state