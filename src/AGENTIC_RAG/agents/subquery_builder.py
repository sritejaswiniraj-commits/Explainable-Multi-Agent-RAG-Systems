from src.AGENTIC_RAG.agents.base_agent import BaseAgent
from src.AGENTIC_RAG.prompts.subquery_builder_prompt import json_format_noptions, subquery_builder_prompt_noptions, json_format_options, subquery_builder_prompt_options


class SubqueryBuilderAgent(BaseAgent):
    def __init__(self, llm, has_options):
        super().__init__("SubqueryBuilderAgent", llm)
        self.has_options = has_options
        self.prompt = subquery_builder_prompt_options if has_options else subquery_builder_prompt_noptions
        self.json_format = json_format_options if has_options else json_format_noptions

    def run(self, state):
        chain = self.prompt | self.llm
        if self.has_options:
             result = chain.invoke({
                "question": state.question,
                "options": state.options,
                "domain": state.domain,
                "vignette": state.vignette,
            })
        else:   
            result = chain.invoke({
                "question": state.question,
                "domain": state.domain,
                "vignette": state.vignette,
            })
        parsed = self.safe_parse(result, self.json_format)
        state.subqueries = parsed if isinstance(parsed, dict) and parsed else {}
        return state