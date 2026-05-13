from abc import ABC, abstractmethod
from src.AGENTIC_RAG.state.agent_state import AgentState
from json_repair import repair_json
from src.AGENTIC_RAG.prompts.json_corrector_prompt import json_corrector_prompt


class BaseAgent(ABC):
    def __init__(self, name, llm):
        self.name = name
        self.llm = llm

    def log(self, message):
        print(f"[{self.name}] {message}")

    def log_section(self, title):
        border = "=" * 60
        print(f"\n{border}")
        print(f"[{self.name}] >>> {title}")
        print(border)

    @abstractmethod
    def run(self, state):
        pass

    def extract_text(self, result):
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def parse_llm_json(self, result):
        raw_text = self.extract_text(result)
        return repair_json(raw_text, return_objects=True)

    def json_corrector(self, input_json, json_format):
        chain = json_corrector_prompt | self.llm
        result = chain.invoke({"input_json": input_json, "json_format": json_format})
        return self.parse_llm_json(result)

    def safe_parse(self, result, json_format):
        raw_text = self.extract_text(result)
        parsed = self.parse_llm_json(raw_text)

        if isinstance(parsed, dict) and parsed:
            return parsed

        corrected = self.json_corrector(raw_text, json_format)
        if isinstance(corrected, (dict, list)) and corrected:
            return corrected

        return {}