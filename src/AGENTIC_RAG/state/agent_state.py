import os
import json
from dataclasses import dataclass, field


@dataclass
class AgentState:
    question: str
    options: dict
    domain: str = ""
    vignette: dict = field(default_factory=dict)
    subqueries: dict = field(default_factory=dict)

    raw_chunks: dict = field(default_factory=dict)
    reranked_chunks: dict = field(default_factory=dict)
    subquery_questions: dict = field(default_factory=dict)
    subquery_thinkings: dict = field(default_factory=dict)
    subquery_possible_answers: dict = field(default_factory=dict)
    subquery_cited_evidence_ids: dict = field(default_factory=dict)
    subquery_reasonings: dict = field(default_factory=dict)
    subquery_evidence_quotes: dict = field(default_factory=dict)

    option_chunks: dict = field(default_factory=dict)

    thinking: str = ""
    answer: str = ""
    answer_letter: str = ""
    reasoning: str = ""
    evidences: list = field(default_factory=list)
    evidences_ids: list = field(default_factory=list)
    quotes: dict = field(default_factory=dict)

    def save_log(self, log_dir: str, sample_id: int):
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"sample_{sample_id}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def to_dict(self):
        def _clean(obj):
            if isinstance(obj, list):
                return [_clean(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if hasattr(obj, "page_content"):
                return {"text": obj.page_content, "metadata": obj.metadata}
            return obj
        return _clean({key: getattr(self, key) for key in self.__dataclass_fields__})