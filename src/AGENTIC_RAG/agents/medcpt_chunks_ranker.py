from src.AGENTIC_RAG.agents.base_agent import BaseAgent
from src.AGENTIC_RAG.state.agent_state import AgentState
import torch


class MedCPTChunkRerankerAgent(BaseAgent):
    def __init__(self, model_wrapper, chunk_index, top_k = 5):
        super().__init__("MedCPTChunkRerankerAgent", llm=None)
        self.model_wrapper = model_wrapper
        self.cross_tokenizer = model_wrapper.cross_tokenizer
        self.cross_model = model_wrapper.cross_model
        self.chunk_index = chunk_index
        self.top_k = top_k

    @staticmethod
    def _get_or_create_id(state, chunk_text):
        if not hasattr(state, "chunk_id_registry"):
            state.chunk_id_registry = {}  
            state.chunk_id_counter  = 0   

        if chunk_text not in state.chunk_id_registry:
            state.chunk_id_counter += 1
            state.chunk_id_registry[chunk_text] = f"E{state.chunk_id_counter}"

        return state.chunk_id_registry[chunk_text]


    def run(self, state):
        subquery = state.subquery_questions[self.chunk_index]
        chunks = [chunk.page_content for chunk in state.raw_chunks[self.chunk_index]]

        if not chunks:
            state.reranked_chunks[self.chunk_index] = []
            return state

        combined_query = f"{state.question}\n\nSpecific focus: {subquery}"
        pairs = [[combined_query, chunk] for chunk in chunks]

        with torch.no_grad():
            encoded = self.cross_tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            logits = self.cross_model(**encoded).logits.squeeze(dim=1).tolist()

        if isinstance(logits, float):
            logits = [logits]

        scored = sorted(
            [
                {
                    "id":self._get_or_create_id(state, chunks[i]),
                    "chunk": chunks[i],
                    "score": round(logits[i], 4),
                }
                for i in range(len(chunks))
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[: self.top_k]

        state.reranked_chunks[self.chunk_index] = scored
        return state