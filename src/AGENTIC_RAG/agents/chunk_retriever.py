from src.AGENTIC_RAG.agents.base_agent import BaseAgent
from src.AGENTIC_RAG.state.agent_state import AgentState


class ChunkRetrieverAgent(BaseAgent):
    def __init__(self, vectorstore, subquery, chunk_index, k= 10):
        super().__init__("ChunkRetrieverAgent", llm=None)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.subquery = subquery
        self.chunk_index = chunk_index

    def run(self, state):
        combined_query = f"{state.question}\n\nSpecific focus: {self.subquery}"
        state.raw_chunks[self.chunk_index] = self.retriever.invoke(combined_query)
        state.subquery_questions[self.chunk_index] = self.subquery 
        return state