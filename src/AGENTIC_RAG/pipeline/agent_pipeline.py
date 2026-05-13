from concurrent.futures import ThreadPoolExecutor, as_completed
from src.AGENTIC_RAG.state.agent_state import AgentState
from src.AGENTIC_RAG.agents.domian_extract import DomainExtractAgent
from src.AGENTIC_RAG.agents.vignette import VignetteAgent
from src.AGENTIC_RAG.agents.subquery_builder import SubqueryBuilderAgent
from src.AGENTIC_RAG.agents.chunk_retriever import ChunkRetrieverAgent
from src.AGENTIC_RAG.agents.medcpt_chunks_ranker import MedCPTChunkRerankerAgent
from src.AGENTIC_RAG.agents.response_generator_agent import ResponseGeneratorAgent
from src.AGENTIC_RAG.agents.answer import AnswerAgent
from src.AGENTIC_RAG.agents.match_option_agent import MatchOptionAgent


class Pipeline:
    def __init__(self, llm_light, llm_heavy, vectorstore, model_wrapper,
                 retrieve_k=10, rerank_top_k=5, log_dir="logs", has_options=True):
        self.llm_light = llm_light
        self.llm_heavy = llm_heavy
        self.vectorstore = vectorstore
        self.model_wrapper = model_wrapper
        self.retrieve_k = retrieve_k
        self.rerank_top_k = rerank_top_k
        self.log_dir = log_dir
        self.has_options = has_options

    def run(self, question, options, sample_id = None):
        state = AgentState(question=question, options=options)

        DomainExtractAgent(self.llm_light).run(state)
        VignetteAgent(self.llm_light).run(state)
        SubqueryBuilderAgent(self.llm_light, has_options=self.has_options).run(state)

        queries = list(state.subqueries.values())

        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = {executor.submit(self._process_subquery, state, idx, q): idx
                       for idx, q in enumerate(queries)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in subquery {idx}: {e}")

        AnswerAgent(self.llm_heavy, has_options=self.has_options).run(state)
        if not self.has_options:
            MatchOptionAgent(self.llm_light).run(state)

        state.save_log(self.log_dir, sample_id)
        return state

    def _process_subquery(self, state, idx, query):
        ChunkRetrieverAgent(vectorstore=self.vectorstore, subquery=query,
                            chunk_index=idx, k=self.retrieve_k).run(state)
        MedCPTChunkRerankerAgent(model_wrapper=self.model_wrapper,
                                 chunk_index=idx, top_k=self.rerank_top_k).run(state)
        ResponseGeneratorAgent(llm=self.llm_heavy, subquery_index=idx).run(state)
