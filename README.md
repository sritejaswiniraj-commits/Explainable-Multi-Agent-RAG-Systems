# Agentic RAG for Medical Question Answering

A multi-agent Retrieval-Augmented Generation (RAG) system for answering USMLE-style medical questions from the **MedQA** dataset. The pipeline combines domain-aware query decomposition, medical-domain embeddings, cross-encoder re-ranking, and an ensemble of LLM agents to produce evidence-grounded answers.

> Course project for **Human Centered Natural Langugae Processing**

---

## Overview

Standard RAG often struggles on clinical questions because they require multi-hop reasoning (symptoms → diagnosis → mechanism → treatment). This project addresses that by breaking each question into focused sub-queries, retrieving evidence per sub-query in parallel, reranking with a medical cross-encoder (MedCPT), and finally synthesizing an answer with a heavier LLM.

Three approaches are implemented for comparison:

1. **Simple RAG** — single retrieval + single LLM call (baseline).
2. **Agentic RAG** — multi-agent pipeline with sub-query decomposition and reranking (main contribution).
3. **Chunking experiments** — fixed-size, semantic, and **ASCM** (Adaptive Semantic Chunk Modeling) chunkers for ablation.

---

## Repository Structure

```
Group10_HCNLP26/
├── HCNLP_Group10_Presentation.pdf      # Project presentation
└── src/
    ├── CHUNKING/                       # Textbook chunking strategies
    │   ├── ascm_chunker.py             # Adaptive semantic chunking (our method)
    │   ├── fixed_size_chunking.py      # Fixed-size baseline
    │   └── langchain_semantic_chunking.py
    │
    ├── VectorDB/
    │   └── store_embeddings.py         # Build Chroma vector store from chunks
    │
    ├── SimpleRag/
    │   └── simple_rag.py               # Baseline single-shot RAG
    │
    └── AGENTIC_RAG/                    # Main multi-agent pipeline
        ├── run_agentic_rag.py          # Entry point
        ├── pipeline/agent_pipeline.py  # Orchestrates the agents
        ├── state/agent_state.py        # Shared state passed between agents
        ├── agents/                     # Individual agents (see below)
        ├── prompts/                    # Prompt templates for each agent
        ├── model_loader/               # MedCPT cross-encoder wrapper
        └── Sample results/             # Example output JSON
```

---

## The Agentic Pipeline

The agentic pipeline runs the following agents per question:

| Step | Agent | Role |
|------|-------|------|
| 1 | `DomainExtractAgent` | Identifies the clinical domain (e.g., cardiology, pharmacology). |
| 2 | `VignetteAgent` | Extracts structured clinical facts from the question stem. |
| 3 | `SubqueryBuilderAgent` | Decomposes the question into focused sub-queries. |
| 4 | `ChunkRetrieverAgent` | Retrieves top-*k* chunks from Chroma per sub-query (parallel). |
| 5 | `MedCPTChunkRerankerAgent` | Reranks chunks with `ncbi/MedCPT-Cross-Encoder`. |
| 6 | `ResponseGeneratorAgent` | Produces per-sub-query reasoning + cited evidence. |
| 7 | `AnswerAgent` | Aggregates insights and selects the final option. |
| 8 | `MatchOptionAgent` *(no-options mode only)* | Maps an open-ended answer to one of the MCQ letters. |

Each sub-query is retrieved, reranked, and reasoned over in parallel using a `ThreadPoolExecutor`, and the final state for each sample is saved as JSON for inspection and evaluation.

---

## Models & Data

- **Embeddings**: [`abhinand/MedEmbed-large-v0.1`](https://huggingface.co/abhinand/MedEmbed-large-v0.1)
- **Reranker**: [`ncbi/MedCPT-Cross-Encoder`](https://huggingface.co/ncbi/MedCPT-Cross-Encoder)
- **LLMs** (via Groq API):
  - Heavy: `openai/gpt-oss-20b`
  - Light: `llama-3.1-8b-instant`
- **Vector store**: ChromaDB (persistent, local)
- **Dataset**: [MedQA — USMLE](https://github.com/jind11/MedQA) (US split, `test.jsonl`); the corpus is the accompanying English medical textbooks.

---

## Acknowledgements

- [MedQA](https://github.com/jind11/MedQA) dataset
- NCBI [MedCPT](https://huggingface.co/ncbi/MedCPT-Cross-Encoder) cross-encoder
- [MedEmbed](https://huggingface.co/abhinand/MedEmbed-large-v0.1) embeddings
- LangChain, ChromaDB, and Groq for infrastructure
