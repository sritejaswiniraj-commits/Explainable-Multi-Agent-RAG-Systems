import os
import json
from tqdm import tqdm
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from src.AGENTIC_RAG.model_loader.model_wrapper import MedCPTModelWrapper
from src.AGENTIC_RAG.pipeline.agent_pipeline import Pipeline

MODEL_FAMILY = "gpt"
MODEL_CONFIG = {
    "llama": {"model": "llama-3.1-8b-instant", "max_tokens": 2048, "model_kwargs": {}, "tag": "llama31_8b"},
    "gpt": {"model": "openai/gpt-oss-20b","max_tokens": 8000, "model_kwargs": {"include_reasoning": False}, "tag": "gpt_oss_20b"},
}
cfg = MODEL_CONFIG[MODEL_FAMILY]

EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"
CHROMA_PATH = "/home/exgr5/HCNLP/chroma_db_ascm_150"
CROSS_MODEL_PATH = "ncbi/MedCPT-Cross-Encoder"
DATA_FILE = "/home/exgr5/HCNLP/data_clean/data_clean/questions/US/test.jsonl"
HAS_OPTIONS= True
LOG_DIR = f"logs_{cfg['tag']}_HAS_OPTIONS_{HAS_OPTIONS}"
RESULTS_FILE = f"multiagent_results_{cfg['tag']}_HAS_OPTIONS_{HAS_OPTIONS}.json"

model_wrapper = MedCPTModelWrapper(CROSS_MODEL_PATH)
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding, collection_name="medical_books")

llm_heavy = ChatGroq(model=cfg["model"], max_tokens=cfg["max_tokens"], temperature=0.0,
                     model_kwargs=cfg["model_kwargs"]) | StrOutputParser()
llm_light = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0) | StrOutputParser()

pipeline = Pipeline(
    llm_light=llm_heavy, llm_heavy=llm_heavy,
    vectorstore=vectorstore, model_wrapper=model_wrapper,
    retrieve_k=10, rerank_top_k=5,
    log_dir=LOG_DIR, has_options=HAS_OPTIONS,
)

with open(DATA_FILE) as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} questions | Heavy model: {cfg['model']}")

results = []
correct_count = 0

for i, item in enumerate(tqdm(data, desc=f"Processing ({cfg['tag']})")):
    true_answer_letter = item["answer_idx"]

    try:
        state = pipeline.run(
            question=item["question"],
            options=item["options"],
            sample_id=i,
        )
        entry = state.to_dict()
        if not entry.get("answer_letter"):
            tqdm.write(f"[WARN] Sample {i} — empty answer_letter. Subquery choices: {state.subquery_best_choices}")
    except Exception as e:
        tqdm.write(f"[ERROR] Sample {i} failed: {e}")
        entry = {
            "question": item["question"],
            "options": item["options"],
            "error": str(e),
            "answer_letter": "",
        }

    entry["index"] = i
    entry["true_answer_letter"] = true_answer_letter

    if entry.get("answer_letter") == true_answer_letter:
        correct_count += 1

    results.append(entry)

    with open(RESULTS_FILE, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)

    acc = correct_count / len(results)
    tqdm.write(f"Sample {i} | Acc: {acc:.4f} | Pred: {entry.get('answer_letter')} | True: {true_answer_letter}")

final_acc = correct_count / len(results) if results else 0
print(f"\nFinished. Total: {len(results)} | Final Accuracy: {final_acc:.4f}")