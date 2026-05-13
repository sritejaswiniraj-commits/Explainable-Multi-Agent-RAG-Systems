import json
import os
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from json_repair import repair_json


GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHUNKS_JSON = "/home/exgr5/HCNLP/src/TESTDATA_RETRIEVED_CHUNK/retrieved_chunks_ascm_150.json"
USE_CHUNK = "5"
OUTPUT_FILE = f"/home/exgr5/HCNLP/src/RAG_RESULTS/gpt_ascm_chunk_{USE_CHUNK}.json"
SUMMARY_FILE = f"/home/exgr5/HCNLP/src/RAG_RESULTS/gpt_ascm_summary_chunk{USE_CHUNK}.json"

option_question_answer_prompt = PromptTemplate(
    input_variables=["chunks", "question", "options_text", "valid_letters"],
    template="""You are a board-certified physician with expertise in clinical medicine. You are participating in a medical knowledge evaluation study. Your task is to answer USMLE-style multiple choice questions using ONLY the retrieved context provided.
Rules:
1. Read the context carefully before evaluating any option.
2. Eliminate options that are NOT supported by the context.
3. Select the option with the STRONGEST and most DIRECT evidence in the context.
4. If the context is insufficient, select the most plausible option based on what IS mentioned.
5. Respond with ONLY the option letter. No explanation, no punctuation, no extra text.

*** Retrieved Medical Context ***
{chunks}

*** Question ***
{question}

*** Answer Choices ***
{options_text}

*** Instructions ***
Based solely on the context above, which single letter ({valid_letters}) is correct?

Answer:"""
)

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    max_tokens=2048,
    temperature=0.0,       
    reasoning_effort="low",    
    model_kwargs={"include_reasoning": False},  
)

chain = option_question_answer_prompt | llm | StrOutputParser()

with open(CHUNKS_JSON) as f:
    medqa_data = json.load(f)
print(f"Loaded {len(medqa_data)} questions")

already_done = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for entry in json.load(f):
            already_done[entry["index"]] = entry
    print(f"Resuming - {len(already_done)} already done")

def build_invoke_inputs(context, question, options):
    options_text = "\n".join([f"{k}) {v}" for k, v in options.items()])
    valid_letters = "/".join(options.keys())
    
    return {
        "chunks": context,
        "question": question,
        "options_text": options_text,
        "valid_letters": valid_letters,
    }

def extract_answer(raw: str, valid_letters: list[str]) -> str:
    cleaned = raw.strip().upper()
    
    for letter in valid_letters:
        if letter in cleaned:
            return letter

    try:
        repaired = repair_json(raw)
        parsed = json.loads(repaired)
        
        if isinstance(parsed, dict):
            for val in parsed.values():
                candidate = str(val).strip().upper()
                for letter in valid_letters:
                    if letter in candidate:
                        return letter
                        
        if isinstance(parsed, str):
            candidate = parsed.strip().upper()
            for letter in valid_letters:
                if letter in candidate:
                    return letter
    except Exception:
        pass
        
    return "X"

def process_item(item):
    idx = item["index"]
    if idx in already_done:
        return already_done[idx]

    context = "\n\n".join(
        item["chunks"].get(str(i), {}).get("content", "")
        for i in range(1, int(USE_CHUNK) + 1)
        if item["chunks"].get(str(i), {}).get("content")
    )

    inputs = build_invoke_inputs(context, item["question"], item["options"])
    raw_answer = chain.invoke(inputs).strip().upper()

    gold = item["answer_idx"].strip().upper()
    predicted = extract_answer(raw_answer, list(item["options"].keys()))

    chunk_source = "\n----\n ".join(
        item["chunks"].get(str(i), {}).get("source", "")
        for i in range(1, int(USE_CHUNK) + 1)
        if item["chunks"].get(str(i), {}).get("source")
    )

    return {
        "index": idx,
        "question": item["question"],
        "options": item["options"],
        "gold": gold,
        "predicted": predicted,
        "correct": predicted == gold,
        "answer": item["answer"],
        "chunk_used": USE_CHUNK,
        "chunk_source": chunk_source,
        "chunk_content": context,
    }

def main():
    results = already_done.copy()

    for item in tqdm(medqa_data, desc=f"Evaluating chunk {USE_CHUNK}"):
        result = process_item(item)
        results[result["index"]] = result

        if result["index"] not in already_done:
            with open(OUTPUT_FILE, "w") as f:
                sorted_results = [results[i] for i in sorted(results.keys())]
                json.dump(sorted_results, f, indent=2)

    correct = sum(1 for r in results.values() if r["correct"])
    total = len(medqa_data)
    accuracy = correct / total * 100
    
    print(f"\nFinal Accuracy: {correct}/{total} = {accuracy:.2f}%")

    with open(SUMMARY_FILE, "w") as f:
        json.dump({
            "model": GROQ_MODEL,
            "chunk_used": USE_CHUNK,
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total": total,
        }, f, indent=2)

    print(f"Results saved {OUTPUT_FILE}")
    print(f"Summary saved {SUMMARY_FILE}")

if __name__ == "__main__":
    main()