import json
import os
from langchain_text_splitters import CharacterTextSplitter

Directory = "/home/exgr5/HCNLP/data_clean/data_clean/textbooks/en"
DATA_FILES = sorted([os.path.join(Directory, f) for f in os.listdir(Directory) if f.endswith(".txt")])
OUTPUT_DIR = "/home/exgr5/HCNLP/fixed_chunk_results_150"

def chunk_file(data_file, chunk_size= 150, chunk_overlap= 0):
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,          
        chunk_overlap=chunk_overlap,
        length_function=lambda t: len(t.split()),  
        is_separator_regex=False,
    )

    raw_chunks = splitter.split_text(text)

    return [
        {
            "index": i,
            "word_count": len(chunk.split()),
            "text": chunk,
        }
        for i, chunk in enumerate(raw_chunks)
    ]

def save_chunks(chunks: list[dict], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks → {output_file}")

for data_file in DATA_FILES:
    print(f"Processing: {data_file}")
    chunks = chunk_file(data_file, chunk_size=150, chunk_overlap=0)
    out_path = os.path.join(
        OUTPUT_DIR,
        os.path.basename(data_file).replace(".txt", "_chunks.json")
    )
    save_chunks(chunks, out_path)