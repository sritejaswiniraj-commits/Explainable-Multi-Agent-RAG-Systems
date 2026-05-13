import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

CHUNK_DIR = "/home/exgr5/HCNLP/fixed_chunk_results_150"
BATCH_SIZE = 500  
DATA_DIR = "/home/exgr5/HCNLP/chroma_db_fixed_size_150"

chunk_files = [f for f in os.listdir(CHUNK_DIR) if f.endswith(".json")]
if not chunk_files:
    raise FileNotFoundError(f"No chunk files found in: {CHUNK_DIR}")

embeddings_model = HuggingFaceEmbeddings(
    model_name="abhinand/MedEmbed-large-v0.1",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
)

vectorstore = None

for chunk_file in tqdm(chunk_files, desc="Storing chunks in ChromaDB"):
    file_path = os.path.join(CHUNK_DIR, chunk_file)
    book_name = "_".join(os.path.basename(file_path).replace("chunks.json", "").split("_"))

    with open(file_path, "r") as f:
        data = json.load(f)
    chunks = [item["text"] for item in data]

    metadatas = [{"source": book_name, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{book_name}_chunk_{i}" for i in range(len(chunks))]

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_texts= chunks[i : i + BATCH_SIZE]
        batch_metadatas= metadatas[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]

        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                texts=batch_texts,
                embedding=embeddings_model,
                persist_directory=DATA_DIR,
                collection_name="medical_books",
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
        else:
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

    print(f" Stored {len(chunks)} chunks | source: '{book_name}'")

print(f" Total chunks in DB: {vectorstore._collection.count()}")