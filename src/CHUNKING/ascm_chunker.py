import json
from tqdm import tqdm
import numpy as np
import math
from langchain_huggingface import HuggingFaceEmbeddings
import os
import pysbd


class ASCMChunker:
    def __init__(self, data_file, embedding_model, batch_size=256, MAX_WORDS=400, MIN_WORDS=30,
                 lambda_=0.5, mu=0.5, alpha=0.5):
        self.data_file = data_file
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.MAX_WORDS = MAX_WORDS
        self.MIN_WORDS = MIN_WORDS
        self.chunks = []
        self.lambda_ = lambda_
        self.mu = mu
        self.alpha = alpha
        self.segmenter = pysbd.Segmenter(language="en", clean=True)
        self.pre_chunked_indices = set()
        self.main()

    def open_file(self):
        with open(self.data_file, "r") as f:
            text = ""
            for line in f:
                text += line
        return text

    def split_text(self, text):
        sentences = self.segmenter.segment(text)
        return sentences

    def split_sentences(self, text):
        normal_blocks = []
        large_sentences = []
        current_block = []

        for sent in text.split("\n"):
            if not sent.strip():
                continue
            if len(sent.strip().split()) > self.MAX_WORDS:
                if current_block:
                    normal_blocks.append(current_block)
                    current_block = []
                large_sentences.append(sent)
            else:
                current_block.append(sent)

        if current_block:
            normal_blocks.append(current_block)

        return normal_blocks, large_sentences

    def run_ascm(self, sentences, pbar=None):
        self.chunks = []
        self.pre_chunked_indices = set()
        embeddings = self.embed_sentences(sentences)
        if pbar:
            pbar.update(len(sentences))
        similarity, distances = self.similarity_cost(embeddings)
        _, chunks = self.ascm_chunking(sentences, similarity, distances)
        return chunks

    def run_ascm_on_large(self, large_sentences, pbar=None):
        large_chunks = []
        for sent in large_sentences:
            sub_sentences = self.split_text(sent)
            chunks = self.run_ascm(sub_sentences, pbar)
            large_chunks.extend(chunks)
        return large_chunks

    def embed_sentences(self, sentences):
        embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i: i + self.batch_size]
            embeddings.extend(self.embedding_model.embed_documents(batch))
        return embeddings

    def similarity_cost(self, embeddings):
        distances = []
        similarity = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            distances.append(float(1 - sim))
            similarity.append(float(sim))
        return similarity, distances

    def total_cost(self, similarity):
        return [self.alpha * s for s in similarity]

    def optimal_size(self, sentences):
        n = len(sentences)
        opt_size = []
        for i in range(n):
            count = 1
            total = len(sentences[i].strip().split())
            for j in range(i + 1, n):
                total += len(sentences[j].strip().split())
                if total <= self.MAX_WORDS:
                    count += 1
                else:
                    break
            opt_size.append(count)
        return opt_size

    def boundary_coherence_term(self, end_index, similarity):
        idx = end_index - 1
        if idx < 0 or idx >= len(similarity):
            return 0.0
        return 1.0 / (1.0 + similarity[idx])

    def coherence_penalty(self, start_index, end_index, distances):
        internal = distances[start_index: end_index - 1]
        if len(internal) == 0:
            return 0.0
        return np.mean(internal)

    def phi(self, start_index, end_index, opt_size, distances, similarity):
        boundary_term = self.boundary_coherence_term(end_index, similarity)

        chunk_len = end_index - start_index
        local_opt = max(1, opt_size[start_index])
        P_len = ((chunk_len - local_opt) ** 2) / local_opt

        P_coh = self.coherence_penalty(start_index, end_index, distances)

        return self.alpha * boundary_term + self.lambda_ * P_len + self.mu * P_coh

    def get_Lmax_Lmin(self, sentences, end_index):
        word_count = 0
        Lmax = 0
        Lmin = None
        for i in range(end_index - 1, -1, -1):
            sentence_words = len(sentences[i].split())
            if word_count + sentence_words > self.MAX_WORDS:
                break
            word_count += sentence_words
            Lmax += 1
            if Lmin is None and word_count >= self.MIN_WORDS:
                Lmin = Lmax
        Lmax = max(Lmax, 1)
        Lmin = Lmin if Lmin is not None else Lmax
        Lmin = min(Lmin, Lmax)
        return Lmax, Lmin

    def ascm_chunking(self, sentences, similarity, distances):
        opt_size = self.optimal_size(sentences)
        n = len(sentences)
        parent = [-1] * (n + 1)
        dp = [math.inf] * (n + 1)
        dp[0] = 0
        _ = self.total_cost(similarity)
        for end_index in range(1, n + 1):
            Lmax, Lmin = self.get_Lmax_Lmin(sentences, end_index)
            for start_index in range(max(0, end_index - Lmax), end_index - Lmin + 1):
                cost = dp[start_index] + self.phi(
                    start_index, end_index, opt_size, distances, similarity
                )
                if start_index == 0 or parent[start_index] != -1:
                    if cost < dp[end_index]:
                        dp[end_index] = cost
                        parent[end_index] = start_index

        segments = []
        j = n
        while j > 0:
            i = parent[j]
            if i < 0:
                break
            segments.append((i, j))
            j = i
        segments.reverse()

        for start, end in segments:
            if end <= len(sentences):
                self.chunks.append(" ".join(sentences[start:end]))
        return segments, self.chunks

    @staticmethod
    def save_chunks_to_json(chunks, output_file):
        data = []
        for i, chunk in enumerate(chunks):
            entry = {
                "index": i,
                "word_count": len(chunk.split()),
                "text": chunk
            }
            data.append(entry)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} chunks to {output_file}")

    def main(self):
        text = self.open_file()
        normal_blocks, large_sentences = self.split_sentences(text)

        normal_chunks = []

        total_sentences = sum(len(b) for b in normal_blocks) + sum(
            len(self.split_text(s)) for s in large_sentences
        )
        with tqdm(total=total_sentences, desc="Embedding", unit="sentence") as pbar:
            for block in normal_blocks:
                chunks = self.run_ascm(block, pbar)
                normal_chunks.extend(chunks)
            large_chunks = self.run_ascm_on_large(large_sentences, pbar)
            all_chunks = normal_chunks + large_chunks

        self.save_chunks_to_json(
            all_chunks,
            os.path.join(
                OUTPUT_DIR,
                os.path.basename(self.data_file).replace(".txt", "_chunks.json")
            )
        )
        return all_chunks


Directory = "/home/exgr5/HCNLP/data_clean/data_clean/textbooks/en"
DATA_FILES = [os.path.join(Directory, f) for f in os.listdir(Directory) if f.endswith(".txt")]
OUTPUT_DIR = "/home/exgr5/HCNLP/ascm_chunker_results_150_50"

embedding_model = HuggingFaceEmbeddings(
    model_name="abhinand/MedEmbed-large-v0.1",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
DATA_FILES = sorted(DATA_FILES)

for data_file in DATA_FILES:
    print(f"Processing file: {data_file}")
    chunker = ASCMChunker(
        data_file,
        embedding_model,
        batch_size=512,
        MAX_WORDS=150,
        MIN_WORDS=50,
        lambda_=0.3,
        mu=0.6,
        alpha=0.6,
    )