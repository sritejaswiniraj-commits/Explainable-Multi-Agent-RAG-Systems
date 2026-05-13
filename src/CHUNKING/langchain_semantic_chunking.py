import glob
import json
import os
from typing import List, Tuple
from langchain_experimental.text_splitter import SemanticChunker, combine_sentences, calculate_cosine_distances
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

TEXT_DIR = "data_clean/data_clean/textbooks/en"
OUTPUT_DIR = "langchain_chunk_results"


class TqdmSemanticChunker(SemanticChunker):
    def _calculate_sentence_distances(
        self, single_sentences_list: List[str], file_label: str = ""
    ) -> Tuple[List[float], List[dict]]:
        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(_sentences, self.buffer_size)
        combined_texts = [s["combined_sentence"] for s in sentences]

        batch_size = getattr(self, "batch_size", 256)
        batches = [
            combined_texts[i : i + batch_size]
            for i in range(0, len(combined_texts), batch_size)
        ]

        desc = f"Embedding [{file_label}]" if file_label else "Embedding"
        all_embeddings = []
        for batch in tqdm(batches, desc=desc, unit="batch", dynamic_ncols=True):
            all_embeddings.extend(self.embeddings.embed_documents(batch))

        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = all_embeddings[i]

        return calculate_cosine_distances(sentences)

    def split_text_with_distances(
        self, text: str, file_label: str = ""
    ) -> Tuple[List[str], List[dict]]:
        single_sentences_list = self._get_single_sentences_list(text)
        if len(single_sentences_list) == 1:
            return single_sentences_list, []

        distances, sentences = self._calculate_sentence_distances(
            single_sentences_list, file_label=file_label
        )

        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
            breakpoint_array = distances
        else:
            (
                breakpoint_distance_threshold,
                breakpoint_array,
            ) = self._calculate_breakpoint_threshold(distances)

        indices_above_thresh = [
            i
            for i, x in enumerate(breakpoint_array)
            if x > breakpoint_distance_threshold
        ]

        chunks, start_index = [], 0
        for index in indices_above_thresh:
            group = sentences[start_index : index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            if self.min_chunk_size and len(combined_text) < self.min_chunk_size:
                continue
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            chunks.append(" ".join([d["sentence"] for d in sentences[start_index:]]))

        distance_records = [
            {
                "index": s["index"],
                "sentence": s["sentence"],
                "combined_sentence": s["combined_sentence"],
                "distance_to_next": s.get("distance_to_next"),
            }
            for s in sentences
        ]

        return chunks, distance_records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_files = glob.glob(os.path.join(TEXT_DIR, "*.txt"))

    embeddings_model = HuggingFaceEmbeddings(
        model_name="abhinand/MedEmbed-large-v0.1",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
    )

    chunker = TqdmSemanticChunker(
        embeddings=embeddings_model,
        buffer_size=2,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=0.5,
    )
    chunker.batch_size = 256

    for txt_path in txt_files:
        file_stem = os.path.splitext(os.path.basename(txt_path))[0]
        print(f"\nProcessing: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as fh:
            text = fh.read()

        chunks, distance_records = chunker.split_text_with_distances(
            text, file_label=file_stem
        )
        print(f"  -> {len(chunks)} chunks produced")

        chunks_path = os.path.join(OUTPUT_DIR, f"{file_stem}_chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {"chunk_index": i, "word_count": len(c.split()), "text": c}
                    for i, c in enumerate(chunks)
                ],
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  Chunks    -> {chunks_path}")

        distances_path = os.path.join(OUTPUT_DIR, f"{file_stem}_distances.json")
        with open(distances_path, "w", encoding="utf-8") as fh:
            json.dump(distance_records, fh, ensure_ascii=False, indent=2)
        print(f"  Distances -> {distances_path}")

    print(f"\nResults in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()