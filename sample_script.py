# Around 30-40 secs to execute 20 prompts on google colab with cpu
from google.colab import ai
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# ---- CONFIG ----

DATASET_PATH = "input_data.json"      # your dataset file
OUTPUT_PATH = "rag_results.jsonl"     # where to save generations

# choose any model from ai.list_models()
MODEL_NAME = "google/gemini-2.5-flash"
SEED = 42

SYSTEM_TEMPLATE = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use one sentence maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Answer:"
)

PROMPT_VARIANTS = {
    "baseline": SYSTEM_TEMPLATE,
    # later you can add:
    # "doc_only_strict": "...",
    # "cot": "...",
}

# NEW: which *document condition* you want to test
CONTEXT_CONDITIONS = [
    "relevant_only",
    "irrelevant_only",
    "mixed",
    # if you ever want the original behavior back, add:
    # "mixed",
]

# ---- UTILS ----


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Expects a JSON file of the form:
    {
      "items": [
        {
          "question": "...",
          "relevant_docs": [...],
          "irrelevant_docs": [...],
          "gold_answer": "..."
        },
        ...
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def build_context(
    item: Dict[str, Any],
    condition: str = "relevant_only",
    shuffle: bool = True,
):
    """
    Build a context string according to the condition:
    - 'relevant_only'   -> only relevant_docs
    - 'irrelevant_only' -> only irrelevant_docs
    - 'mixed'           -> both (original behavior, if you want it)

    Returns:
        context_str: str
        docs: list of {"text": ..., "label": "relevant"/"irrelevant"}
    """
    docs = []

    if condition in ("relevant_only", "mixed"):
        for d in item.get("relevant_docs", []):
            docs.append({"text": d, "label": "relevant"})

    if condition in ("irrelevant_only", "mixed"):
        for d in item.get("irrelevant_docs", []):
            docs.append({"text": d, "label": "irrelevant"})

    if shuffle:
        random.shuffle(docs)

    context_str = "\n\n".join(d["text"] for d in docs)
    return context_str, docs


def build_system_prompt(question: str, context: str, template: str) -> str:
    return template.format(question=question, context=context)


def call_model(prompt: str, model_name: str = MODEL_NAME) -> str:
    """
    Single-call wrapper around google.colab.ai.generate_text.
    """
    return ai.generate_text(prompt, model_name=model_name).strip()


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- MAIN EVAL LOOP ----

def run_rag_evaluation():
    random.seed(SEED)

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset file '{DATASET_PATH}' not found. "
            f"Upload it to Colab or change DATASET_PATH."
        )

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} items from {DATASET_PATH}")

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset):
        question = item["question"]
        gold_answer = item.get("gold_answer", "")

        for context_condition in CONTEXT_CONDITIONS:
            # Build context according to condition (relevant_only / irrelevant_only / mixed)
            context_str, docs_metadata = build_context(
                item,
                condition=context_condition,
                shuffle=False,
            )

            for variant_name, template in PROMPT_VARIANTS.items():
                system_prompt = build_system_prompt(
                    question, context_str, template)

                try:
                    model_answer = call_model(system_prompt, MODEL_NAME)
                except Exception as e:
                    print(f"[ERROR] item {idx}, variant '{variant_name}', "
                          f"context '{context_condition}': {e}")
                    model_answer = f"[ERROR] {e}"

                record = {
                    "item_index": idx,
                    "prompt_variant": variant_name,
                    "context_condition": context_condition,  # <--- key for analysis
                    "question": question,
                    "gold_answer": gold_answer,
                    "context": context_str,
                    "documents": docs_metadata,   # each has text + label
                    "system_prompt": system_prompt,
                    "model_answer": model_answer,
                }
                results.append(record)

                print(
                    f"[{idx:03d}][{variant_name}][{context_condition}] Q: {question}")
                print(f" -> {model_answer}\n")

    save_jsonl(results, OUTPUT_PATH)
    print(f"\nSaved {len(results)} generations to {OUTPUT_PATH}")


# Actually run it
run_rag_evaluation()
