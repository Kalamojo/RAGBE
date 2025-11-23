# Around 30-40 secs to execute 20 prompts on google colab with cpu
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from google.colab import ai
#from google import genai
import time
import os
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# ---- CONFIG (you already had this, keep or adjust) ----

DATASET_PATH = "data/input_data.json"      # your dataset file
OUTPUT_PATH = "results/rag_results.jsonl"  # where to save generations

MODEL_NAME = "google/gemini-2.5-flash"  # choose any model from ai.list_models()
#MODEL_NAME = "gemini-2.5-flash"
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
    # add more
}

#client = genai.Client(api_key=gemini_api_key)

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


def build_context(item: Dict[str, Any], shuffle: bool = True):
    """
    Combine relevant + irrelevant docs into one context string.
    Also keep labels so you can analyze later.
    """
    docs = []

    for d in item["relevant_docs"]:
        docs.append({"text": d, "label": "relevant"})
    for d in item["irrelevant_docs"]:
        docs.append({"text": d, "label": "irrelevant"})

    if shuffle:
        random.shuffle(docs)

    context_str = "\n\n".join(d["text"] for d in docs)
    return context_str, docs


def build_system_prompt(question: str, context: str, template: str) -> str:
    return template.format(question=question, context=context)


# Cant have a single model hard-coded
def call_model(prompt: str, model_name: str = MODEL_NAME) -> str:
    """
    Single-call wrapper around google.colab.ai.generate_text.
    """
    # ai.generate_text returns a string already
    return ai.generate_text(prompt, model_name=model_name).strip()
    # return client.models.generate_content(
    #     model=model_name, contents=prompt
    # ).text.strip()


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

        # Build combined context (RAG)
        context_str, docs_metadata = build_context(item, shuffle=True)

        for variant_name, template in PROMPT_VARIANTS.items():
            system_prompt = build_system_prompt(question, context_str, template)

            try:
                # model name needs to be variable, as well as potentially model instance
                model_answer = call_model(system_prompt, MODEL_NAME)
                time.sleep(10)
            except Exception as e:
                print(f"[ERROR] item {idx}, variant '{variant_name}': {e}")
                model_answer = f"[ERROR] {e}"

            record = {
                "item_index": idx,
                "prompt_variant": variant_name,
                "question": question,
                "gold_answer": gold_answer,
                "context": context_str,
                "documents": docs_metadata,   # each has text + label
                "system_prompt": system_prompt,
                "model_answer": model_answer,
            }
            results.append(record)

            print(f"[{idx:03d}][{variant_name}] Q: {question}")
            print(f" -> {model_answer}\n")

    save_jsonl(results, OUTPUT_PATH)
    print(f"\nSaved {len(results)} generations to {OUTPUT_PATH}")


# Actually run it
run_rag_evaluation()