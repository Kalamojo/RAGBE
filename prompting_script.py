from ollama import chat
from ollama import ChatResponse
from tqdm.contrib.itertools import product
from joblib import Parallel, delayed
#import time
import json
import random
from pathlib import Path
import dill as pickle

def load_dataset(path: str) -> list[dict[str, any]]:
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

def prompt_combos(prompts_dict: dict[str, dict[str, str]]):
    combos = []
    for prompt in prompts_dict.keys():
        combo = [
            {"name": prompt + "_base"} | prompts_dict[prompt],
            {"name": prompt + "_system"} | {"system": "\n".join([
                    prompts_dict[prompt]["system"], 
                    prompts_dict[prompt]["user"]
                ])},
            {"name": prompt + "_user"} | {"user": "\n".join([
                    prompts_dict[prompt]["system"], 
                    prompts_dict[prompt]["user"]
                ])}
        ]
        combos += combo
    
    return combos

def build_context(
    item: dict[str, any],
    condition: str = "relevant_only",
    shuffle: bool = False
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

def build_prompt_messages(question: str, context: str, template: dict[str, str]) -> list[dict[str, str]]:
    if 'user' not in template:
        messages = [{
                "role": "system",
                "content": template["system"].format(question=question, context=context)
            }]
    elif 'system' not in template:
        messages = [{
                "role": "user",
                "content": template["user"].format(question=question, context=context)
            }]
    else:
        messages = [
                {
                    "role": "system",
                    "content": template["system"]
                },
                {
                    "role": "user",
                    "content": template["user"].format(question=question, context=context)
                }
            ]
    return messages

def call_model(prompt_messages: list[dict[str, str]], is_gemini: bool, model_name: str, answer_start: str | None = None) -> str:
    """
    Single-call wrapper around google.colab.ai.generate_text.
    """
    if is_gemini:
        from google.colab import ai

        prompt = "\n".join([message["content"] for message in prompt_messages])
        answer = ai.generate_text(prompt, model_name=model_name).strip()
        # return client.models.generate_content(
        #     model=model_name, contents=prompt
        # ).text.strip()
    else:
        response: ChatResponse = chat(model=model_name, messages=prompt_messages)
        answer = response.message.content
    
    if answer_start is not None:
        start_ind = answer.find(answer_start)
        if start_ind == -1:
            return answer
        return answer[start_ind + len(answer_start)]

    return answer

def save_jsonl(records: list[dict[str, any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def rag_evaluation(dataset_item: tuple[int, dict[str, any]], context_condition: str, model: str, prompt_combo: dict[str, str]) -> dict[str, any]:
    idx, item = dataset_item
    question = item["question"]
    gold_answer = item.get("gold_answer", "")

    context_str, docs_metadata = build_context(
        item,
        condition=context_condition,
        shuffle=(context_condition == "mixed"),
    )

    prompt_messages = build_prompt_messages(question, context_str, prompt_combo)
    is_gemini = (model.startswith("gemini"))
    answer_start = "FINAL ANSWER:" if prompt_combo["name"].startswith("mastra_cot") else None
    try:
        model_answer = call_model(prompt_messages, is_gemini, model, answer_start)
        #time.sleep(10)
    except Exception as e:
        print(f"[ERROR] item {idx}, variant '{prompt_combo["name"]}', "
                f"context '{context_condition}': {e}")
        model_answer = f"[ERROR] {e}"
    
    record = {
        "item_index": idx,
        "model_name": model,
        "prompt_variant": prompt_combo["name"],
        "context_condition": context_condition,  # <--- key for analysis
        "question": question,
        "gold_answer": gold_answer,
        "context": context_str,
        "documents": docs_metadata,   # each has text + label
        "system_prompt": prompt_combo.get("system", None),
        "user_prompt": prompt_combo.get("user", None),
        "model_answer": model_answer,
    }

    print(
        f"[{idx:03d}][{prompt_combo["name"]}][{context_condition}] Q: {question}")
    print(f" -> {model_answer}\n")

    return record

MODELS = [
    "llama2:7b",
    "llama3.2",
    "llama3.2:1b",
    "qwen3:4b"
]

CONTEXT_CONDITIONS = [
    "relevant_only",
    "irrelevant_only",
    "mixed"
]

PROMPT_VARIANTS = {
    "langchain": {
        "system": (
            "You are an assistant for question-answering tasks. "
            "If you don't know the answer, just say that you don't know. "
            "Use one sentence maximum and keep the answer concise.\n"
        ),
        "user": (
            "Use the following pieces of retrieved context to answer the question. "
            "Question: {question}\n"
            "Context: {context}\n"
            "Answer:"
        )
    },
    "llamaindex": {
        "system": (
            "If you don't know the answer, just say that you don't know. "
            "Use one sentence maximum and keep the answer concise.\n"
        ),
            "user": (
            "We have provided context information below."
            "---------------------"
            "{context}"
            "---------------------"
            "Given this information, please answer the question: {question}"
        )
    },
    "claude_rag": {
        "system": (
            "Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already. "
            "Answer the question now, and avoid providing preamble such as 'Here is the answer', etc. "
            "If you don't know the answer, just say that you don't know. "
            "Use one sentence maximum and keep the answer concise.\n"
        ),
        "user": (
            "You have been tasked with helping us to answer the following query: "
            "<query>"
            "{question}"
            "</query>"
            "You have access to the following documents which are meant to provide context as you answer the query:"
            "<documents>"
            "{context}"
            "</documents>"
        )
    },
    "mastra_cot": {
        "system": (
            "You are a helpful assistant that answers questions based on the provided context."
            "Follow these steps for each response:"

            "1. First, carefully analyze the retrieved context chunks and identify key information."
            "2. Break down your thinking process about how the retrieved information relates to the query."
            "3. Explain how you're connecting different pieces from the retrieved chunks."
            "4. Draw conclusions based only on the evidence in the retrieved context."
            "5. If the retrieved chunks don't contain enough information, explicitly state what's missing."

            "Format your response as:"
            "THOUGHT PROCESS:"
            "- Step 1: [Initial analysis of retrieved chunks]"
            "- Step 2: [Connections between chunks]"
            "- Step 3: [Reasoning based on chunks]"

            "FINAL ANSWER:"
            "[Your concise answer based on the retrieved context. Use one sentence maximum]"

            "Important: When asked to answer a question, please base your answer only on the context provided in the tool. "
            "If the context doesn't contain enough information to fully answer the question, please state that explicitly."
            "If you don't know the answer, just say that you don't know. "
            "Remember: Explain how you're using the retrieved information to reach your conclusions.\n"
        ),
        "user": (
            "Question: {question}\n"
            "Context: {context}\n"
        )
    }
}


def main():
    SEED = 42
    DATASET_PATH = "data/input_data.json"
    OUTPUT_PATH = f"results/rag_results.jsonl"

    random.seed(SEED)

    PROMPT_COMBOS = prompt_combos(PROMPT_VARIANTS)
    dataset = load_dataset(DATASET_PATH)

    results = Parallel(n_jobs=-1, prefer="threads")(delayed(rag_evaluation)(
            dataset_item, 
            condition, 
            model, 
            prompt_combo
        ) for dataset_item, condition, model, prompt_combo in product(
                list(enumerate(dataset)), 
                CONTEXT_CONDITIONS, 
                MODELS, 
                PROMPT_COMBOS
            ))
    
    save_jsonl(results, OUTPUT_PATH)
    print(f"\nSaved {len(results)} generations to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
