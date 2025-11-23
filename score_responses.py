import polars as pl
from metrics import Metric, RougeL, BLEU, BertScore
import json
from tqdm import tqdm

def main():
    RESULTS_PATH = "results/rag_results.jsonl"
    METRICS_PATH = "results/metric_scores.json"
    metrics: list[Metric] = [RougeL(), BLEU(), BertScore()]
    
    df: pl.DataFrame = pl.read_ndjson(RESULTS_PATH)
    results = {"item_index": df["item_index"].to_list()}
    for metric in tqdm(metrics):
        results |= metric.score_many(df["gold_answer"], df["model_answer"])
    
    #print("results:", results)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(results))

if __name__ == "__main__":
    main()
