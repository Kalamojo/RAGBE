import polars as pl
from metrics import Metric, RougeL, BLEU, BertScore
import json
from tqdm import tqdm

def main():
    RESULTS_PATH = "results/rag_results.jsonl"
    METRICS_PATH = "results/metric_scores.json"
    metrics: list[Metric] = [RougeL(), BLEU(), BertScore()]
    
    df: pl.DataFrame = pl.read_ndjson(RESULTS_PATH)
    
    idk = ["I don't know", "The retrieved context does not contain information to answer the question"]
    df = df.with_columns(
        pl.when(
            pl.col("context_condition") == "irrelevant_only"
        ).then(idk).otherwise(pl.concat_list(pl.col("gold_answer"))).alias("reference_answers")
    )

    #results = {"item_index": df["item_index"].to_list()}
    metric_columns = []
    for metric in tqdm(metrics):
        #results |= metric.score_many(df["reference_answers"], df["model_answer"])
        metric_columns.append(pl.from_dict(metric.score_many(df["reference_answers"], df["model_answer"])))
    
    df = pl.concat([df] + metric_columns, how="horizontal")
    
    #print("results:", results)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(df.write_json())

if __name__ == "__main__":
    main()
