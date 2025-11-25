import polars as pl
from rouge_score import rouge_scorer
from nltk.translate import bleu_score
from bert_score import score as score_bert
from collections import defaultdict

class Metric:
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    def score_many(self, references: pl.Series, candidates: pl.Series) -> dict[str, list[float]]:
        raise NotImplementedError("Please implement this method")

class RougeL(Metric):
    def __init__(self, use_stemmer: bool = False):
        super().__init__('rougeL')
        self.scorer = rouge_scorer.RougeScorer([self.name], use_stemmer=use_stemmer)

    def score_many(self, references: pl.Series, candidates: pl.Series) -> dict[str, list[float]]:
        result = defaultdict(list)
        for refs, cand in zip(references, candidates):
            score = self.scorer.score_multi(refs, cand)[self.name]
            result[self.name + "_precision"].append(score.precision)
            result[self.name + "_recall"].append(score.recall)
        
        return result

class BLEU(Metric):
    def __init__(self):
        super().__init__('bleu')

    def score_many(self, references: pl.Series, candidates: pl.Series) -> dict[str, list[float]]:
        result = defaultdict(list)
        for refs, cand in zip(references, candidates):
            score = bleu_score.sentence_bleu(refs, cand)
            result[self.name].append(score)
        
        return result

class BertScore(Metric):
    def __init__(self):
        super().__init__('bert_score')

    def score_many(self, references: pl.Series, candidates: pl.Series) -> dict[str, list[float]]:
        scores = score_bert(candidates.to_list(), references.to_list(), 
                            rescale_with_baseline=True, lang='en', use_fast_tokenizer=True)
        result = {
            self.name + "_precision": scores[0].tolist(),
            self.name + "_recall": scores[1].tolist()
        }
        
        return result
