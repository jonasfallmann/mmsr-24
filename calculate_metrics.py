import pandas as pd
import numpy as np
from baseline_script import BaselineIRSystem, EvaluationProtocol, EvaluationMetric, preprocess
from text_irsystem import TextIRSystem
from tqdm import tqdm

class PrecisionAtK(EvaluationMetric):
    def __init__(self, k=10):
        self.k = k

    def evaluate(self, recommended_tracks, relevant_tracks):
        if not recommended_tracks:
            return 0
        recommended_set = set(recommended_tracks[:self.k])
        relevant_set = set(relevant_tracks)
        return len(recommended_set & relevant_set) / self.k

class RecallAtK(EvaluationMetric):
    def __init__(self, k=10):
        self.k = k

    def evaluate(self, recommended_tracks, relevant_tracks):
        if not relevant_tracks:
            return 0
        recommended_set = set(recommended_tracks[:self.k])
        relevant_set = set(relevant_tracks)
        return len(recommended_set & relevant_set) / len(relevant_set)

class NDCGAtK(EvaluationMetric):
    def __init__(self, k=10):
        self.k = k

    def evaluate(self, recommended_tracks, relevant_tracks):
        def dcg(recommended, relevant, k):
            return sum((1 / np.log2(idx + 2)) for idx, track in enumerate(recommended[:k]) if track in relevant)
        
        def idcg(relevant, k):
            return sum((1 / np.log2(idx + 2)) for idx in range(min(k, len(relevant))))
        
        dcg_value = dcg(recommended_tracks, relevant_tracks, self.k)
        idcg_value = idcg(relevant_tracks, self.k)
        return dcg_value / idcg_value if idcg_value > 0 else 0

class MRR(EvaluationMetric):
    def __init__(self):
        pass

    def evaluate(self, recommended_tracks, relevant_tracks):
        for idx, track in enumerate(recommended_tracks):
            if track in relevant_tracks:
                return 1 / (idx + 1)
        return 0

class MetricsEvaluation(EvaluationProtocol):
    def __init__(self, tracks):
        self.tracks = tracks

    def evaluate(self, ir_system, k=10):
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        mrr_scores = []

        for query_track in tqdm(self.tracks, desc="Evaluating IR system"):
            relevant_ids = [
                track.track_id
                for track in self.tracks
                if track.track_id != query_track.track_id and (
                    any(top_genre in track.top_genres for top_genre in query_track.top_genres)
                )
            ]
            
            recommended_tracks = ir_system.query(query_track, n=k)
            recommended_ids = [track.track_id for track in recommended_tracks]

            precision_metric = PrecisionAtK(k)
            recall_metric = RecallAtK(k)
            ndcg_metric = NDCGAtK(k)
            mrr_metric = MRR()

            precision = precision_metric.evaluate(recommended_ids, relevant_ids)
            recall = recall_metric.evaluate(recommended_ids, relevant_ids)
            ndcg = ndcg_metric.evaluate(recommended_ids, relevant_ids)
            mrr = mrr_metric.evaluate(recommended_ids, relevant_ids)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)

        results = {
            "Precision@10": np.mean(precision_scores),
            "Recall@10": np.mean(recall_scores),
            "NDCG@10": np.mean(ndcg_scores),
            "MRR": np.mean(mrr_scores),
        }
        return results
    
if __name__ == "__main__":
    basic_info_df = pd.read_csv("dataset/id_information_mmsr.tsv", sep='\t')
    youtube_urls_df = pd.read_csv("dataset/id_url_mmsr.tsv", sep='\t')
    tfidf_df = pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep='\t', index_col=0)
    genres_df = pd.read_csv("dataset/id_genres_mmsr.tsv", sep='\t', index_col=0)
    tags_df = pd.read_csv("dataset/id_tags_dict.tsv", sep='\t')

    tracks = preprocess(basic_info_df, youtube_urls_df, tfidf_df, genres_df, tags_df)
    baseline_ir = BaselineIRSystem(tracks)
    print("Baseline IR System")
    text_ir = TextIRSystem(tracks)
    print("Text IR System")
    evaluation_protocol = MetricsEvaluation(tracks)

    metrics_baseline = evaluation_protocol.evaluate(baseline_ir)
    metrics_text = evaluation_protocol.evaluate(text_ir)

    print("Baseline:")
    for metric, score in metrics_baseline.items():
        print(f"{metric}: {score:.4f}")
    print("\nText IR System:")
    for metric, score in metrics_text.items():
        print(f"{metric}: {score:.4f}")
