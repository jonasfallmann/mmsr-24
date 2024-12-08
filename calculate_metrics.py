import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from baseline_script import BaselineIRSystem, preprocess

class EvaluationMetric:
    @staticmethod
    def precision_at_k(recommended_tracks, relevant_tracks, k=10):
        recommended_set = set(recommended_tracks[:k])
        relevant_set = set(relevant_tracks)
        return len(recommended_set & relevant_set) / k

    @staticmethod
    def recall_at_k(recommended_tracks, relevant_tracks, k=10):
        recommended_set = set(recommended_tracks[:k])
        relevant_set = set(relevant_tracks)
        return len(recommended_set & relevant_set) / len(relevant_set) if len(relevant_set) > 0 else 0

    @staticmethod
    def ndcg_at_k(recommended_tracks, relevant_tracks, k=10):
        y_true = [1 if track in relevant_tracks else 0 for track in recommended_tracks]
        y_score = sorted(y_true, reverse=True)
        return ndcg_score([y_score], [y_true], k=k)

    @staticmethod
    def mrr(recommended_tracks, relevant_tracks):
        for i, track in enumerate(recommended_tracks):
            if track in relevant_tracks:
                return 1 / (i + 1)
        return 0.0

class EvaluationProtocol:
    def __init__(self, tracks):
        self.tracks = tracks

    def evaluate(self, ir_system, k=10):
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        mrr_scores = []
        for query_track in self.tracks:
            recommended_tracks = ir_system.query(query_track, n=k)
            relevant_tracks = [track for track in self.tracks if track.track_id != query_track.track_id and track.album_name == query_track.album_name]
            recommended_ids = [track.track_id for track in recommended_tracks]
            relevant_ids = [track.track_id for track in relevant_tracks]
            precision_scores.append(EvaluationMetric.precision_at_k(recommended_ids, relevant_ids, k))
            recall_scores.append(EvaluationMetric.recall_at_k(recommended_ids, relevant_ids, k))
            ndcg_scores.append(EvaluationMetric.ndcg_at_k(recommended_ids, relevant_ids, k))
            mrr_scores.append(EvaluationMetric.mrr(recommended_ids, relevant_ids))
        return {
            "Precision@10": np.mean(precision_scores),
            "Recall@10": np.mean(recall_scores),
            "NDCG@10": np.mean(ndcg_scores),
            "MRR": np.mean(mrr_scores),
        }
    
if __name__ == "__main__":
    basic_info_df = pd.read_csv("dataset/id_information_mmsr.tsv", sep='\t')
    youtube_urls_df = pd.read_csv("dataset/id_url_mmsr.tsv", sep='\t')
    tfidf_df = pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep='\t', index_col=0)

    tracks = preprocess(basic_info_df, youtube_urls_df, tfidf_df)
    baseline_ir = BaselineIRSystem(tracks)
    evaluation_protocol = EvaluationProtocol(tracks)
    
    metrics = evaluation_protocol.evaluate(baseline_ir)
    print("Baseline IR system metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")