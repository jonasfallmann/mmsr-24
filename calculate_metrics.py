from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

import baseline_script
from baseline_script import BaselineIRSystem, EvaluationProtocol, EvaluationMetric, preprocess
from early_fusion_irsystem import EarlyFusionIrSystem, FeatureType, ReducerType
from text_irsystem import TextIRSystem
from audio_irsystem import AudioIRSystem
from visual_irsystem import VisualIRSystem
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


class Popularity(EvaluationMetric):
    def __init_(self):
        pass

    def evaluate(self, recommended_tracks, _):
        return sum([track.popularity for track in recommended_tracks])/len(recommended_tracks)


class DiversityAtK(EvaluationMetric):
    def __init__(self, k=10, threshold = 50):
        self.k = k
        self.threshold = threshold

    def evaluate(self, recommended_tracks: list[baseline_script.Track], _):
        # considering all provided tags (excluding genre) per song, average number of unique tag occurences
        # among retrieved documents
        occurring_tags = []
        for track in recommended_tracks[:self.k]:
            # append tag names
            if track.tags is None:
                continue

            # ignore tags that are a genre
            occurring_tags.extend([tag.tag for tag in track.tags if tag.tag not in track.genres and tag.weight > self.threshold])
        tags_set = set(occurring_tags)
        return len(tags_set) / self.k


class MetricsEvaluation(EvaluationProtocol):
    def __init__(self, tracks):
        self.tracks = tracks

    def evaluate(self, ir_system, k=10):
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        mrr_scores = []
        popularity_scores = []
        diversity_scores = []

        for index, query_track in enumerate(self.tracks):
            # print progress every 10%
            if index % (len(self.tracks) // 10) == 0:
                print(f"[{ir_system.name}] Progress: {index/len(self.tracks)*100:.2f}%")

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
            popularity_metric = Popularity()
            diversity_metric = DiversityAtK(k)

            precision = precision_metric.evaluate(recommended_ids, relevant_ids)
            recall = recall_metric.evaluate(recommended_ids, relevant_ids)
            ndcg = ndcg_metric.evaluate(recommended_ids, relevant_ids)
            mrr = mrr_metric.evaluate(recommended_ids, relevant_ids)
            popularity = popularity_metric.evaluate(recommended_tracks, relevant_ids)
            diversity = diversity_metric.evaluate(recommended_tracks, relevant_ids)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            popularity_scores.append(popularity)
            diversity_scores.append(diversity)

        print(f"[{ir_system.name}] Progress: 100.00%")
        results = {
            "Precision@10": np.mean(precision_scores),
            "Recall@10": np.mean(recall_scores),
            "NDCG@10": np.mean(ndcg_scores),
            "MRR": np.mean(mrr_scores),
            "Popularity": np.mean(popularity_scores),
            "Diversity": np.mean(diversity_scores)
        }
        return results
    
if __name__ == "__main__":
    # Load all necessary data
    print("Loading datasets...")
    
    # Basic information
    basic_info_df = pd.read_csv("dataset/id_information_mmsr.tsv", sep='\t')
    youtube_urls_df = pd.read_csv("dataset/id_url_mmsr.tsv", sep='\t')
    genres_df = pd.read_csv("dataset/id_genres_mmsr.tsv", sep='\t')
    tags_df = pd.read_csv("dataset/id_tags_dict.tsv", sep='\t')
    spotify_df = pd.read_csv('dataset/id_metadata_mmsr.tsv', sep='\t')
    lastfm_df = pd.read_csv('dataset/id_total_listens.tsv', sep='\t')
    
    # Text features
    tfidf_df = pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep='\t', index_col=0)
    bert_df = pd.read_csv("dataset/id_lyrics_bert_mmsr.tsv", sep='\t', index_col=0)
    
    # Audio features
    spectral_df = pd.read_csv("dataset/id_blf_spectral_mmsr.tsv", sep='\t', index_col=0)
    musicnn_df = pd.read_csv("dataset/id_musicnn_mmsr.tsv", sep='\t', index_col=0)
    
    # Visual features
    resnet_df = pd.read_csv("dataset/id_resnet_mmsr.tsv", sep='\t', index_col=0)
    vgg19_df = pd.read_csv("dataset/id_vgg19_mmsr.tsv", sep='\t', index_col=0)

    # Preprocess tracks
    print("Preprocessing tracks...")
    tracks = preprocess(
        basic_info_df, 
        youtube_urls_df,
        tfidf_df,
        genres_df,
        tags_df,
        spotify_df,
        lastfm_df,
        bert_df,
        spectral_df,
        musicnn_df,
        resnet_df,
        vgg19_df
    )

    # Initialize all IR systems
    print("\nInitializing IR systems...")
    baseline_ir = BaselineIRSystem(tracks).set_name("Baseline")
    text_ir_tfidf = TextIRSystem(tracks, feature_type='tfidf').set_name("Text-TF-IDF")
    text_ir_bert = TextIRSystem(tracks, feature_type='bert').set_name("Text-BERT")
    audio_ir_spectral = AudioIRSystem(tracks, feature_type='spectral').set_name("Audio-Spectral")
    audio_ir_musicnn = AudioIRSystem(tracks, feature_type='musicnn').set_name("Audio-MusicNN")
    visual_ir_resnet = VisualIRSystem(tracks, feature_type='resnet').set_name("Visual-ResNet")
    visual_ir_vgg = VisualIRSystem(tracks, feature_type='vgg19').set_name("Visual-VGG19")
    # set to 100 dims as this is way faster to compute with marginal loss in performance
    early_fusion_irsystem = EarlyFusionIrSystem(tracks, FeatureType.BERT, FeatureType.MUSICNN, n_dims=100).set_name("Early Fusion BERT+MusicNN 100")

    # Initialize evaluation protocol
    evaluation_protocol = MetricsEvaluation(tracks)

    # Evaluate all systems
    print("\nEvaluating systems...")
    results = {}
    tasks = [
        ("Baseline", baseline_ir),
        ("Text-TF-IDF", text_ir_tfidf),
        ("Text-BERT", text_ir_bert),
        ("Audio-Spectral", audio_ir_spectral),
        ("Audio-MusicNN", audio_ir_musicnn),
        ("Visual-ResNet", visual_ir_resnet),
        ("Visual-VGG19", visual_ir_vgg),
        ("Early Fusion BERT+MusicNN 100", early_fusion_irsystem)
     ]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluation_protocol.evaluate, ir_system) for _, ir_system in tasks]
        for (system_name, _), future in zip(tasks, futures):
            results[system_name] = future.result()

    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for system_name, metrics in results.items():
        print(f"\n{system_name}:")
        print("-" * 30)
        for metric, score in metrics.items():
            print(f"{metric}: {score:.4f}")

    print("\nBest Performing Systems:")
    print("=" * 50)
    metrics_list = ["Precision@10", "Recall@10", "NDCG@10", "MRR"]
    for metric in metrics_list:
        best_system = max(results.items(), key=lambda x: x[1][metric])
        print(f"\nBest {metric}: {best_system[0]} ({best_system[1][metric]:.4f})")


def run_full_early_fusion_experiment(tasks):
    # add all combinations of early fusion systems with full dimensions and 100 dimensions
    for idx, feature1 in enumerate(FeatureType):
        for jdx, feature2 in enumerate(FeatureType):
            if idx >= jdx:
                continue
            early_fusion_irsystem = EarlyFusionIrSystem(tracks, feature1, feature2).set_name(
                f"Early Fusion {feature1.value}+{feature2.value} Full Dimension")
            early_fusion_irsystem_100 = EarlyFusionIrSystem(tracks, feature1, feature2, n_dims=100).set_name(
                f"Early Fusion {feature1.value}+{feature2.value} 100 Dimensions")
            tasks.append((early_fusion_irsystem.name, early_fusion_irsystem))
            tasks.append((early_fusion_irsystem_100.name, early_fusion_irsystem_100))


def run_diversification_experiment(tasks):
    #music cnn compared with different stages of diversification
    for diversification in [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]:
        system = AudioIRSystem(tracks, feature_type='musicnn', diversification=diversification, n_diverse=5).set_name(f"Audio-MusicNN-{diversification}")
        tasks.append((f"Audio-MusicNN-{diversification}", system))