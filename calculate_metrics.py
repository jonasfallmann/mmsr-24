from concurrent.futures import ThreadPoolExecutor

from audio_irsystem import AudioIRSystem
from baseline_script import BaselineIRSystem, FeatureType
from clap_irsystem import CombinedCLAPIRSystem
from diversity_rerank import DiversityRerank
from early_fusion_irsystem import EarlyFusionIrSystem
from late_fusion_irsystem import LateFusionIRSystem
from metrics import MetricsEvaluation
from preprocess import load_data_and_preprocess
from text_irsystem import TextIRSystem
from visual_irsystem import VisualIRSystem
import pandas as pd

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


def run_intrinsic_diversification_experiment(tasks):
    # music cnn compared with different stages of diversification
    for diversification in [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]:
        system = AudioIRSystem(tracks, feature_type='musicnn', diversification=diversification, n_diverse=5).set_name(
            f"Audio-MusicNN-{diversification}")
        tasks.append((f"Audio-MusicNN-{diversification}", system))


def run_diversification_rerank_experiment(tasks):
    # rerate with different dissimilarity features
    for dissimilarity_feature in [FeatureType.TFIDF]:
        for diversification in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            system = DiversityRerank(tracks, audio_ir_musicnn, diversification=diversification,
                                     dissimilarity_feature=dissimilarity_feature, pool_multiple=10).set_name(
                f"Audio-MusicNN-Diversification-{diversification}-{dissimilarity_feature}")
            tasks.append((f"Audio-MusicNN-Diversification-{diversification}-{dissimilarity_feature}", system))


def load_filtered_tracks():
    tracks = load_data_and_preprocess()
    ids_df = pd.read_csv('dataset/id_clap_audio_mmsr.tsv', sep='\t')
    valid_ids = set(ids_df['id'])
    filtered_tracks = [track for track in tracks if track.track_id in valid_ids]
    return filtered_tracks

if __name__ == "__main__":
    tracks = load_filtered_tracks()

    # Initialize all IR systems
    print("\nInitializing IR systems...")
    baseline_ir = BaselineIRSystem(tracks).set_name("Baseline")
    text_ir_tfidf = TextIRSystem(tracks, feature_type='tfidf').set_name("Text-TF-IDF")
    text_ir_bert = TextIRSystem(tracks, feature_type='bert').set_name("Text-BERT")
    text_ir_clap = TextIRSystem(tracks, feature_type='clap_text').set_name("Text-CLIP")
    audio_ir_spectral = AudioIRSystem(tracks, feature_type='spectral').set_name("Audio-Spectral")
    audio_ir_musicnn = AudioIRSystem(tracks, feature_type='musicnn').set_name("Audio-MusicNN")
    audio_ir_clap = AudioIRSystem(tracks, feature_type='clap_audio').set_name("Audio-CLAP")
    visual_ir_resnet = VisualIRSystem(tracks, feature_type='resnet').set_name("Visual-ResNet")
    visual_ir_vgg = VisualIRSystem(tracks, feature_type='vgg19').set_name("Visual-VGG19")
    # set to 100 dims as this is way faster to compute with marginal loss in performance
    early_fusion_irsystem = EarlyFusionIrSystem(tracks, FeatureType.BERT, FeatureType.MUSICNN, n_dims=100).set_name(
        "Early Fusion BERT+MusicNN 100")
    late_fusion_ir = LateFusionIRSystem(tracks, [text_ir_bert, audio_ir_musicnn, visual_ir_resnet],
                                        [0.3, 0.3, 0.4]).set_name('LateFusion-Bert-MusicNN-ResNet')
    late_fusion_clap_ir = LateFusionIRSystem(tracks, [text_ir_clap, audio_ir_clap],
                                        [0.7, 0.3]).set_name('LateFusion-CLAP')
    diversification_irsystem = DiversityRerank(tracks=tracks, ir_system=audio_ir_musicnn, diversification=0.5,
                                               dissimilarity_feature=FeatureType.TFIDF).set_name(
        "Audio-MusicNN-Diversification")
    clap_irsystem = CombinedCLAPIRSystem(tracks).set_name("EarlyFusion-Avg-CLAP")

    # Initialize evaluation protocol
    evaluation_protocol = MetricsEvaluation(tracks)

    # Evaluate all systems
    print("\nEvaluating systems...")
    results = {}
    tasks = [
        ("Baseline", baseline_ir),
        ("Text-TF-IDF", text_ir_tfidf),
        ("Text-BERT", text_ir_bert),
        ("Text-CLAP", text_ir_clap),
        ("Audio-Spectral", audio_ir_spectral),
        ("Audio-MusicNN", audio_ir_musicnn),
        ("Audio-CLAP", audio_ir_clap),
        ("Visual-ResNet", visual_ir_resnet),
        ("Visual-VGG19", visual_ir_vgg),
        ("Early Fusion BERT+MusicNN 100", early_fusion_irsystem),
        ("LateFusion-Bert-MusicNN-ResNet", late_fusion_ir),
        ("EarlyFusion-Avg-CLAP", clap_irsystem),
        ("LateFusion-CLAP", late_fusion_clap_ir),
    ]

    # run_diversification_rerate_experiment(tasks)

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
