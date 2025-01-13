import pandas as pd
import numpy as np

import baseline_script
from diversity_rerank import DiversityRerank
from early_fusion_irsystem import EarlyFusionIrSystem
from late_fusion_irsystem import LateFusionIRSystem
from preprocess import load_data_and_preprocess
from audio_irsystem import AudioIRSystem
from text_irsystem import TextIRSystem
from baseline_script import preprocess, Track, FeatureType


def load_filtered_tracks():
    tracks = load_data_and_preprocess()
    ids_df = pd.read_csv('dataset/id_clap_mmsr.tsv', sep='\t')
    valid_ids = set(ids_df['id'])
    filtered_tracks = [track for track in tracks if track.track_id in valid_ids]
    return filtered_tracks

def test_retrieval_systems():
    tracks = load_filtered_tracks()
    
    # Initialize IR systems
    print("\nInitializing IR systems...")
    
    # Audio systems
    audio_ir_spectral = AudioIRSystem(tracks, feature_type='spectral')
    audio_ir_musicnn = AudioIRSystem(tracks, feature_type='musicnn')
    audio_ir_clap = AudioIRSystem(tracks, feature_type='clap')

    # Text systems
    text_ir_bert = TextIRSystem(tracks, feature_type='bert')
    text_ir_clap_tags = TextIRSystem(tracks, feature_type='clap_tags')

    # diversification system
    early_fusion_ir_musicnn = EarlyFusionIrSystem(tracks, featureSet1=FeatureType.BERT, featureSet2=FeatureType.MUSICNN, n_dims=100)
    early_fusion_ir_clap = EarlyFusionIrSystem(tracks, featureSet1=FeatureType.CLAP_TAGS, featureSet2=FeatureType.CLAP, n_dims=100)
    
    # Test retrieval with a sample query
    print("\nTesting retrieval systems...")
    query_track = tracks[0]
    n = 10
    
    print(f"\nQuery track: {query_track}")
    
    # Test audio-based systems
    print("\nAudio-based Retrieval Results:")
    print("\n1. Spectral Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*audio_ir_spectral.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")
        
    print("\n2. MusicNN Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*audio_ir_musicnn.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")
            
    print("\n3. CLAP Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*audio_ir_clap.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

    # Test text-based systems
    print("\nText-based Retrieval Results:")
    print("\n1. BERT Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*text_ir_bert.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")
    
    print("\n2. CLAP tags Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*text_ir_clap_tags.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

    print("\Early fusion results MusicNN + BERT:")
    print("-" * 50)
    for (track, probability) in zip(*early_fusion_ir_musicnn.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

    print("\Early fusion results CLAP + CLAP tags:")
    print("-" * 50)
    for (track, probability) in zip(*early_fusion_ir_clap.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

if __name__ == "__main__":
    test_retrieval_systems()
