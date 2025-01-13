import pandas as pd
import numpy as np

import baseline_script
from diversity_rerank import DiversityRerank
from early_fusion_irsystem import EarlyFusionIrSystem
from late_fusion_irsystem import LateFusionIRSystem
from preprocess import load_data_and_preprocess
from audio_irsystem import AudioIRSystem
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

    # diversification system
    # diversification_ir = DiversityRerank(tracks, audio_ir_musicnn, diversification=0.5, dissimilarity_feature=FeatureType.BERT)
    
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

    # # Test diversification system
    # print("\nDiversification Retrieval Results:")
    # print("-" * 50)
    # for (track, probability) in zip(*diversification_ir.query(query_track, n=n)):
    #     print(f"[{probability:.2f}] {track}")

if __name__ == "__main__":
    test_retrieval_systems()
