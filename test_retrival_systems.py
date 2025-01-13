import pandas as pd
import numpy as np

import baseline_script
from diversity_rerank import DiversityRerank
from early_fusion_irsystem import EarlyFusionIrSystem
from late_fusion_irsystem import LateFusionIRSystem
from preprocess import load_data_and_preprocess
from text_irsystem import TextIRSystem
from audio_irsystem import AudioIRSystem
from visual_irsystem import VisualIRSystem
from baseline_script import preprocess, Track, FeatureType


def test_retrieval_systems():
    tracks = load_data_and_preprocess()
    
    # Initialize IR systems
    print("\nInitializing IR systems...")
    # Text systems
    text_ir_tfidf = TextIRSystem(tracks, feature_type='tfidf')
    text_ir_bert = TextIRSystem(tracks, feature_type='bert')
    
    # Audio systems
    audio_ir_spectral = AudioIRSystem(tracks, feature_type='spectral')
    audio_ir_musicnn = AudioIRSystem(tracks, feature_type='musicnn')
    
    # Visual systems
    visual_ir_resnet = VisualIRSystem(tracks, feature_type='resnet')
    visual_ir_vgg = VisualIRSystem(tracks, feature_type='vgg19')

    # early fusion system
    early_fusion_ir = EarlyFusionIrSystem(tracks, featureSet1=FeatureType.BERT, featureSet2=FeatureType.MUSICNN, n_dims=100)

    # late fusion system
    late_fusion_ir = LateFusionIRSystem(tracks, [text_ir_tfidf, audio_ir_spectral, visual_ir_resnet])

    # diversification system
    diversification_ir = DiversityRerank(tracks, audio_ir_musicnn, diversification=0.5, dissimilarity_feature=FeatureType.BERT)
    
    # Test retrieval with a sample query
    print("\nTesting retrieval systems...")
    query_track = tracks[0]
    n = 5
    
    print(f"\nQuery track: {query_track}")
    
    # Test text-based systems
    print("\nText-based Retrieval Results:")
    print("\n1. TF-IDF Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*text_ir_tfidf.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")
    
    print("\n2. BERT Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*text_ir_bert.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

    
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
    
    # Test visual-based systems
    print("\nVisual-based Retrieval Results:")
    print("\n1. ResNet Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*visual_ir_resnet.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")
        
    print("\n2. VGG19 Based Similar Tracks:")
    print("-" * 50)
    for (track, probability) in zip(*visual_ir_vgg.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

    # Test early fusion system
    print("\nEarly Fusion Retrieval Results:")
    print("-" * 50)
    for (track, probability) in zip(*early_fusion_ir.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

    # Test late fusion system
    print("\nLate Fusion Retrieval Results:")
    print("-" * 50)
    for (track, probability) in zip(*late_fusion_ir.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

    # Test diversification system
    print("\nDiversification Retrieval Results:")
    print("-" * 50)
    for (track, probability) in zip(*diversification_ir.query(query_track, n=n)):
        print(f"[{probability:.2f}] {track}")

if __name__ == "__main__":
    test_retrieval_systems()