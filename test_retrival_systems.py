import pandas as pd
import numpy as np

from early_fusion_irsystem import FeatureType, EarlyFusionIrSystem
from text_irsystem import TextIRSystem
from audio_irsystem import AudioIRSystem
from visual_irsystem import VisualIRSystem
from baseline_script import preprocess, Track

def test_retrieval_systems():
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
    early_fusion_ir = EarlyFusionIrSystem(tracks, featureSet1=FeatureType.TFIDF, featureSet2=FeatureType.VGG19, n_dims=100)
    
    # Test retrieval with a sample query
    print("\nTesting retrieval systems...")
    query_track = tracks[0]
    n = 5
    
    print(f"\nQuery track: {query_track}")
    
    # Test text-based systems
    print("\nText-based Retrieval Results:")
    print("\n1. TF-IDF Based Similar Tracks:")
    print("-" * 50)
    for track in text_ir_tfidf.query(query_track, n=n):
        print(track)
    
    print("\n2. BERT Based Similar Tracks:")
    print("-" * 50)
    for track in text_ir_bert.query(query_track, n=n):
        print(track)
    
    # Test audio-based systems
    print("\nAudio-based Retrieval Results:")
    print("\n1. Spectral Based Similar Tracks:")
    print("-" * 50)
    for track in audio_ir_spectral.query(query_track, n=n):
        print(track)
        
    print("\n2. MusicNN Based Similar Tracks:")
    print("-" * 50)
    for track in audio_ir_musicnn.query(query_track, n=n):
        print(track)
    
    # Test visual-based systems
    print("\nVisual-based Retrieval Results:")
    print("\n1. ResNet Based Similar Tracks:")
    print("-" * 50)
    for track in visual_ir_resnet.query(query_track, n=n):
        print(track)
        
    print("\n2. VGG19 Based Similar Tracks:")
    print("-" * 50)
    for track in visual_ir_vgg.query(query_track, n=n):
        print(track)

    # Test early fusion system
    print("\nEarly Fusion Retrieval Results:")
    print("-" * 50)
    for track in early_fusion_ir.query(query_track, n=n):
        print(track)

if __name__ == "__main__":
    test_retrieval_systems()