from baseline_script import BaselineIRSystem, preprocess, FeatureType
from text_irsystem import TextIRSystem
from audio_irsystem import AudioIRSystem
from visual_irsystem import VisualIRSystem
from text_irsystem import TextIRSystem
from early_fusion_irsystem import EarlyFusionIrSystem
from late_fusion_irsystem import LateFusionIRSystem
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
from metrics import PrecisionAtK, RecallAtK, NDCGAtK, MRR, Popularity, DiversityAtK

# web interface
st.set_page_config(layout="wide")
st.title("Retrieval system")

@st.cache_data
def load_data():
    basic_info_df = pd.read_csv("deployment_data/id_information_mmsr.tsv", sep='\t')
    youtube_urls_df = pd.read_csv("deployment_data/id_url_mmsr.tsv", sep='\t')
    genres_df = pd.read_csv("deployment_data/id_genres_mmsr.tsv", sep='\t')
    tags_df = pd.read_csv("deployment_data/id_tags_dict.tsv", sep='\t')
    tfidf_df = pd.read_csv("deployment_data/id_lyrics_tf-idf_mmsr.tsv", sep='\t', index_col=0)
    spotify_df = pd.read_csv('deployment_data/id_metadata_mmsr.tsv', sep='\t')
    lastfm_df = pd.read_csv('deployment_data/id_total_listens.tsv', sep='\t')
    return basic_info_df, youtube_urls_df, genres_df, tags_df, tfidf_df, spotify_df, lastfm_df

@st.cache_data
def preprocess_tracks():
    basic_info_df, youtube_urls_df, genres_df, tags_df, tfidf_df, spotify_df, lastfm_df = load_data()
    return preprocess(
        basic_info_df, 
        youtube_urls_df,
        tfidf_df,
        genres_df,
        tags_df,
        spotify_df,
        lastfm_df,
    )

@st.cache_data
def load_precomputed_similarities():
    with open("precomputed_similarities.pkl", "rb") as f:
        return pickle.load(f)
    
@st.cache_data
def load_precomputed_relevant_tracks():
    with open("precomputed_relevant_songs.pkl", "rb") as f:
        return pickle.load(f)

# Util for displaying results with youtube video
def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

@st.cache_data
def load_datasets():
    basic_info_df = pd.read_csv("deployment_data/id_information_mmsr.tsv", sep='\t')
    youtube_urls_df = pd.read_csv("deployment_data/id_url_mmsr.tsv", sep='\t')
    tfidf_df = pd.read_csv("deployment_data/id_lyrics_tf-idf_mmsr.tsv", sep='\t', index_col=0)
    genres_df = pd.read_csv("deployment_data/id_genres_mmsr.tsv", sep='\t')
    tags_df = pd.read_csv("deployment_data/id_tags_dict.tsv", sep='\t')
    spotify_df = pd.read_csv('deployment_data/id_metadata_mmsr.tsv', sep='\t')
    lastfm_df = pd.read_csv('deployment_data/id_total_listens.tsv', sep='\t')
    return basic_info_df, youtube_urls_df, tfidf_df, genres_df, tags_df, spotify_df, lastfm_df

basic_info_df, youtube_urls_df, tfidf_df, genres_df, tags_df, spotify_df, lastfm_df = load_datasets()

# Preprocess datasets to tracks objects
# Load and preprocess data
tracks = preprocess_tracks()
@st.cache_data
def load_ir_systems(_tracks):
    baseline_ir = BaselineIRSystem(_tracks)
    text_ir_tfidf = TextIRSystem(_tracks, feature_type='tfidf')
    text_ir_bert = TextIRSystem(_tracks, feature_type='bert')
    text_ir_clap = TextIRSystem(_tracks, feature_type='clap_text')
    audio_ir_spectral = AudioIRSystem(_tracks, feature_type='spectral')
    audio_ir_musicnn = AudioIRSystem(_tracks, feature_type='musicnn')
    visual_ir_resnet = VisualIRSystem(_tracks, feature_type='resnet')
    visual_ir_vgg = VisualIRSystem(_tracks, feature_type='vgg19')
    early_fusion_ir = EarlyFusionIrSystem(tracks, FeatureType.BERT, FeatureType.MUSICNN, n_dims=100).set_name("EarlyFusion-Bert-MusicNN")
    late_fusion_ir = LateFusionIRSystem(_tracks, [text_ir_bert, audio_ir_musicnn, visual_ir_resnet], [0.3, 0.3, 0.4]).set_name('LateFusion-Bert-MusicNN-ResNet')
    return baseline_ir, text_ir_tfidf, text_ir_bert, text_ir_clap, audio_ir_spectral, audio_ir_musicnn, visual_ir_resnet, visual_ir_vgg, early_fusion_ir, late_fusion_ir


# Precompute and store similarities
def precompute_similarities(ir_systems, tracks):
    similarities = {}
    total_systems = len(ir_systems)
    overall_progress_bar = st.progress(0)
    system_progress_bar = st.progress(0)
    status_text = st.text("Precomputing similarities, please wait...")
    total_tracks = len(tracks)
    
    for system_idx, (ir_system_name, ir_system) in enumerate(ir_systems.items()):
        status_text.text(f"Precomputing similarities for {ir_system_name}, please wait...")
        similarities[ir_system_name] = {}
        for idx, track in enumerate(tracks):
            recommended_tracks, _ = ir_system.query(track, n=100)
            similarities[ir_system_name][track.track_id] = [rec.track_id for rec in recommended_tracks]
            system_progress_bar.progress((idx + 1) / total_tracks)
        overall_progress_bar.progress((system_idx + 1) / total_systems)
        system_progress_bar.empty()  # Reset system progress bar for next system
    
    with open("precomputed_similarities.pkl", "wb") as f:
        pickle.dump(similarities, f)
        
    overall_progress_bar.empty()
    status_text.text("Precomputation complete.")
    status_text.empty()  # Remove status_text after done



if not os.path.exists("precomputed_similarities.pkl"):
    baseline_ir, text_ir_tfidf, text_ir_bert, text_ir_clap, audio_ir_spectral, audio_ir_musicnn, visual_ir_resnet, visual_ir_vgg, early_fusion_ir, late_fusion_ir = load_ir_systems(tracks)
    ir_systems = {
    "Baseline": baseline_ir,
    "Text-TF-IDF": text_ir_tfidf,
    "Text-BERT": text_ir_bert,
    "Text-CLAP": text_ir_clap,
    "Audio-Spectral": audio_ir_spectral,
    "Audio-MusicNN": audio_ir_musicnn,
    "Visual-ResNet": visual_ir_resnet,
    "Visual-VGG19": visual_ir_vgg,
    "EarlyFusion-Bert-MusicNN": early_fusion_ir,
    "LateFusion-Bert-MusicNN-ResNet": late_fusion_ir}
    precompute_similarities(ir_systems, tracks)

# Load precomputed similarities
precomputed_similarities = load_precomputed_similarities()

# Option in ui input
input_options = []
for track in tracks:
    input_options.append(track.track_name + " - " + track.artist)

option = st.selectbox(
    label = "Choose a song",
    options = (input_options),
    index = None,
    placeholder = "Choose a query track",
)

number_retrieved = st.slider("How many songs do you want to retrieve?", 1, 100, 10)


if option is not None:
    # Get track id for input title and artist
    track_name, artist = re.split(r' - ', option, maxsplit=1)
    for track in tracks:
        if track.track_name == track_name and track.artist == artist:
            query_track = track
            break
else:
    query_track = None


precision = PrecisionAtK(k=number_retrieved)
recall = RecallAtK(k=number_retrieved)
ndcg = NDCGAtK(k=number_retrieved)
mrr = MRR()
popularity = Popularity()
diversity = DiversityAtK(k=number_retrieved, max_tags=29, threshold=6)
systems = ["Baseline", "Text-TF-IDF", "Text-BERT", "Text-CLAP", "Audio-Spectral", "Audio-MusicNN", "Visual-ResNet", "Visual-VGG19", "EarlyFusion-Bert-MusicNN", "LateFusion-Bert-MusicNN-ResNet"]

def get_metrics(query_track, number_retrieved, query_relevant_tracks):
    grid_metrics = make_grid(len(systems)+1, 7)
    grid_metrics[0][0].write("Metrics")
    grid_metrics[0][1].write("Precision")
    grid_metrics[0][2].write("Recall")
    grid_metrics[0][3].write("nDCG")
    grid_metrics[0][4].write("MRR")
    grid_metrics[0][5].write("Popularity")
    grid_metrics[0][6].write("Diversity")
    for system in systems:
        metrics_recommended_track_ids = precomputed_similarities[system][query_track.track_id][:number_retrieved]
        metrics_recommended_tracks = [track for track in tracks if track.track_id in metrics_recommended_track_ids]
        p = round(precision.evaluate(metrics_recommended_track_ids, query_relevant_tracks),4)
        r = round(recall.evaluate(metrics_recommended_track_ids, query_relevant_tracks),4)
        n = round(ndcg.evaluate(metrics_recommended_track_ids, query_relevant_tracks),4)
        m = round(mrr.evaluate(metrics_recommended_track_ids, query_relevant_tracks),4)
        pop = round(popularity.evaluate(metrics_recommended_tracks, query_relevant_tracks),4)
        d = round(diversity.evaluate(metrics_recommended_tracks, query_relevant_tracks),4)
        metrics = {"Precision":p, "Recall":r, "nDCG":n, "MRR":m, "Popularity":pop, "Diversity":d}
        grid_metrics[systems.index(system)][0].write(system)
        for m in range(len(metrics)):
            grid_metrics[systems.index(system)][m+1].write(f"{list(metrics.values())[m]}")
    return grid_metrics

if query_track is not None:
    relevant_tracks = load_precomputed_relevant_tracks()
    relevant_tracks_query = relevant_tracks[query_track.track_id]
    top_genres_string = ", ".join(query_track.top_genres)   
    if len(query_track.top_genres)>1:
        st.text(f"The top genres of your query track are: {top_genres_string}")
    else:
        st.text(f"The top genre of your query track is: {top_genres_string}")
    with st.expander("Metrics for your chosen song at N across IR Systems"):
        get_metrics(query_track, number_retrieved, relevant_tracks_query)
    
        

ir_system = st.radio(
    "Select an IR system",
    systems,
    index=None,
    horizontal=True,
)

if query_track is not None and ir_system is not None:
    recommended_track_ids = precomputed_similarities[ir_system][query_track.track_id][:number_retrieved]
    recommended_tracks = [track for track in tracks if track.track_id in recommended_track_ids]
else:
    recommended_tracks = None

# Results section
if recommended_tracks is None:
    st.write("No results to show yet. Please choose a track for the query and an IR system to receive results") 
else: 
    # st.subheader("Evaluation metrics: ")
    # evaluation_protocol = MetricsEvaluation(tracks)
    # evaluation = evaluation_protocol.evaluate(text_ir_bert)
    # metrics_grid = make_grid(1, 4)
    # metrics_grid[0][0].write("Precision@10: {:.2f}".format(evaluation["Precision@10"]))
    # metrics_grid[0][1].write("Recall@10: {:.2f}".format(evaluation["Recall@10"]))
    # metrics_grid[0][2].write("NDCG@10: {:.2f}".format(evaluation["NDCG@10"]))
    # metrics_grid[0][3].write("MRR: {:.2f}".format(evaluation["MRR"]))

    st.header(f"Top {number_retrieved} most similar songs")
    mygrid = make_grid(number_retrieved+1, 6)
    mygrid[0][0].write("Id")
    mygrid[0][1].write("Title")
    mygrid[0][2].write("Artist")
    mygrid[0][3].write("Album")
    mygrid[0][4].write("Top genres")
    mygrid[0][5].write("Video")
    for i in range(len(recommended_tracks)):
        mygrid[i+1][0].write(recommended_tracks[i].track_id)
        mygrid[i+1][1].write(recommended_tracks[i].track_name)
        mygrid[i+1][2].write(recommended_tracks[i].artist)
        mygrid[i+1][3].write(recommended_tracks[i].album_name)
        mygrid[i+1][4].write(", ".join(recommended_tracks[i].top_genres))
        mygrid[i+1][5].video(recommended_tracks[i].url)
