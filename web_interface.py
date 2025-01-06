from baseline_script import BaselineIRSystem, preprocess
from text_irsystem import TextIRSystem
from audio_irsystem import AudioIRSystem
from visual_irsystem import VisualIRSystem
from text_irsystem import TextIRSystem
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# web interface
st.set_page_config(layout="wide")
st.title("Retrieval system")

@st.cache_data
def load_data():
    basic_info_df = pd.read_csv("dataset/id_information_mmsr.tsv", sep='\t')
    youtube_urls_df = pd.read_csv("dataset/id_url_mmsr.tsv", sep='\t')
    genres_df = pd.read_csv("dataset/id_genres_mmsr.tsv", sep='\t')
    tags_df = pd.read_csv("dataset/id_tags_dict.tsv", sep='\t')
    tfidf_df = pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep='\t', index_col=0)
    bert_df = pd.read_csv("dataset/id_lyrics_bert_mmsr.tsv", sep='\t', index_col=0)
    spectral_df = pd.read_csv("dataset/id_blf_spectral_mmsr.tsv", sep='\t', index_col=0)
    musicnn_df = pd.read_csv("dataset/id_musicnn_mmsr.tsv", sep='\t', index_col=0)
    resnet_df = pd.read_csv("dataset/id_resnet_mmsr.tsv", sep='\t', index_col=0)
    vgg19_df = pd.read_csv("dataset/id_vgg19_mmsr.tsv", sep='\t', index_col=0)
    return basic_info_df, youtube_urls_df, genres_df, tags_df, tfidf_df, bert_df, spectral_df, musicnn_df, resnet_df, vgg19_df

@st.cache_data
def preprocess_tracks():
    basic_info_df, youtube_urls_df, genres_df, tags_df, tfidf_df, bert_df, spectral_df, musicnn_df, resnet_df, vgg19_df = load_data()
    return preprocess(
        basic_info_df, 
        youtube_urls_df,
        tfidf_df,
        genres_df,
        tags_df,
        bert_df,
        spectral_df,
        musicnn_df,
        resnet_df,
        vgg19_df
    )

@st.cache_data
def load_precomputed_similarities():
    with open("precomputed_similarities.pkl", "rb") as f:
        return pickle.load(f)

# Util for displaying results with youtube video
def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

# Load and preprocess data
tracks = preprocess_tracks()

baseline_ir = BaselineIRSystem(tracks)
text_ir_tfidf = TextIRSystem(tracks, feature_type='tfidf')
text_ir_bert = TextIRSystem(tracks, feature_type='bert')
audio_ir_spectral = AudioIRSystem(tracks, feature_type='spectral')
audio_ir_musicnn = AudioIRSystem(tracks, feature_type='musicnn')
visual_ir_resnet = VisualIRSystem(tracks, feature_type='resnet')
visual_ir_vgg = VisualIRSystem(tracks, feature_type='vgg19')

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
            recommended_tracks = ir_system.query(track)
            similarities[ir_system_name][track.track_id] = [rec.track_id for rec in recommended_tracks]
            system_progress_bar.progress((idx + 1) / total_tracks)
        overall_progress_bar.progress((system_idx + 1) / total_systems)
        system_progress_bar.empty()  # Reset system progress bar for next system
    
    with open("precomputed_similarities.pkl", "wb") as f:
        pickle.dump(similarities, f)
    
    overall_progress_bar.empty()
    status_text.text("Precomputation complete.")
    status_text.empty()  # Remove status_text after done

ir_systems = {
    "Baseline": baseline_ir,
    "Text-TF-IDF": text_ir_tfidf,
    "Text-BERT": text_ir_bert,
    "Audio-Spectral": audio_ir_spectral,
    "Audio-MusicNN": audio_ir_musicnn,
    "Visual-ResNet": visual_ir_resnet,
    "Visual-VGG19": visual_ir_vgg
}

if not os.path.exists("precomputed_similarities.pkl"):
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

ir_system = st.radio(
    "Select an IR system",
    ["Baseline", "Text-TF-IDF", "Text-BERT", "Audio-Spectral", "Audio-MusicNN", "Visual-ResNet", "Visual-VGG19"],
    index=None,
    horizontal=True,
)

if option is not None:
    # Get track id for input title and artist
    track_name, artist = re.split(r' - ', option, maxsplit=1)
    for track in tracks:
        if track.track_name == track_name and track.artist == artist:
            query_track = track
            break
else:
    query_track = None

if query_track is not None and ir_system is not None:
    recommended_track_ids = precomputed_similarities[ir_system][query_track.track_id]
    recommended_tracks = [track for track in tracks if track.track_id in recommended_track_ids]
else:
    recommended_tracks = None

# Results section
if recommended_tracks is None:
    st.write("No results to show yet. Please choose a track for the query and an IR system to receive results") 
else: 
    st.subheader("Your query choice is: " + option)
    # st.subheader("Evaluation metrics: ")
    # evaluation_protocol = MetricsEvaluation(tracks)
    # evaluation = evaluation_protocol.evaluate(text_ir_bert)
    # metrics_grid = make_grid(1, 4)
    # metrics_grid[0][0].write("Precision@10: {:.2f}".format(evaluation["Precision@10"]))
    # metrics_grid[0][1].write("Recall@10: {:.2f}".format(evaluation["Recall@10"]))
    # metrics_grid[0][2].write("NDCG@10: {:.2f}".format(evaluation["NDCG@10"]))
    # metrics_grid[0][3].write("MRR: {:.2f}".format(evaluation["MRR"]))

    st.header("Top 10 most similar songs")
    mygrid = make_grid(11,6)
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
