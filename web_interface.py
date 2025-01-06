from baseline_script import BaselineIRSystem, preprocess
from text_irsystem import TextIRSystem
from audio_irsystem import AudioIRSystem
from visual_irsystem import VisualIRSystem
from text_irsystem import TextIRSystem
import streamlit as st
import pandas as pd
import numpy as np
import re

# Util for displaying results with youtube video
def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

# Basic information
basic_info_df = pd.read_csv("dataset/id_information_mmsr.tsv", sep='\t')
youtube_urls_df = pd.read_csv("dataset/id_url_mmsr.tsv", sep='\t')
genres_df = pd.read_csv("dataset/id_genres_mmsr.tsv", sep='\t')
tags_df = pd.read_csv("dataset/id_tags_dict.tsv", sep='\t')

# Text features
tfidf_df = pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep='\t', index_col=0)
bert_df = pd.read_csv("dataset/id_lyrics_bert_mmsr.tsv", sep='\t', index_col=0)

# Audio features
spectral_df = pd.read_csv("dataset/id_blf_spectral_mmsr.tsv", sep='\t', index_col=0)
musicnn_df = pd.read_csv("dataset/id_musicnn_mmsr.tsv", sep='\t', index_col=0)

# Visual features
resnet_df = pd.read_csv("dataset/id_resnet_mmsr.tsv", sep='\t', index_col=0)
vgg19_df = pd.read_csv("dataset/id_vgg19_mmsr.tsv", sep='\t', index_col=0)

# Preprocess datasets to tracks objects
tracks = preprocess(
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

baseline_ir = BaselineIRSystem(tracks)
text_ir_tfidf = TextIRSystem(tracks, feature_type='tfidf')
text_ir_bert = TextIRSystem(tracks, feature_type='bert')
audio_ir_spectral = AudioIRSystem(tracks, feature_type='spectral')
audio_ir_musicnn = AudioIRSystem(tracks, feature_type='musicnn')
visual_ir_resnet = VisualIRSystem(tracks, feature_type='resnet')
visual_ir_vgg = VisualIRSystem(tracks, feature_type='vgg19')

# Option in ui input
input_options = []
for track in tracks:
    input_options.append(track.track_name + " - " + track.artist)

# web interface
st.set_page_config(layout="wide")
st.title("Retrieval system")

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
st.write("You selected:", ir_system)

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
    if ir_system == "Baseline":
        recommended_tracks = baseline_ir.query(query_track)
    elif ir_system == "Text-TF-IDF":
        recommended_tracks = text_ir_tfidf.query(query_track)
    elif ir_system == "Text-BERT":
        recommended_tracks = text_ir_bert.query(query_track)
    elif ir_system == "Audio-Spectral":
        recommended_tracks = audio_ir_spectral.query(query_track)
    elif ir_system == "Audio-MusicNN":
        recommended_tracks = audio_ir_musicnn.query(query_track)
    elif ir_system == "Visual-ResNet":
        recommended_tracks = visual_ir_resnet.query(query_track)
    elif ir_system == "Visual-VGG19":
        recommended_tracks = visual_ir_vgg.query(query_track)
else:
    recommended_tracks = None


# Results section
if recommended_tracks is None:
    st.write("No results to show yet. Please choose a track for the query and an IR system to receive results") 
else: 
    st.header("Top 10 most similar songs")
    st.subheader("Your query choice is: " + option)
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
    