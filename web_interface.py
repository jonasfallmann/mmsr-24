from baseline_script import BaselineIRSystem, preprocess
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

# Load datasets
basic_information = pd.read_csv('dataset/id_information_mmsr.tsv', sep='\t')
youtube_urls = pd.read_csv('dataset/id_url_mmsr.tsv', sep='\t')

# Preprocess datasets to tracks objects
tracks = preprocess(basic_information, youtube_urls)

baseline_ir = BaselineIRSystem(tracks)

# Option in ui input
input_options = []
for track in tracks:
    input_options.append(track.track_name + " - " + track.artist)




# Stream lit
st.set_page_config(layout="wide")
st.title("Retrieval system")

option = st.selectbox(
    "Select a song",
    (input_options),
)
# Get track id for input title and artist
track_name, artist = re.split(r' - ', option, maxsplit=1)
for track in tracks:
    if track.track_name == track_name and track.artist == artist:
        query_track = track
        break
recommended_tracks = baseline_ir.query(query_track)

# Results section
st.header("Top 10 most similar songs")
mygrid = make_grid(11,5)
mygrid[0][0].write("Id")
mygrid[0][1].write("Title")
mygrid[0][2].write("Artist")
mygrid[0][3].write("Album")
mygrid[0][4].write("Video")

for i in range(len(recommended_tracks)):
    mygrid[i+1][0].write(recommended_tracks[i].track_id)
    mygrid[i+1][1].write(recommended_tracks[i].track_name)
    mygrid[i+1][2].write(recommended_tracks[i].artist)
    mygrid[i+1][3].write(recommended_tracks[i].album_name)
    mygrid[i+1][4].video(recommended_tracks[i].url)
    
