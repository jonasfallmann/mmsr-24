import os
import csv
import requests
from msclap import CLAP
from tqdm import tqdm
import re

EMBED_LEN = 1024

def load_song_information(info_file):
    """Load song information into a dictionary."""
    song_info = {}
    with open(info_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip the header
        for row in reader:
            song_id, artist, title, _ = row
            song_info[song_id] = {'artist': artist, 'title': title}
    return song_info

def fetch_lyrics(artist, title):
    """Fetch lyrics for a given artist and title from the API."""
    url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        lyrics = data.get('lyrics', '').replace('\n', ' ').strip()  # Clean the lyrics
        lyrics = re.sub(r'[^a-zA-Z0-9,.!? ]+', '', lyrics)
        # print(lyrics)
        return lyrics if lyrics else None
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch lyrics for {artist} - {title}: {e}")
        return None

# Initialize the CLAP model
clap_model = CLAP(version='2023', use_cuda=False)

# Path to the song information file
info_file_path = "dataset/id_information_mmsr.tsv"

# Load song information
song_info = load_song_information(info_file_path)
print(f"Loaded song information for {len(song_info)} songs.")

# Write the text embeddings into a TSV file
output_file = "dataset/id_clap_lyrics_mmsr.tsv"
with open(output_file, 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    
    # Write the header
    header = ['id'] + [f'embedding_{i}' for i in range(EMBED_LEN)]
    writer.writerow(header)
    
    for song_id, info in tqdm(song_info.items(), total=len(song_info), desc="Embedding Lyrics"):
        artist = info['artist']
        title = info['title']
        
        # Fetch and clean the lyrics
        lyrics = fetch_lyrics(artist, title)
        if not lyrics:
            print(f"No lyrics found for {artist} - {title}. Skipping...")
            continue
        
        # Generate text embedding for the lyrics
        text_embedding = clap_model.get_text_embeddings([lyrics])[0]
        
        # Write to the file
        writer.writerow([song_id] + text_embedding.tolist())

print(f"Lyrics embeddings have been written to {output_file}")