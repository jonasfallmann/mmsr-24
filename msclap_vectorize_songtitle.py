import os
import csv
from msclap import CLAP
from tqdm import tqdm

EMBED_LEN = 1024

def load_song_information(info_file):
    """Load song information into a dictionary mapping IDs to 'artist - song'."""
    song_info = {}
    with open(info_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip the header
        for row in reader:
            song_id = row[0]
            artist = row[1]
            song = row[2]
            song_info[song_id] = f"{artist} - {song}"
    return song_info

# Initialize the CLAP model
clap_model = CLAP(version='2023', use_cuda=False)

# Path to the song information file
info_file_path = "dataset/id_information_mmsr.tsv"

# Load the song information
song_information = load_song_information(info_file_path)

# Write the text embeddings into a TSV file
output_file = "dataset/id_clap_songtitles_mmsr.tsv"
with open(output_file, 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    
    # Write the header
    header = ['id'] + [f'embedding_{i}' for i in range(EMBED_LEN)]
    writer.writerow(header)
    
    for song_id, song_title in tqdm(song_information.items(), total=len(song_information), desc="Embedding Songs"):
        # Generate text embedding for song title
        text_embedding = clap_model.get_text_embeddings([song_title])[0]
        
        # Write to the file
        writer.writerow([song_id] + text_embedding.tolist())

print(f"Song title embeddings have been written to {output_file}")