import os
import csv
from laion_clap import CLAP_Module
from tqdm import tqdm

# Parameters
EMBED_LEN = 512  # Adjust based on your model's output dimension

# Load song titles into a dictionary
def load_song_titles(info_file):
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

# Initialize LAION-CLAP model
def initialize_model(checkpoint_path):
    """Initialize the CLAP model."""
    model = CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(checkpoint_path)
    return model

# Vectorize and write song titles to TSV
def vectorize_song_titles(model, song_titles, output_file):
    """Generate and save text embeddings for song titles."""
    with open(output_file, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        
        # Write header
        header = ['id'] + [f'embedding_{i}' for i in range(EMBED_LEN)]
        writer.writerow(header)
        
        for song_id, song_title in tqdm(song_titles.items(), total=len(song_titles), desc="Vectorizing Song Titles"):
            # Generate text embedding
            text_embedding = model.get_text_embedding([song_title])[0]
            
            # Write the embedding to the TSV
            writer.writerow([song_id] + text_embedding.tolist())

    print(f"Song title embeddings have been written to {output_file}")

# Main script
if __name__ == "__main__":
    # Path to song information file (TSV)
    info_file_path = "dataset/id_information_mmsr.tsv"
    
    # Output file for embeddings
    output_file = "dataset/id_laionclap_songtitles.tsv"
    
    # Checkpoint path for the CLAP model
    checkpoint_path = "music_audioset_epoch_15_esc_90.14.pt"
    
    # Load song titles
    song_titles = load_song_titles(info_file_path)
    
    # Initialize the model
    clap_model = initialize_model(checkpoint_path)
    
    # Vectorize song titles
    vectorize_song_titles(clap_model, song_titles, output_file)