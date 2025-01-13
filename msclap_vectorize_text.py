import os
import csv
from msclap import CLAP
from tqdm import tqdm

EMBED_LEN = 1024

def load_genre_data(genre_file):
    """Load genre data from a TSV file into a dictionary."""
    genre_data = {}
    with open(genre_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip the header
        for row in reader:
            song_id = row[0]
            genres = eval(row[1])  # Convert string representation of list back to list
            genre_data[song_id] = " ".join(genres)  # Concatenate genres into a single string
    return genre_data

# Initialize the CLAP model
clap_model = CLAP(version='2023', use_cuda=False)

# Path to the genre file
genre_file_path = "dataset/id_genres_mmsr.tsv"

# Load the genre data
genre_data = load_genre_data(genre_file_path)

# Write the text embeddings into a TSV file
output_file = "dataset/id_clap_genres_mmsr.tsv"
with open(output_file, 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    
    # Write the header
    header = ['id'] + [f'embedding_{i}' for i in range(EMBED_LEN)]
    writer.writerow(header)
    
    for song_id, genres in tqdm(genre_data.items(), total=len(genre_data), desc="Embedding Genres"):
        # Generate text embedding for concatenated genres
        text_embedding = clap_model.get_text_embeddings([genres])[0]
        
        # Write to the file
        writer.writerow([song_id] + text_embedding.tolist())

print(f"Genre embeddings have been written to {output_file}")