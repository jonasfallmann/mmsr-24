import os
import csv
from msclap import CLAP
from tqdm import tqdm

EMBED_LEN = 1024

def load_genre_list(genre_file):
    """Load unique genres from a TSV file into a set."""
    genres = set()
    with open(genre_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip the header
        for row in reader:
            genre_list = eval(row[1])  # Convert string representation of list back to list
            genres.update(genre_list)  # Add to the set
    return genres

def load_tags_data(tags_file, exclude_genres):
    """Load tags data from a TSV file and exclude specified genres."""
    tags_data = {}
    with open(tags_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip the header
        for row in reader:
            song_id = row[0]
            tags_dict = eval(row[1])  # Convert string representation of dict back to dict
            # Exclude tags that contain any genre as a substring
            filtered_tags = [
                tag for tag in tags_dict.keys() 
                if not any(genre in tag for genre in exclude_genres)
            ]
            tags_data[song_id] = " ".join(filtered_tags)  # Concatenate tags into a single string
    return tags_data

# Initialize the CLAP model
clap_model = CLAP(version='2023', use_cuda=False)

# Path to the genre and tags files
genre_file_path = "dataset/id_genres_mmsr.tsv"
tags_file_path = "dataset/id_tags_dict.tsv"

# Load the genre list
genres = load_genre_list(genre_file_path)
print(genres)

# Load the tags data, excluding genres
tags_data = load_tags_data(tags_file_path, genres)

# Write the text embeddings into a TSV file
output_file = "dataset/id_clap_tags_mmsr.tsv"
with open(output_file, 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    
    # Write the header
    header = ['id'] + [f'embedding_{i}' for i in range(EMBED_LEN)]
    writer.writerow(header)
    
    for song_id, tags in tqdm(tags_data.items(), total=len(tags_data), desc="Embedding Tags"):
        # Generate text embedding for concatenated tags
        text_embedding = clap_model.get_text_embeddings([tags])[0]
        
        # Write to the file
        writer.writerow([song_id] + text_embedding.tolist())

print(f"Tag embeddings have been written to {output_file}")