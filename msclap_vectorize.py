import os
import csv
from msclap import CLAP

clap_model = CLAP(version='2023', use_cuda=False)

# Get all .ogg files in the downloads/converted folder
folder_path = "downloads/converted"
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.ogg')]

# Vectorize the audio files
audio_embeddings = clap_model.get_audio_embeddings(file_paths)

# Write the vectors into a TSV file
with open('dataset/id_clap_mmsr.tsv', 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    for file_path, embedding in zip(file_paths, audio_embeddings):
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        writer.writerow([file_id] + embedding.tolist())

print("Audio embeddings have been written to audio_vectors.tsv")