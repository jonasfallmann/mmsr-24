import os
import csv
from msclap import CLAP
from tqdm import tqdm

N_CHUNKS = 5
EMBED_LEN = 1024

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

clap_model = CLAP(version='2023', use_cuda=False)

# Get all .ogg files in the downloads/converted folder
folder_path = "downloads/converted"
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.ogg')]

# Write the vectors into a TSV file
with open('dataset/id_clap_mmsr.tsv', 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    
    # Write the header
    header = ['id'] + [f'embedding_{i}' for i in range(EMBED_LEN)]
    writer.writerow(header)
    
    for file_chunk in tqdm(chunks(file_paths, N_CHUNKS), total=len(file_paths)//N_CHUNKS + 1, desc="Vectorizing"):
        # Vectorize the audio files in the current chunk
        audio_embeddings = clap_model.get_audio_embeddings(file_chunk)
        for file_path, embedding in zip(file_chunk, audio_embeddings):
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            writer.writerow([file_id] + embedding.tolist())

print("Audio embeddings have been written to dataset/id_clap_mmsr.tsv")