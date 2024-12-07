import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from baseline_script import Track, IRSystem, preprocess

class TextIRSystem(IRSystem):
    def __init__(self, tracks):
        super().__init__(tracks)
        valid_tracks = [track for track in tracks if track.tfidf_vector is not None]
        self.tracks = valid_tracks
        self.tfidf_matrix = np.vstack([track.tfidf_vector.reshape(1, -1) for track in valid_tracks])
    
    def query(self, query: Track, n=10):
        if query.tfidf_vector is None:
            raise ValueError("Query track does not have TF-IDF vector")

        query_vector = query.tfidf_vector.reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        query_idx = self.tracks.index(query) if query in self.tracks else -1
        top_indices = np.argsort(similarities)[::-1]
        
        if query_idx != -1:
            top_indices = top_indices[top_indices != query_idx]
        
        return [self.tracks[idx] for idx in top_indices[:n]]

if __name__ == "__main__":
    basic_info_df = pd.read_csv("dataset/id_information_mmsr.tsv", sep='\t')
    youtube_urls_df = pd.read_csv("dataset/id_url_mmsr.tsv", sep='\t')
    tfidf_df = pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep='\t', index_col=0)
    
    tracks = preprocess(basic_info_df, youtube_urls_df, tfidf_df)
    
    ir_system = TextIRSystem(tracks)
    
    query_track = tracks[0]
    similar_tracks = ir_system.query(query_track, n=5)
    
    print(f"Query track: {query_track}")
    print("\nSimilar tracks:")
    for track in similar_tracks:
        print(track)