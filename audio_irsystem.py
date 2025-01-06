import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from baseline_script import Track, IRSystem, preprocess

class AudioIRSystem(IRSystem):
    """
    Audio Information Retrieval System that can use either Spectral or MusicNN features.
    
    Spectral Features (BLF - Block-Level Features):
    - Captures frequency content and patterns in the audio
    - Represents the spectral characteristics of the music
    - Uses cosine similarity because:
        1. Spectral patterns are dense feature vectors
        2. The relative distribution of frequencies is more important than absolute values
        3. Scale invariance helps handle volume differences
    
    MusicNN:
    - Deep learning embeddings trained specifically for music understanding
    - High-level musical feature representations
    - Uses cosine similarity because:
        1. Neural embeddings typically work best with cosine similarity
        2. The features are dense and directional
        3. The magnitude is less important than the pattern it represents
    """
    
    def __init__(self, tracks, feature_type='spectral'):
        """
        Initialize Audio IR System with either Spectral or MusicNN features
        
        Args:
            tracks: List of Track objects
            feature_type: 'spectral' or 'musicnn'
        """
        super().__init__(tracks)
        self.feature_type = feature_type.lower()
        
        # Validate feature type
        if self.feature_type not in ['spectral', 'musicnn']:
            raise ValueError("feature_type must be either 'spectral' or 'musicnn'")
        
        # Select the appropriate vector attribute
        vector_attr = 'spectral_vector' if self.feature_type == 'spectral' else 'musicnn_vector'
        
        # Filter valid tracks and create matrix
        valid_tracks = [track for track in tracks if getattr(track, vector_attr) is not None]
        self.tracks = valid_tracks
        self.embedding_matrix = np.vstack([getattr(track, vector_attr).reshape(1, -1) 
                                         for track in valid_tracks])
    
    def query(self, query: Track, n=10):
        """Find n most similar tracks based on chosen audio features"""
        # Get the appropriate vector based on feature type
        vector_attr = 'spectral_vector' if self.feature_type == 'spectral' else 'musicnn_vector'
        query_vector = getattr(query, vector_attr)
        
        if query_vector is None:
            raise ValueError(f"Query track does not have {self.feature_type} vector")
            
        query_vector = query_vector.reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.embedding_matrix)[0]
        
        # Handle case where query track is in the dataset
        query_idx = self.tracks.index(query) if query in self.tracks else -1
        top_indices = np.argsort(similarities)[::-1]
        
        if query_idx != -1:
            top_indices = top_indices[top_indices != query_idx]
        
        return [self.tracks[idx] for idx in top_indices[:n]]