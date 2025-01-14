import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from baseline_script import Track, IRSystem, preprocess

class CombinedCLAPIRSystem(IRSystem):
    """
    Information Retrieval System combining CLAP text and audio embeddings.
    """

    def __init__(self, tracks):
        """
        Initialize the system with tracks and prepare the combined embedding matrix.
        
        Args:
            tracks: List of Track objects
        """
        super().__init__(tracks)
        
        # Ensure both CLAP text and audio embeddings exist
        valid_tracks = [
            track for track in tracks 
            if track.clap_text_vector is not None and track.clap_audio_vector is not None
        ]
        self.tracks = valid_tracks
        
        # Combine embeddings by averaging CLAP text and audio embeddings
        self.embedding_matrix = np.vstack([
            self.combine_embeddings(track.clap_text_vector, track.clap_audio_vector)
            for track in valid_tracks
        ])
    
    @staticmethod
    def combine_embeddings(text_vector, audio_vector):
        """
        Combine CLAP text and audio embeddings by averaging them.
        
        Args:
            text_vector: CLAP text embedding
            audio_vector: CLAP audio embedding
        
        Returns:
            Combined embedding vector
        """
        return (text_vector + audio_vector) / 2
    
    def query(self, query: Track, n=10):
        """
        Find n most similar tracks based on combined CLAP features.
        
        Args:
            query: Query Track object
            n: Number of tracks to return
        
        Returns:
            List of n most similar tracks and their similarity scores
        """
        # Ensure query has both CLAP text and audio embeddings
        if query.clap_text_vector is None or query.clap_audio_vector is None:
            raise ValueError("Query track does not have both CLAP text and audio embeddings")
        
        # Combine query embeddings
        query_vector = self.combine_embeddings(query.clap_text_vector, query.clap_audio_vector)
        query_vector = query_vector.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.embedding_matrix)[0]
        
        # Get top n indices
        top_indices = np.argsort(similarities)[::-1][:n]
        
        # Handle case where query is in the dataset
        query_idx = self.tracks.index(query) if query in self.tracks else -1
        if query_idx in top_indices:
            top_indices = np.delete(top_indices, np.where(top_indices == query_idx))
        
        probabilities = similarities[top_indices]
        return [self.tracks[i] for i in top_indices], probabilities[:n]
    
    def calculate_similarities(self, query: Track) -> np.ndarray:
        """
        Calculate cosine similarities between the query and all tracks in the dataset.
        
        Args:
            query: Query Track object
        
        Returns:
            Array of cosine similarities
        """
        if query.clap_text_vector is None or query.clap_audio_vector is None:
            raise ValueError("Query track does not have both CLAP text and audio embeddings")
        
        # Combine query embeddings
        query_vector = self.combine_embeddings(query.clap_text_vector, query.clap_audio_vector)
        query_vector = query_vector.reshape(1, -1)
        
        # Compute cosine similarity
        return cosine_similarity(query_vector, self.embedding_matrix)[0]