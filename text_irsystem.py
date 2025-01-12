import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from baseline_script import Track, IRSystem, preprocess

class TextIRSystem(IRSystem):
    """
    Text Information Retrieval System using either TF-IDF or BERT features.
    
    TF-IDF Features:
    - Traditional bag-of-words approach weighted by term importance
    - Good at capturing lexical matches and keyword importance
    - Sparse representation focusing on direct word overlap
    
    BERT Features:
    - Contextual embeddings from deep transformer model
    - Captures semantic relationships and context
    - Dense representation that understands word meaning
    
    Both feature types capture lyrical elements like:
    - Thematic content
    - Vocabulary usage
    - Semantic meaning
    - Writing style
    
    Similarity Measure:
    - Uses cosine similarity because:
        1. Both sparse (TF-IDF) and dense (BERT) vectors work well with cosine
        2. Length normalization helps compare texts of different lengths
        3. Focuses on relative term importance rather than absolute values
        4. Standard approach for both classical and neural text embeddings
    """
    
    def __init__(self, tracks, feature_type='tfidf'):
        """
        Initialize Text IR System with either TF-IDF or BERT features
        
        Args:
            tracks: List of Track objects
            feature_type: 'tfidf' or 'bert'
        """
        super().__init__(tracks)
        self.feature_type = feature_type.lower()
        
        # Validate feature type
        if self.feature_type not in ['tfidf', 'bert']:
            raise ValueError("feature_type must be either 'tfidf' or 'bert'")
        
        # Select the appropriate vector attribute
        vector_attr = 'tfidf_vector' if self.feature_type == 'tfidf' else 'bert_vector'
        
        # Filter valid tracks and create matrix
        valid_tracks = [track for track in tracks if getattr(track, vector_attr) is not None]
        self.tracks = valid_tracks
        self.embedding_matrix = np.vstack([getattr(track, vector_attr).reshape(1, -1) 
                                         for track in valid_tracks])
    
    def query(self, query: Track, n=10):
        """Find n most similar tracks based on chosen text features"""
        # Get the appropriate vector based on feature type
        vector_attr = 'tfidf_vector' if self.feature_type == 'tfidf' else 'bert_vector'
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

        probabilities = similarities[top_indices]
        
        return [self.tracks[idx] for idx in top_indices[:n]], probabilities[:n]

    def calculate_similarities(self, query: Track) -> list[float] | np.ndarray:
        """
        Calculate the cosine similarities between the query track and all tracks in the dataset

        Args:
            query: Query Track object

        Returns:
            list[float] | np.ndarray: List or array of cosine similarities
        """
        # Get the appropriate vector based on feature type
        vector_attr = 'tfidf_vector' if self.feature_type == 'tfidf' else 'bert_vector'
        query_vector = getattr(query, vector_attr)

        if query_vector is None:
            raise ValueError(f"Query track does not have {self.feature_type} vector")

        query_vector = query_vector.reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.embedding_matrix)[0]
        return similarities
