import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from baseline_script import Track, IRSystem, preprocess

class VisualIRSystem(IRSystem):
    """
    Visual Information Retrieval System using either ResNet or VGG19 features.
    
    ResNet Features:
    - Deep CNN embeddings from video frames sampled at 1Hz
    - Deeper architecture specialized in recognizing fine-grained visual patterns
    - Features are aggregated at track level using max and mean pooling
    
    VGG19 Features:
    - Classic CNN architecture known for its simplicity and effectiveness
    - Good at capturing hierarchical visual features
    - Also sampled at 1Hz and aggregated at track level
    
    Both feature types capture visual elements like:
    - Stage setups
    - Performance styles
    - Music video aesthetics
    - Visual themes and patterns
    
    Similarity Measure:
    - Uses cosine similarity because:
        1. Both networks produce dense neural embeddings where direction is more important than magnitude
        2. Cosine similarity is standard for deep learning embeddings as it captures semantic similarity
        3. Scale invariance helps handle different video qualities and lighting conditions
        4. The features are already normalized in their embedding space
    """
    
    def __init__(self, tracks, feature_type='resnet'):
        """
        Initialize Visual IR System with either ResNet or VGG19 features
        
        Args:
            tracks: List of Track objects
            feature_type: 'resnet' or 'vgg19'
        """
        super().__init__(tracks)
        self.feature_type = feature_type.lower()
        
        # Validate feature type
        if self.feature_type not in ['resnet', 'vgg19']:
            raise ValueError("feature_type must be either 'resnet' or 'vgg19'")
        
        # Select the appropriate vector attribute
        vector_attr = 'resnet_vector' if self.feature_type == 'resnet' else 'vgg19_vector'
        
        # Filter valid tracks and create matrix
        valid_tracks = [track for track in tracks if getattr(track, vector_attr) is not None]
        self.tracks = valid_tracks
        self.embedding_matrix = np.vstack([getattr(track, vector_attr).reshape(1, -1) 
                                         for track in valid_tracks])
    
    def query(self, query: Track, n=10, late_fusion=False):
        """Find n most similar tracks based on chosen visual features"""
        # Get the appropriate vector based on feature type
        vector_attr = 'resnet_vector' if self.feature_type == 'resnet' else 'vgg19_vector'
        query_vector = getattr(query, vector_attr)
        
        if query_vector is None:
            raise ValueError(f"Query track does not have {self.feature_type} vector")
            
        query_vector = query_vector.reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.embedding_matrix)[0]
        if late_fusion:
            return similarities
        
        # Handle case where query track is in the dataset
        query_idx = self.tracks.index(query) if query in self.tracks else -1
        top_indices = np.argsort(similarities)[::-1]
        
        if query_idx != -1:
            top_indices = top_indices[top_indices != query_idx]
        
        return [self.tracks[idx] for idx in top_indices[:n]]