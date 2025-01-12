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
    
    def __init__(self, tracks, feature_type='spectral', diversification = 0.0, n_diverse = 0):
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
        self.diversification = diversification

        # get all top genres of all tracks
        all_genres = set()
        for track in valid_tracks:
            all_genres.update(track.top_genres)

        # calculate a vector for all genres by averaging the embeddings of all tracks with that genre
        genre_vectors = {}
        for genre in all_genres:
            genre_tracks = [track for track in valid_tracks if genre in track.top_genres]
            genre_vectors[genre] = np.mean([getattr(track, vector_attr) for track in genre_tracks], axis=0)

        self.genre_vectors = genre_vectors
        self.n_diverse = n_diverse
    
    def query(self, query: Track, n=10):
        """Find n most similar tracks based on chosen audio features"""
        # Get the appropriate vector based on feature type
        vector_attr = 'spectral_vector' if self.feature_type == 'spectral' else 'musicnn_vector'
        query_vector = getattr(query, vector_attr)
        
        if query_vector is None:
            raise ValueError(f"Query track does not have {self.feature_type} vector")


        top_indices, similarities = self.get_top_indices(query, query_vector)[:n]

        # if diversification is enabled, get top indices for diversification vectors and replace the last n_diverse tracks
        # with the diversified tracks
        if self.diversification > 0:
            diversification_vectors = []
            # get the mean vector of the query vector genres
            query_genres = query.top_genres
            query_genre_vector = np.mean([self.genre_vectors[genre] for genre in query_genres], axis=0)

            # get vector furthest away from the query genre vector using cosine similarity
            # stack all genre vectors into a matrix
            genre_vectors = np.vstack(list(self.genre_vectors.values()))
            genre_distances = cosine_similarity(query_genre_vector.reshape(1, -1), genre_vectors)[0]

            # get the top 5 genre vectors that are furthest away
            furthest_genre_indices = np.argsort(genre_distances)[:5]
            furthest_vectors = genre_vectors[furthest_genre_indices]
            # furthest_genres = [list(self.genre_vectors.keys())[i] for i in furthest_genre_indices]

            # calculate query vectors, each being pulled toward one of the furthest genre vectors
            for vector in furthest_vectors:
                diversification_vectors.append(query_vector + self.diversification * (vector - query_vector))
            distribution = self.distribute_integer(self.n_diverse, len(diversification_vectors))
            # drop last n_diverse tracks
            replacement_indices = np.zeros(self.n_diverse, dtype=int)
            overwrite_index = 0
            for idx, div_vector in enumerate(diversification_vectors):
                div_top_indices = self.get_top_indices(query, div_vector)
                dist = distribution[idx]
                top_idx = 0
                while dist > 0:
                    if div_top_indices[top_idx] not in top_indices and div_top_indices[top_idx] not in replacement_indices:
                        replacement_indices[overwrite_index] = div_top_indices[top_idx]
                        overwrite_index += 1
                        dist -= 1
                    top_idx += 1
            top_indices[-self.n_diverse:] = replacement_indices

        probabilities = similarities[top_indices]

        return [self.tracks[idx] for idx in top_indices[:n]], probabilities[:n]

    def calculate_similarities(self, query: Track) -> list[float] | np.ndarray:
        """
        Calculate the cosine similarities between the query track and all tracks in the dataset

        Args:
            query: Query Track object

        Returns:
            list[float] | np.ndarray: List of cosine similarities
        """
        # Get the appropriate vector based on feature type
        vector_attr = 'spectral_vector' if self.feature_type == 'spectral' else 'musicnn_vector'
        query_vector = getattr(query, vector_attr)

        if query_vector is None:
            raise ValueError(f"Query track does not have {self.feature_type} vector")

        query_vector = query_vector.reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.embedding_matrix)[0]

        return similarities

    def get_top_indices(self, query, query_vector):
        query_vector = query_vector.reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.embedding_matrix)[0]

        # Handle case where query track is in the dataset
        query_idx = self.tracks.index(query) if query in self.tracks else -1
        top_indices = np.argsort(similarities)[::-1]

        if query_idx != -1:
            top_indices = top_indices[top_indices != query_idx]

        return top_indices, similarities

    def distribute_integer(self, total, workers):
        # Basic division
        base_share = total // workers
        # Remaining amount to distribute
        remainder = total % workers

        # Create the distribution list
        distribution = [base_share] * workers

        # Distribute the remainder among the first `remainder` workers
        for i in range(remainder):
            distribution[i] += 1

        return distribution