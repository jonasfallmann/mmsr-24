from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE

import baseline_script
from baseline_script import IRSystem


# define an enum for type of feature
class FeatureType(Enum):
    TFIDF = 'tfidf'
    BERT = 'bert'
    SPECTRAL = 'spectral'
    MUSICNN = 'musicnn'
    RESNET = 'resnet'
    VGG19 = 'vgg19'

class ReducerType(Enum):
    PCA = 'pca'
    ICA = 'ica'

class EarlyFusionIrSystem(IRSystem):
    def __init__(self,  tracks: list[baseline_script.Track], featureSet1: FeatureType, featureSet2: FeatureType, n_dims=-1, reducer: ReducerType = ReducerType.PCA):
        """
        Initialize an IR system that uses early fusion of two feature sets

        Args:
            tracks: List of Track objects
            featureSet1: FeatureType enum for the first feature set
            featureSet2: FeatureType enum for the second feature set
        """
        super().__init__(tracks)
        self.featureSet1 = featureSet1
        self.featureSet2 = featureSet2
        self.n_dims = n_dims
        self.reducer = reducer

        # featureSet1 must not be the same as featureSet2
        if self.featureSet1 == self.featureSet2:
            raise ValueError("featureSet1 and featureSet2 must be different")
        # Concatenate feature vectors
        embedding_matrix = self.to_fused_vector(tracks)
        self.dimensionality_reducer = self.prepare_dimensionality_reducer(embedding_matrix)
        if self.dimensionality_reducer is not None:
            embedding_matrix = self.dimensionality_reducer.transform(embedding_matrix)
        self.embedding_matrix = embedding_matrix



    def query(self, query: baseline_script.Track, n=10):
        """
        Find n most similar tracks based on early fused features

        Args:
            query: Query Track object
            n: Number of tracks to return

        Returns:
            list[Track]: List of n most similar tracks
        """
        # Get the appropriate vector based on feature type
        query_vector = self.to_fused_vector([query])
        if self.dimensionality_reducer is not None:
            query_vector = self.dimensionality_reducer.transform(query_vector)

        # Calculate cosine similarity between query and all tracks
        similarities = cosine_similarity(query_vector, self.embedding_matrix)[0]

        # Get indices of n most similar tracks
        indices = similarities.argsort()[::-1]

        # remove the query track from the list
        query_idx = self.tracks.index(query) if query in self.tracks else -1
        if query_idx != -1:
            indices = indices[indices != query_idx]

        # Return n most similar tracks
        return [self.tracks[i] for i in indices[:n]]


    def prepare_dimensionality_reducer(self, matrix: np.ndarray):
        # prepare pca to reduce the dimensionality
        if self.n_dims > 0:
            dimensionality_reducer = PCA(n_components=self.n_dims)
            if self.reducer == ReducerType.ICA:
                dimensionality_reducer = FastICA(n_components=self.n_dims)
            dimensionality_reducer.fit(matrix)
            return dimensionality_reducer
        return None

    def to_fused_vector(self, tracks: list[baseline_script.Track]):
        """
        Convert a list of tracks to a fused feature vector

        Args:
            tracks: List of Track objects

        Returns:
            np.ndarray: Fused feature vector
        """
        feature_vectors1 = np.asarray(self.select_feature_vectors(self.featureSet1, tracks))
        feature_vectors2 = np.asarray(self.select_feature_vectors(self.featureSet2, tracks))

        # Normalize feature vectors
        feature_vectors1 = feature_vectors1 / np.linalg.norm(feature_vectors1, axis=1)[:, None]
        feature_vectors2 = feature_vectors2 / np.linalg.norm(feature_vectors2, axis=1)[:, None]

        # Concatenate feature vectors
        concatenated = np.hstack([feature_vectors1, feature_vectors2])
        return concatenated

    def select_feature_vectors(self, featureSet: FeatureType, tracks: list[baseline_script.Track]):
        match featureSet:
            case FeatureType.TFIDF:
                return [track.tfidf_vector for track in tracks]
            case FeatureType.BERT:
                return [track.bert_vector for track in tracks]
            case FeatureType.SPECTRAL:
                return [track.spectral_vector for track in tracks]
            case FeatureType.MUSICNN:
                return [track.musicnn_vector for track in tracks]
            case FeatureType.RESNET:
                return [track.resnet_vector for track in tracks]
            case FeatureType.VGG19:
                return [track.vgg19_vector for track in tracks]
            case _:
                raise ValueError("Invalid feature type")