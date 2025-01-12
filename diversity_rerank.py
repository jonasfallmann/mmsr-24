from baseline_script import Track, FeatureType, IRSystem
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DiversityRerank(IRSystem):
    def __init__(self, tracks: list[Track], ir_system: IRSystem, diversification: float,
                 dissimilarity_feature: FeatureType, pool_multiple = 5):
        super().__init__(tracks)
        self.diversification = diversification
        self.ir_system = ir_system
        self.dissimilarity_feature = dissimilarity_feature
        self.pool_multiple = pool_multiple

    def query(self, query: Track, n=10):
        # query n*5 tracks from the ir system
        tracks, similarities = self.ir_system.query(query, n * self.pool_multiple)
        dissimilarities = self.calculate_dissimilarity(query, tracks)

        # calculate a score for each track based on the similarity and dissimilarity
        scores = (1 - self.diversification) * similarities + self.diversification * dissimilarities

        # sort the tracks based on the scores
        sorted_indices = np.argsort(scores)
        sorted_similarities = similarities[sorted_indices]

        return [tracks[idx] for idx in sorted_indices[:n]], sorted_similarities[:n]

    def calculate_dissimilarity(self, query: Track, tracks: list[Track]):
        # calculate the dissimilarity between the query and the tracks based on the dissimilarity feature
        features = self.get_dissimilarity_features(tracks)
        query_feature = self.get_dissimilarity_features([query])

        similarities = cosine_similarity(query_feature, features)[0]
        dissimilarities = 1 - similarities
        return dissimilarities

    def get_dissimilarity_features(self, tracks: list[Track]):
        match self.dissimilarity_feature:
            case FeatureType.SPECTRAL:
                return np.asarray([track.spectral_vector for track in tracks])
            case FeatureType.MUSICNN:
                return np.asarray([track.musicnn_vector for track in tracks])
            case FeatureType.RESNET:
                return np.asarray([track.resnet_vector for track in tracks])
            case FeatureType.VGG19:
                return np.asarray([track.vgg19_vector for track in tracks])
            case FeatureType.BERT:
                return np.asarray([track.bert_vector for track in tracks])
            case FeatureType.TFIDF:
                return np.asarray([track.tfidf_vector for track in tracks])
            case _:
                raise ValueError("Invalid dissimilarity feature")
