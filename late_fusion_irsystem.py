import pandas as pd
import numpy as np
from baseline_script import Track, IRSystem

class LateFusionIRSystem(IRSystem):
    def __init__(self, tracks: list[Track], systems: list[IRSystem], weights=None):
        """
        Initialize the late fusion IR system.
        
        Args:
            systems: List of IR system objects (e.g., TextIRSystem, VisualIRSystem).
            weights: List of weights for each system. Defaults to equal weighting.
        """
        super().__init__(tracks)
        if len(systems) < 2:
            raise ValueError("You need to provide at least 2 ir systems for fusion")
        if weights:
            if len(weights) != len(systems):
                raise ValueError("The number of weights have to equal the number of ir systems provided")
        if weights and sum(weights) != 1:
            raise ValueError("Weights have to add up to 1")
        self.tracks = tracks
        self.systems = systems
        self.weights = weights or [1 / len(systems)] * len(systems)
    
    def query(self, query_track: Track, n=10):
        """
        Perform late fusion to find the top N tracks.
        
        Args:
            query: The query track.
            n: Number of top tracks to retrieve.
        
        Returns:
            List of tuples (track, final_score).
        """
        similarities = []
        for irsys in self.systems:
            similarity = irsys.calculate_similarities(query_track)
            similarities.append(similarity)
        if self.weights:
            combined_similarities = np.average(similarities, axis=0, weights=self.weights)
        else:   
            combined_similarities = sum(similarities)/len(similarities)
        query_idx = self.tracks.index(query_track) if query_track in self.tracks else -1
        top_indices = np.argsort(combined_similarities)[::-1]
        
        if query_idx != -1:
            top_indices = top_indices[top_indices != query_idx]

        probabilities = combined_similarities[top_indices]

        return [self.tracks[idx] for idx in top_indices[:n]], probabilities[:n]
    