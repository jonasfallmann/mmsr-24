# %%
import pandas as pd
import numpy as np


# %%
class Track:
    def __init__(self, track_id, track_name, artist, album_name):
        self.track_id = track_id
        self.track_name = track_name
        self.artist = artist
        self.album_name = album_name
      

    def __str__(self):
        return f'{self.track_id} - {self.track_name} - {self.artist} - {self.album_name}'


# %%
class EvaluationMetric:
    def __init__(self):
        pass

    def evaluate(self, recommended_tracks, relevant_tracks):
        pass

# %%
class EvaluationProtocol:
    def __init__(self):
        pass

    def evaluate(self, ir_system):
        pass

# %%
class IRSystem:
    def __init__(self, tracks):
        self.tracks = tracks

    def query(self, query: Track, n=10):
        pass


# %%
class BaselineIRSystem(IRSystem):
    def __init__(self, tracks):
        super().__init__(tracks)

    def query(self, query: Track, n = 10):
        # return n random tracks, excluding the query track
        remaining_tracks = [t for t in self.tracks if t.track_id != query.track_id]
        return np.random.choice(remaining_tracks, n, replace=False).tolist()




