# %%
import pandas as pd
import numpy as np
from tqdm import tqdm


# %%
class Track:
    def __init__(self, track_id, track_name, artist, album_name, url, tfidf_vector, genres):
        self.track_id = track_id
        self.track_name = track_name
        self.artist = artist
        self.album_name = album_name
        self.url = url
        self.tfidf_vector = tfidf_vector
        self.genres = genres
      
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

def preprocess(basic_information: pd.DataFrame, youtube_urls: pd.DataFrame, tfidf_df: pd.DataFrame, genres_df: pd.DataFrame):
    basic_with_links = pd.merge(basic_information, youtube_urls, how="left", on="id")
    tracks = []
    for index, row in basic_with_links.iterrows():
        track_id = row['id']
        tfidf_vector = tfidf_df.loc[track_id].values if track_id in tfidf_df.index else None
        genres = eval(genres_df.loc[track_id].genre) if track_id in genres_df.index else []
        
        track = Track(
            track_id,
            row['song'],
            row['artist'],
            row['album_name'],
            row["url"],
            tfidf_vector,
            genres
        )
        tracks.append(track)
    return tracks


