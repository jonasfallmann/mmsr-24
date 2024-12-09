# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from ast import literal_eval


# %%
class Track:
    def __init__(self, track_id, track_name, artist, album_name, url, tfidf_vector, genres, top_genres):
        self.track_id = track_id
        self.track_name = track_name
        self.artist = artist
        self.album_name = album_name
        self.url = url
        self.tfidf_vector = tfidf_vector
        self.genres = genres
        self.top_genres = top_genres
      
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

def preprocess(basic_information: pd.DataFrame, youtube_urls: pd.DataFrame, tfidf_df: pd.DataFrame, genres_df: pd.DataFrame, tags_df: pd.DataFrame):
    basic_with_links = pd.merge(basic_information, youtube_urls, how="left", on="id")
    tracks = []

    def get_top_genres(tag_weight_dict):
        tags = literal_eval(tag_weight_dict)
        max_score = max(tags.values())
        top_genres = [tag for tag, score in tags.items() if score == max_score]
        return top_genres
    
    tags_df['top_genre'] = tags_df['(tag, weight)'].apply(get_top_genres)
    tags_dict = tags_df[['id', 'top_genre']].set_index('id').to_dict()['top_genre']

    for _, row in basic_with_links.iterrows():
        track_id = row['id']
        tfidf_vector = tfidf_df.loc[track_id].values if track_id in tfidf_df.index else None
        genres = eval(genres_df.loc[track_id].genre) if track_id in genres_df.index else []
        top_genres = tags_dict.get(track_id, None)
        
        track = Track(
            track_id,
            row['song'],
            row['artist'],
            row['album_name'],
            row["url"],
            tfidf_vector,
            genres,
            top_genres
        )
        tracks.append(track)
    return tracks


