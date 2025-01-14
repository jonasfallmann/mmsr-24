# %%
from enum import Enum

import pandas as pd
import numpy as np
from tqdm import tqdm
from ast import literal_eval
from typing import Protocol
from scipy.stats import rankdata

class FeatureType(Enum):
    TFIDF = 'tfidf'
    BERT = 'bert'
    CLAP_TEXT = 'clap_text'
    SPECTRAL = 'spectral'
    MUSICNN = 'musicnn'
    RESNET = 'resnet'
    VGG19 = 'vgg19'
    CLAP_AUDIO = 'clap_audio'


class Tag:
    """
    Tag class representing a tag with associated weight.

    Attributes:
        tag: Tag name
        weight: Tag weight
    """

    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight

    def __str__(self):
        return f'{self.tag} ({self.weight})'

# %%
class Track:
    """
    Track class representing a music track with multiple feature representations.
    
    Attributes:
        track_id: Unique identifier for the track
        track_name: Name of the song
        artist: Artist name
        album_name: Album name
        url: YouTube URL for the track
        
        Text Features:
        tfidf_vector: TF-IDF representation of lyrics
        bert_vector: BERT embedding of lyrics
        clap_text_vector: Microsoft CLAP text embedding
        
        Audio Features:
        spectral_vector: Spectral pattern features from BLF
        musicnn_vector: Deep learning features from MusicNN
        clap_vector: Microsoft CLAP features
        
        Visual Features:
        resnet_vector: ResNet features from video frames
        vgg19_vector: VGG19 features from video frames
        
        Metadata:
        genres: List of genres associated with the track
        top_genres: Most relevant genres based on tag weights
    """
    
    def __init__(
        self,
        track_id,
        track_name,
        artist,
        album_name,
        url,
        tfidf_vector,
        bert_vector=None,
        clap_text_vector=None,
        spectral_vector=None,
        musicnn_vector=None,
        clap_vector=None,
        resnet_vector=None,
        vgg19_vector=None,
        genres=None,
        top_genres=None,
        popularity=None,
        tags: list[Tag] | None = None
    ):
        # Basic information
        self.track_id = track_id
        self.track_name = track_name
        self.artist = artist
        self.album_name = album_name
        self.url = url
        
        # Text features
        self.tfidf_vector = tfidf_vector
        self.bert_vector = bert_vector
        self.clap_text_vector = clap_text_vector
        
        # Audio features
        self.spectral_vector = spectral_vector
        self.musicnn_vector = musicnn_vector
        self.clap_vector = clap_vector
        
        # Visual features
        self.resnet_vector = resnet_vector
        self.vgg19_vector = vgg19_vector
        
        # Metadata
        self.genres = genres if genres is not None else []
        self.top_genres = top_genres
        
        self.popularity = popularity

        self.tags = tags
      
    
    def __str__(self):
        """String representation of the track"""
        return f'{self.track_id} - {self.track_name} - {self.artist} - {self.album_name}'


# %%
class EvaluationMetric:
    def __init__(self):
        pass

    def evaluate(self, recommended_tracks, relevant_tracks):
        pass

# %%
class EvaluationProtocol(Protocol):
    def __init__(self):
        pass

    def evaluate(self, ir_system):
        pass


class IRSystem:
    def __init__(self, tracks):
        self.tracks = tracks
        self.name = ""

    def query(self, query: Track, n=10) -> (list[Track], list[float]):
        pass

    def set_name(self, name):
        self.name = name
        return self

    def calculate_similarities(self, query: Track) -> list[float] | np.ndarray:
        pass


class BaselineIRSystem(IRSystem):
    def __init__(self, tracks):
        super().__init__(tracks)

    def query(self, query: Track, n = 10):
        # return n random tracks, excluding the query track
        remaining_tracks = [t for t in self.tracks if t.track_id != query.track_id]
        return np.random.choice(remaining_tracks, n, replace=False).tolist(), np.random.rand(n)

    def calculate_similarities(self, query: Track) -> list[float] | np.ndarray:
        return np.random.rand(len(self.tracks))


def preprocess(
    basic_information: pd.DataFrame, 
    youtube_urls: pd.DataFrame, 
    tfidf_df: pd.DataFrame, 
    genres_df: pd.DataFrame, 
    tags_df: pd.DataFrame,
    spotify_df: pd.DataFrame,
    lastfm_df: pd.DataFrame,
    bert_df: pd.DataFrame = None,
    clap_text_df: pd.DataFrame = None,
    spectral_df: pd.DataFrame = None,
    musicnn_df: pd.DataFrame = None,
    clap_df: pd.DataFrame = None,
    resnet_df: pd.DataFrame = None,
    vgg19_df: pd.DataFrame = None
):
    """
    Preprocess data and create Track objects with multiple feature types.
    
    Parameters:
        basic_information: DataFrame with basic track information
        youtube_urls: DataFrame with YouTube URLs
        tfidf_df: DataFrame with TF-IDF vectors
        genres_df: DataFrame with genre information
        tags_df: DataFrame with tag information
        bert_df: DataFrame with BERT embeddings
        clap_text_df: DataFrame with Microsoft CLAP text embeddings
        spectral_df: DataFrame with spectral audio features
        musicnn_df: DataFrame with MusicNN features
        clap_df: DataFrame with Microsoft CLAP features
        resnet_df: DataFrame with ResNet features
        vgg19_df: DataFrame with VGG19 features
    
    Returns:
        List of Track objects with all available features
    """
    # Merge basic info with URLs
    basic_with_links = pd.merge(basic_information, youtube_urls, how="left", on="id")
    tracks = []

    # Genre processing
    def get_top_genres(tag_weight_dict, genre_tags):
        tags = literal_eval(tag_weight_dict)
        genre_tags = {k: tags[k] for k in genre_tags if k in tags}
        max_score = max(genre_tags.values()) if genre_tags else 0
        top_genres = [tag for tag, score in genre_tags.items() if score == max_score]
        return top_genres
    
    def get_popularity_score(spotify_df, lastfm_df):
        df = pd.merge(spotify_df, lastfm_df, how="left", on="id")[["id", "popularity", "total_listens"]]
        df['percentile_popularity'] = rankdata(df['popularity'], method='average') / len(df)
        df['log_clicks'] = np.log1p(df['total_listens'])
        df['percentile_clicks'] = rankdata(df['log_clicks'], method='average') / len(df)
        df['combined_percentile_score'] = (df['percentile_popularity'] + df['percentile_clicks'])/2
        return df[['id', 'combined_percentile_score']]
    
    basic_with_links = pd.merge(basic_with_links, get_popularity_score(spotify_df, lastfm_df), how="left", on="id")
    genre_tags = set([genre for sublist in genres_df['genre'].apply(literal_eval) for genre in sublist])
    tags_df['top_genre'] = tags_df['(tag, weight)'].apply(lambda x: get_top_genres(x, genre_tags))
    tags_dict = tags_df[['id', 'top_genre']].set_index('id').to_dict()['top_genre']

    # Process each track
    for _, row in basic_with_links.iterrows():
        track_id = row['id']
        
        # Text features
        tfidf_vector = tfidf_df.loc[track_id].values if track_id in tfidf_df.index else None
        bert_vector = bert_df.loc[track_id].values if bert_df is not None and track_id in bert_df.index else None
        clap_text_vector = clap_text_df.loc[track_id].values if clap_text_df is not None and track_id in clap_text_df.index else None
        
        # Audio features
        spectral_vector = spectral_df.loc[track_id].values if spectral_df is not None and track_id in spectral_df.index else None
        musicnn_vector = musicnn_df.loc[track_id].values if musicnn_df is not None and track_id in musicnn_df.index else None
        clap_vector = clap_df.loc[track_id].values if clap_df is not None and track_id in clap_df.index else None
        
        # Visual features
        resnet_vector = resnet_df.loc[track_id].values if resnet_df is not None and track_id in resnet_df.index else None
        vgg19_vector = vgg19_df.loc[track_id].values if vgg19_df is not None and track_id in vgg19_df.index else None
        
        # Genre and tag information
        genres = []
        if track_id in genres_df["id"].tolist():
            genres = genres_df.loc[genres_df["id"] == track_id, "genre"].values[0]
        top_genres = tags_dict.get(track_id, None)

        tags = tags_df.loc[tags_df['id'] == track_id, '(tag, weight)'].values[0]
        tags = literal_eval(tags) if tags else {}
        tags = [Tag(tag, weight) for tag, weight in tags.items()]
        # order tags by weight
        tags = sorted(tags, key=lambda x: x.weight, reverse=True)
        
        # Create track object
        track = Track(
            track_id=track_id,
            track_name=row['song'],
            artist=row['artist'],
            album_name=row['album_name'],
            url=row["url"],
            tfidf_vector=tfidf_vector,
            bert_vector=bert_vector,
            clap_text_vector=clap_text_vector,
            spectral_vector=spectral_vector,
            musicnn_vector=musicnn_vector,
            clap_vector=clap_vector,
            resnet_vector=resnet_vector,
            vgg19_vector=vgg19_vector,
            genres=genres,
            top_genres=top_genres,
            popularity=row['combined_percentile_score'],
            tags=tags
        )
        tracks.append(track)
    
    return tracks
