import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys
import random
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import cornac
from cornac.metrics import RMSE
from cornac.models import ItemKNN
from cornac.eval_methods import RatioSplit
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import zipfile
import kagglehub
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer


def download_data():
    path = kagglehub.dataset_download("asaniczka/trending-youtube-videos-113-countries")

    filePath = os.path.join(path, "trending_yt_videos_113_countries.csv")
    data = pd.read_csv(filePath, sep=',')

    return data

def filter_columns(data):

    columns_useful = [
    'title', 'channel_name', 'country', 'view_count', 'like_count', 'comment_count',
    'video_tags', 'kind', 'publish_date', 'language'
    ]


    columns_validas = [c for c in columns_useful if c in data.columns]


    filtered_data = data[columns_validas]

    return filtered_data


def calculate_sentiment(data):

    data["video_tags"] = data["video_tags"].astype(str).str.lower()

    data["tokens"] = data["video_tags"].apply(word_tokenize)

    stop_words = set(stopwords.words("english"))
    data["filtered_tokens"] = data["tokens"].apply(lambda words: [w for w in words if w not in stop_words])

    # Convert tokens back into a string for sentiment analysis
    data["clean_text"] = data["filtered_tokens"].apply(lambda words: " ".join(words))

    # Apply Sentiment Analysis using VADER
    sia = SentimentIntensityAnalyzer()
    data["sentiment_score"] = data["clean_text"].apply(lambda text: sia.polarity_scores(text)["compound"])

    # Categorize sentiment based on score
    data["sentiment"] = data["sentiment_score"].apply(lambda score: 
        "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral"))
    
    data[["title", "video_tags", "clean_text", "sentiment_score", "sentiment"]].head()
    
    return data


def generate_engagement_score(data):

    data['engagement_score'] = (data['like_count'] * 2 + data['comment_count'] * 3) / (data['view_count'] + 1)

    # Sorting by engagement score in descending order
    data_sorted = data.sort_values(by='engagement_score', ascending=False)

    min_score = data['engagement_score'].min()
    max_score = data['engagement_score'].max()

    # Create the scaled engagement score column
    data['scaled_engagement_score'] = 1 + 9 * ((data['engagement_score'] - min_score) / (max_score - min_score))

    # Select the relevant columns and sort by scaled_engagement_score in descending order
    data_sorted = data[['title', 'engagement_score', 'scaled_engagement_score']].sort_values(by='scaled_engagement_score', ascending=False)

    channel_engagement = data.groupby('channel_name')['engagement_score'].mean().reset_index()
    channel_engagement = channel_engagement.rename(columns={'engagement_score': 'avg_engagement_score'})

    # Calculate min and max of avg_engagement_score
    min_score_channel = channel_engagement['avg_engagement_score'].min()
    max_score_channel = channel_engagement['avg_engagement_score'].max()

    # Create the scaled average engagement score column
    channel_engagement['scaled_avg_engagement_score'] = 1 + 9 * ((channel_engagement['avg_engagement_score'] - min_score_channel) / (max_score_channel - min_score_channel))

    # Sort by avg_engagement_score in descending order
    channel_analysis = channel_engagement.sort_values(by='avg_engagement_score', ascending=False)

    return channel_analysis

def initialize_data():

    data = download_data()

    # Filter unused columns
    data = filter_columns(data)

    data = generate_engagement_score(data)

    return data


def sparse_to_dataframe(sparse_matrix, column_prefix="feature"):

    df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix)
    df.columns = [f"{column_prefix}_{i}" for i in range(df.shape[1])]
    return df


def vectorize_df(data):
    vectorizer = HashingVectorizer(n_features=5000, alternate_sign=False)

    data_tf = vectorizer.transform(data["clean_text"])

    data_tf_df = sparse_to_dataframe(data_tf)

    data_tf_df.insert(0, "title", data["title"].values)

    return data_tf_df

def recommend_videos_sentiment(data, user_sentiment, num_recommendations=5, sample_size=100):

    data_tf_df = vectorize_df(data)
#  Filter videos with the desired feeling
    filtered_videos = data[data["sentiment"] == user_sentiment]

    if filtered_videos.empty or len(filtered_videos) < num_recommendations:
        print("There aren't enough videos with this feeling.")
        return []


# Get the common indices between videos and features
    filtered_indices = filtered_videos.index.intersection(data_tf_df.index)

    if filtered_indices.empty:
        print("No matches were found with the vectorized data.")
        return []

   
# Select features (except 'title' column)
    features_matrix = data_tf_df.loc[filtered_indices, data_tf_df.columns[1:]]

    if features_matrix.shape[0] < 2:
        print("There is not enough data to recommend.")
        return []

   
#  Take a random sample (max sample_size)
    sampled_indices = np.random.choice(features_matrix.index, min(sample_size, len(features_matrix)), replace=False)
    sampled_features = features_matrix.loc[sampled_indices]

    #  Fit and find neighbors
    nn_model = NearestNeighbors(n_neighbors=num_recommendations + 1, metric="cosine")
    nn_model.fit(sampled_features)

    # Choose one of the sampled videos to search for similar ones
    query_vector = sampled_features.iloc[[0]]  

    distances, indices = nn_model.kneighbors(query_vector)
    
    recommended_indices = sampled_features.iloc[indices[0][1:]].index  

    recommended_titles = data.loc[recommended_indices]["title"].tolist()
    return recommended_titles, sampled_features, indices


def clustering_with_neighbors(data_tf_df, data, user_sentiment="positive", num_queries=5, num_recommendations=5, sample_size=500):
    filtered_videos = data[data["sentiment"] == user_sentiment]
    filtered_indices = filtered_videos.index.intersection(data_tf_df.index)

    if filtered_indices.empty:
        print("No se encontraron coincidencias.")
        return

    features_matrix = data_tf_df.loc[filtered_indices, data_tf_df.columns[1:]]
    if features_matrix.shape[0] < 2:
        print("Datos insuficientes.")
        return

    # RANDOM SAMPLE
    sampled_indices = np.random.choice(features_matrix.index, min(sample_size, len(features_matrix)), replace=False)
    sampled_features = features_matrix.loc[sampled_indices]
    
    # Initialize
    cluster_labels = np.full(len(sampled_features), -1)
    cluster_id = 0
    
    nn_model = NearestNeighbors(n_neighbors=num_recommendations + 1, metric="cosine")
    nn_model.fit(sampled_features)

    used_indices = set()

    for i in range(num_queries):
        query_idx = np.random.choice(sampled_features.index.difference(used_indices))
        distances, indices = nn_model.kneighbors(sampled_features.loc[[query_idx]])
        group_indices = sampled_features.iloc[indices[0]].index

        for idx in group_indices:
            cluster_labels[sampled_features.index.get_loc(idx)] = cluster_id
            used_indices.add(idx)
        
        cluster_id += 1

    # Filter valid data
    valid_mask = cluster_labels != -1
    final_features = sampled_features[valid_mask]
    final_labels = cluster_labels[valid_mask]

    return final_features, final_labels

def sil_score(final_features, final_labels, num_queries):

    # Silhouette Score
    sil_score = silhouette_score(final_features, final_labels)
    sil_values = silhouette_samples(final_features, final_labels)

    # Graphic
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10

    for i in np.unique(final_labels):
        ith_values = sil_values[final_labels == i]
        ith_values.sort()
        size = len(ith_values)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10

    ax.axvline(sil_score, color="red", linestyle="--", label=f"Average silhouette : {sil_score:.2f}")
    ax.set_title(f"Silhouette Plot with {num_queries} clusters of NearestNeighbors")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Group")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return sil_score


