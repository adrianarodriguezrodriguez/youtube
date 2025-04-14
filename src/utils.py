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
from sklearn.metrics.pairwise import cosine_similarity
from cornac import Experiment
from cornac.metrics import RMSE
from cornac.models import ItemKNN
from cornac.eval_methods import RatioSplit


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


def channel_based_cluster(data):

    # We group by channel and join tags from all the videos on the channel
    tags_por_canal = data.groupby("channel_name")["video_tags"].apply(lambda x: ' '.join(str(v) for v in x.dropna()))

    # TF-IDF + KMeans
    ## converts tags to numbers so you can do math with them. TF-IDF gives more weight to "important" words within the channel
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(tags_por_canal)
    km = KMeans(n_clusters=5, random_state=42).fit(X)

    # Asignamos clusters
    canales_cluster = pd.DataFrame({
        "channel_name": tags_por_canal.index,
        "cluster": km.labels_
    })

    return canales_cluster


def print_reco_target_cluster(canales_cluster, tags_por_canal, target_cluster):
    #  We select the target cluster (where there are more channels)
    cluster_objetivo = target_cluster

    # We filter the channels that belong to that cluster
    canales_en_cluster = canales_cluster[canales_cluster["cluster"] == cluster_objetivo]

    #  We select a random channel from that cluster
    canal_elegido = random.choice(canales_en_cluster["channel_name"].values)
    print(f"Channel automatically chosen from the cluster {cluster_objetivo}: '{canal_elegido}'\n")

    # TF-IDF of the cluster channel tags
    tags_cluster = tags_por_canal.loc[canales_en_cluster["channel_name"]]
    tfidf = TfidfVectorizer()
    X_cluster = tfidf.fit_transform(tags_cluster)

    # Index of the chosen channel
    index_canal = tags_cluster.index.get_loc(canal_elegido)

    # Cosine similarity between the channel and all others
    similitudes = cosine_similarity(X_cluster[index_canal], X_cluster).flatten()

    # 
    TOPK = 10
    similares_idx = similitudes.argsort()[::-1][1:TOPK+1]
    canales_similares = tags_cluster.index[similares_idx]


    print(f"Channels similars to '{canal_elegido}' in the cluster {cluster_objetivo}:\n")
    for i, canal in enumerate(canales_similares):
        print(f"{i+1}. {canal} (similarity: {similitudes[similares_idx[i]]:.2f})")


def pca(data):
    # Select numerical values
    numerical_features = ['view_count', 'like_count', 'comment_count']

    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numerical_features])

    # Apply PCA
    pca = PCA(n_components=len(numerical_features))  # We use as many components as numeric columns
    pca_result = pca.fit_transform(scaled_data)

    # Graph of the explained variance
    explained_variance = pca.explained_variance_ratio_

    return explained_variance

def create_engagement(data):

    # Creating the engagement score column
    data['engagement_score'] = (data['like_count'] * 2 + data['comment_count'] * 3) / (data['view_count'] + 1)

    # Sorting by engagement score in descending order
    data_sorted = data.sort_values(by='engagement_score', ascending=False)

    return data_sorted


def normalized_data(data):
    
    # Calculate min and max values of the engagement_score
    min_score = data['engagement_score'].min()
    max_score = data['engagement_score'].max()

    # Create the scaled engagement score column
    data['scaled_engagement_score'] = 1 + 9 * ((data['engagement_score'] - min_score) / (max_score - min_score))

    # Select the relevant columns and sort by scaled_engagement_score in descending order
    data_sorted = data[['title', 'engagement_score', 'scaled_engagement_score']].sort_values(by='scaled_engagement_score', ascending=False)

    return data_sorted


def average_engagement_by_channel(data):
    # Group by 'channel_name' and calculate the average engagement score
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

def recommend_similar_channels(data, channel_name, top_n=5):

    # We create a matrix to calculate similarity
    channel_matrix = data[["scaled_avg_engagement_score"]].values

    #  Cosine similarity between channels
    similarity_matrix = cosine_similarity(channel_matrix)  #we use the similitud of the cosine

    #  We created an index to easily search by name
    channel_names = data["channel_name"].values
    channel_index = {name: idx for idx, name in enumerate(channel_names)}

    if channel_name not in channel_index:
        print(f" Channel '{channel_name}' not found.")
        return []

    idx = channel_index[channel_name]
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # We sort by similarity (except the same channel)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar = sim_scores[1:top_n+1]  # omitimos el primero (es él mismo)

    print(f"Channels similar to '{channel_name}':")
    for i, score in top_similar:
        print(f"  - {channel_names[i]} (Similarity: {score:.3f})")

    return [channel_names[i] for i, _ in top_similar]

def generate_map_data(data):
    # Dictionary from ISO-2 a ISO-3
    codigo_iso3_manual = {
        'US': 'USA', 'BR': 'BRA', 'IN': 'IND', 'GB': 'GBR', 'FR': 'FRA',
        'DE': 'DEU', 'IT': 'ITA', 'ES': 'ESP', 'RU': 'RUS', 'JP': 'JPN',
        'KR': 'KOR', 'MX': 'MEX', 'CA': 'CAN', 'AU': 'AUS', 'ZW': 'ZWE',
        'AR': 'ARG', 'NG': 'NGA', 'SA': 'SAU', 'TR': 'TUR', 'CN': 'CHN'
    }


    data_mapa = data[["country", "engagement_score"]].dropna()

    # Convert codes from ISO-2 to ISO-3
    data_mapa["country"] = data_mapa["country"].map(codigo_iso3_manual)


    # Delete rows with unmapped countries
    data_mapa = data_mapa.dropna(subset=["country"])


    # Group and add the total engagement by country
    engagement_por_pais = data_mapa.groupby("country")["engagement_score"].sum().reset_index()
    engagement_por_pais.columns = ["country", "total_engagement"]


    # Calculate the total overall engagement
    total_global = engagement_por_pais["total_engagement"].sum()

    # Calculate the percentage by country
    engagement_por_pais["percentage"] = (engagement_por_pais["total_engagement"] / total_global) * 100

    # Round to display with 2 decimal places
    engagement_por_pais["percentage"] = engagement_por_pais["percentage"].round(2)

    return engagement_por_pais

def recomendar_canales_similares_en_pais(canal_objetivo, pais, df, top_n=10):
    # Filter by country
    df_pais = df[df["country"] == pais]

    if df_pais.empty:
        print(f"No channels found for the country: {pais}")
        return

    # Group by channel to calculate average engagement
    canal_avg = (
        df_pais.groupby("channel_name")["engagement_score"]
        .mean()
        .reset_index()
        .rename(columns={"engagement_score": "avg_engagement"})
    )

# Find engagement on the target channel

    if canal_objetivo not in canal_avg["channel_name"].values:
        print(f"The channel '{canal_objetivo}' not found in the country'{pais}'.")
        return

    eng_objetivo = canal_avg[canal_avg["channel_name"] == canal_objetivo]["avg_engagement"].values[0]

    # Calculate difference with the rest
    canal_avg["diff"] = (canal_avg["avg_engagement"] - eng_objetivo).abs()

    
    recomendaciones = (
        canal_avg[canal_avg["channel_name"] != canal_objetivo]
        .sort_values(by="diff")
        .head(top_n)
    )

    print(f"Channel simiñlar to '{canal_objetivo}' in'{pais}' (by average engagemennt):")
    display(recomendaciones[["channel_name", "avg_engagement", "diff"]])


def thematic_clusters(data, canales_cluster):
    
    data_country = data[["channel_name", "country"]].drop_duplicates()

    canales_cluster = canales_cluster.merge(data_country, on="channel_name", how="left")

    # Filter only US channels
    us_canales = canales_cluster[canales_cluster["country"] == "US"]

    return us_canales


def recommend_channels_us(data,us_canales, target_cluster):
    #  Filter only US cluster 0
    cluster_objetivo = target_cluster
    canales_cluster_0 = us_canales[us_canales["cluster"] == cluster_objetivo]

    #select a random country from cluster 0
    canal_base = np.random.choice(canales_cluster_0["channel_name"].unique())
    print(f"Channel selected from cluster 0: '{canal_base}'")

    # Get the average engagement of all channels in cluster 0
    df_engagement = (
        data[data["channel_name"].isin(canales_cluster_0["channel_name"])]
        .groupby("channel_name")["engagement_score"]
        .mean()
        .reset_index()
        .rename(columns={"engagement_score": "avg_engagement"})
    )

    #  Get base channel engagement
    eng_base = df_engagement[df_engagement["channel_name"] == canal_base]["avg_engagement"].values[0]

    # Calculate absolute difference with respect to the base channel
    df_engagement["diff"] = np.abs(df_engagement["avg_engagement"] - eng_base)


    recomendaciones = df_engagement[df_engagement["channel_name"] != canal_base]


    recomendaciones = recomendaciones.sort_values(by="diff").head(10)


    print(f"\n Channels similar to '{canal_base}' in EE.UU. (cluster {cluster_objetivo}):")
    display(recomendaciones)


def monthly_engagement(data):
    
    # Ensure publish_date is datetime
    data["publish_date"] = pd.to_datetime(data["publish_date"], errors="coerce")


    # Create column with the month of publication

    data["publish_month"] = data["publish_date"].dt.to_period("M")


    # Calculate the average engagement per month
    engagement_por_mes = (
        data.groupby("publish_month")["engagement_score"]
        .mean()
        .reset_index()
        .sort_values("publish_month")
    )

    # Convert publish_month to string for graphing
    engagement_por_mes["publish_month"] = engagement_por_mes["publish_month"].astype(str)

    return engagement_por_mes

def generate_matrix(data):
    
    # Preprocess titles and tags: combine 'title' and 'video_tags' into a single column
    data['title_tags_clean'] = data['title'].fillna('').str.lower() + ' ' + data['video_tags'].fillna('').str.lower()


    # Vectorize with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2), min_df=5)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['title_tags_clean'])

    return tfidf_matrix

def recomendar_videos_por_nombre(data, tfidf_matrix, video_title, top_n=5):
    video_title_lower = video_title.lower()

    
# Find all indexes that have the same title
    matching_indices = data[data['title'].str.lower() == video_title_lower].index.tolist()

    if not matching_indices:
        print(f"No video found with title: '{video_title}'")
        return

    
# Use the first index as a reference for the similarity calculation
    base_idx = matching_indices[0]

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[base_idx], tfidf_matrix).flatten()

    #Exclude all videos with the same title
    for idx in matching_indices:
        cosine_similarities[idx] = -1

    # Get the indexes of the most similar videos
    similar_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    similar_indices = [i for i in similar_indices if i not in matching_indices]

    print(f"\nRecommended videos for '{data['title'][base_idx]}':")
    for i in similar_indices:
        print(f"- {data['title'][i]} (Similarity: {cosine_similarities[i]:.2f})")


def generate_neural_network(data):

    # Selecting relevant columns
    features = ["view_count", "like_count", "comment_count", "country"]
    target = "engagement_score"


    # We eliminate rows with null values
    data_model = data[features + [target]].dropna()


    X = data_model[features]
    y = data_model[target]


    # Preprocessing: numeric scalar and one-hot for country
    numerical_features = ["view_count", "like_count", "comment_count"]
    categorical_features = ["country"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


    # Definition of the neural network model
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)

    # Complete Pipeline 
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", mlp)
    ])

    #  train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trainig
    pipeline.fit(X_train, y_train)

    #Prediction and evaluation
    y_pred = pipeline.predict(X_test)

    return y, y_test, y_pred


def create_experiment(data):
    top_items = data.sort_values(by="engagement_score", ascending=False).head(400000)

    # We prepare the data to recommend channels (video = item, channel = user)
    cornac_data = top_items[["title", "channel_name", "engagement_score"]].copy()
    cornac_data.rename(columns={
        "channel_name": "user",           
        "title": "item",    
        "engagement_score": "rating"
    }, inplace=True)

    # We convert to a list of tuples (video, channel, score)
    cornac_tuples = cornac_data[["user", "item", "rating"]].values.tolist()

    # train/test
    ratio_split = RatioSplit(data=cornac_tuples, test_size=0.1, seed=42, verbose=True)


    K = 50
    iknn_cosine = ItemKNN(k=K, similarity="cosine", name="ItemKNN-Cosine", verbose=True)
    iknn_pearson = ItemKNN(k=K, similarity="pearson", name="ItemKNN-Pearson", verbose=True)
    iknn_pearson_mc = ItemKNN(k=K, similarity="pearson", mean_centered=True, name="ItemKNN-Pearson-MC", verbose=True)
    iknn_adjusted = ItemKNN(k=K, similarity="cosine", mean_centered=True, name="ItemKNN-AdjustedCosine", verbose=True)


    Experiment(
        eval_method=ratio_split,
        models=[iknn_cosine, iknn_pearson, iknn_pearson_mc, iknn_adjusted],
        metrics=[RMSE()]
    ).run()
    
    # Get the model's videos (items)
    videos_entrenados = list(iknn_pearson.train_set.item_ids)

    # Get the channels associated with those videos
    canales_entrenados = data[data["title"].isin(videos_entrenados)]["channel_name"].dropna().unique()


    print("\n Channels present in the recommendation model (based on the trained videos):")
    for canal in canales_entrenados[:20]: 
        print("-", canal)


    print(f"\nTotal unique channels in the model: {len(canales_entrenados)}")

    return iknn_pearson_mc


def recommend_videos_item_based(data, iknn_pearson_mc):
    
    # Channel from which we want to get recommendations
    canal_elegido = "JISOO"  

    # We verify that the channel is in the training set
    if canal_elegido not in iknn_pearson_mc.train_set.uid_map:
        print(f"The channel '{canal_elegido}'is not in the data training")
    else:
        UIDX = iknn_pearson_mc.train_set.uid_map[canal_elegido]
        TOPK = 10

    # We use Cornac's rank function to get recommendations
        recomendaciones, puntuaciones = iknn_pearson_mc.rank(UIDX)

        print(f"\n TOP {TOPK} VIDEOS RECOMMENDED for the video '{canal_elegido}':")
        mostrados = set()
        i = 0
        for item_idx in recomendaciones:
            # We get the channel to which that recommended video belongs
            video = iknn_pearson_mc.train_set.item_ids[item_idx]
            canal_recomendado = data[data["title"] == video]["channel_name"].values[0]

            # We avoid recommending the same channel and duplicates
            if canal_recomendado != canal_elegido and canal_recomendado not in mostrados:
                print(f"{i+1}. {canal_recomendado} — (video: '{video}') — score: {puntuaciones[item_idx]:.2f}")
                mostrados.add(canal_recomendado)
                i += 1
                if i == TOPK:
                    break


def set_seed():
    SEED = 123

    random.seed(SEED)
    np.random.seed(SEED)