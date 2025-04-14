import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_samples


def sil_score_plot(sil_score, final_features, final_labels, num_queries):

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

def canales_clusters(canales_cluster):

    sns.countplot(data=canales_cluster, x="cluster")
    plt.title("Number of channels per cluster")
    plt.show()


def show_cluster_tags(canales_cluster, tags_por_canal):

    # Show the type of tags for each cluster
    for cluster_num in range(5):
        print(f"\nCluster {cluster_num}:")
        # We get the channels that belong to this cluster
        cluster_channels = canales_cluster[canales_cluster["cluster"] == cluster_num]
        # We get the tags from those channels
        cluster_tags = tags_por_canal[cluster_channels["channel_name"]].tolist()
        # We join all the tags in a list and display them
        all_tags = ' '.join(cluster_tags).split()
        #We get the most frequent tags in this cluster
        tags_freq = pd.Series(all_tags).value_counts().head(10)
        print(tags_freq)


def pca_barplot(explained_variance):

    # Create a barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[f'PC{i+1}' for i in range(len(explained_variance))], y=explained_variance)
    plt.title('Variance Explained by Each Principal Component (PCA)')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.show()


def top_10_channels(data):

    plt.figure(figsize=(12, 6))
    plt.barh(data["channel_name"][:10], data["scaled_avg_engagement_score"][:10], color='skyblue')
    plt.xlabel("Scaled Average Engagement Score")
    plt.ylabel("Channel Name")
    plt.title("Top 10 Channels by Scaled Average Engagement Score")
    plt.gca().invert_yaxis() 
    plt.show()


def map_data(data):
    
    # Create the map with percentages
    fig = px.choropleth(
        data,
        locations="country",
        locationmode="ISO-3",
        color="percentage",
        color_continuous_scale="Blues",
        title="Engagement Percentage by Country",
        hover_name="country",
        hover_data={"percentage": True, "country": False}, 
    )

    fig.update_layout(coloraxis_colorbar=dict(title="% Engagement"))


    fig.show()


def plot_thematic_clusters(us_canales):

    plt.figure(figsize=(8,5))
    sns.countplot(data=us_canales, x="cluster", palette="Set2")

    plt.title("Number of channels per cluster in the US")
    plt.xlabel("Thematic cluster")
    plt.ylabel("Number of clusters")
    plt.show()


def plot_monthly_engagement(data):
    
    # Plot with logarithmic scale on the Y axis
    plt.figure(figsize=(12, 5))
    plt.plot(data["publish_month"], data["engagement_score"], marker="o")
    plt.yscale("log") 
    plt.xticks(rotation=45)
    plt.title("Average Engagement per Month of Publication (Log Scale)")
    plt.xlabel("Publication Month")
    plt.ylabel("Average Engagement (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_neural_network_results(y, y_test, y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Real Engagement")
    plt.ylabel("Predicted Engagement")
    plt.title(" Comparison between Actual and Predicted Values")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Ideal line
    plt.grid(True)
    plt.show()

def plot_average_sentiment(data):
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=data, x="sentiment", y="engagement_score", palette="viridis")
    plt.title("Average engagement by sentiment type")
    plt.ylabel("Average Engagement Score")
    plt.xlabel("Sentiment")
    plt.show()