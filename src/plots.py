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