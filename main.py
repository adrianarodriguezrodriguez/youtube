import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from scipy.spatial.distance import pdist, squareform

filePath = os.path.join(os.getcwd(), "dataset", "trending_yt_videos_113_countries.csv")
data = pd.read_csv(filePath, sep=',')

columns_useful = ['title', 'channel_name', 'view_count', 'like_count', 'comment_count', 
                  'video_tags', 'kind', 'publish_date', 'langauge']

# Filter the columns that exist in the dataframe
columns_validas = [c for c in columns_useful if c in data.columns]

# Select only the valid columns
data = data[columns_validas]

# Creating the engagement score column
data['engagement_score'] = (data['like_count'] * 2 + data['comment_count'] * 3) / (data['view_count'] + 1)

# Sorting by engagement score in descending order
data_sorted = data.sort_values(by='engagement_score', ascending=False)

# Calculate min and max values of the engagement_score
min_score = data['engagement_score'].min()
max_score = data['engagement_score'].max()

# Create the scaled engagement score column
data['scaled_engagement_score'] = 1 + 9 * ((data['engagement_score'] - min_score) / (max_score - min_score))

# Select the relevant columns and sort by scaled_engagement_score in descending order
data_sorted = data[['title', 'engagement_score', 'scaled_engagement_score']].sort_values(by='scaled_engagement_score', ascending=False)

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

channel_analysis = channel_analysis.sort_values(by="scaled_avg_engagement_score", ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(channel_analysis["channel_name"][:10], channel_analysis["scaled_avg_engagement_score"][:10], color='skyblue')
plt.xlabel("Scaled Average Engagement Score")
plt.ylabel("Channel Name")
plt.title("Top 10 Channels by Scaled Average Engagement Score")
plt.gca().invert_yaxis() 
plt.show()

from scipy.spatial.distance import pdist, squareform

# Select only the numeric columns
numeric_data = data.select_dtypes(include=['number'])

chunk_size = 1000  # Adjust chunk size based on available memory
chunks = [numeric_data.iloc[i:i+chunk_size] for i in range(0, numeric_data.shape[0], chunk_size)]

distance_matrices = []

for chunk in chunks:
    distance_matrix = pdist(chunk, metric='euclidean')
    distance_matrix_square = squareform(distance_matrix)
    distance_matrices.append(distance_matrix_square)

# Now, pad the last chunk if necessary
# Determine the largest matrix size (assumed to be the first chunk size)
max_rows = distance_matrices[0].shape[0]

# Pad all the distance matrices to the same size
padded_distance_matrices = [
    np.pad(matrix, ((0, max_rows - matrix.shape[0]), (0, max_rows - matrix.shape[1])), mode='constant', constant_values=0)
    for matrix in distance_matrices
]

# Concatenate the matrices
full_distance_matrix = np.concatenate(padded_distance_matrices, axis=0)

# Convert to DataFrame for easier manipulation
distance_matrix_df = pd.DataFrame(full_distance_matrix)

distance_matrix_df.head()