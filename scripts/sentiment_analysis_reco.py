from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from src.utils import initialize_data, calculate_sentiment, recommend_videos_sentiment

download('all')


# Now we will focus on analysing and filtering important data on the tags and title columns. With this method, we will be able to identify key words to generate relationships

# First, we retrive data from kaggle
data = initialize_data()

# Calculate Sentiment Score
data = calculate_sentiment(data)


print("Average engagement by sentiment :")
display(data)


plt.figure(figsize=(8, 5))
sns.barplot(data=data, x="sentiment", y="engagement_score", palette="viridis")
plt.title("Average engagement by sentiment type")
plt.ylabel("Average Engagement Score")
plt.xlabel("Sentiment")
plt.show()


recommendations, sampled_features, neighbor_indices = recommend_videos_sentiment(user_sentiment="positive", num_recommendations=5, sample_size=500)
print("Recommended videos:", recommendations)
