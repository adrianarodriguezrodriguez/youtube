from cornac import Experiment
from cornac.metrics import RMSE
from cornac.models import ItemKNN
from cornac.eval_methods import RatioSplit
from src.utils import initialize_data, calculate_sentiment

data = initialize_data()

# Calculate Sentiment Score
data = calculate_sentiment(data)

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


# trains the model


# Get the model's videos (items)
videos_entrenados = list(iknn_pearson.train_set.item_ids)

# Get the channels associated with those videos
canales_entrenados = data[data["title"].isin(videos_entrenados)]["channel_name"].dropna().unique()


print("\n Channels present in the recommendation model (based on the trained videos):")
for canal in canales_entrenados[:20]: 
    print("-", canal)


print(f"\nTotal unique channels in the model: {len(canales_entrenados)}")



print("----------- Recommendation -------------")

# Generating a recommendation

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
