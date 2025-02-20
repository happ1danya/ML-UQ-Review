import kaggle

# Download dataset (replace with the dataset name)
cs = "parisrohan/credit-score-classification"
kaggle.api.dataset_download_files(cs, path="./Credit_Score", unzip=True)

print(f"Downloaded dataset: Credit Score")

tweets = "mexwell/depressivenon-depressive-tweets-data"
kaggle.api.dataset_download_files(tweets, path="./Tweets", unzip=True)

print(f"Downloaded dataset: Tweets")

ah = "shanegerami/ai-vs-human-text"
kaggle.api.dataset_download_files(ah, path="./AI_vs_Human", unzip=True)

print("Downloaded dataset: AI vs. Human")

neo = "ivansher/nasa-nearest-earth-objects-1910-2024"
kaggle.api.dataset_download_files(neo, path="./NEO", unzip=True)

print("Downloaded dataset: NEO")
