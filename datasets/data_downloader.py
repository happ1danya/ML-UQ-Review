import kaggle

# Download dataset (replace with the dataset name)
cs = "parisrohan/credit-score-classification"
kaggle.api.dataset_download_files(cs, path="./datasets/credit_score", unzip=True)

print(f"Downloaded dataset: Credit Score")

tweets = "mexwell/depressivenon-depressive-tweets-data"
kaggle.api.dataset_download_files(tweets, path="./datasets/tweets", unzip=True)

print(f"Downloaded dataset: Tweets")

ah = "shanegerami/ai-vs-human-text"
kaggle.api.dataset_download_files(ah, path="./datasets/ai_vs_human", unzip=True)

print("Downloaded dataset: AI vs. Human")

neo = "ivansher/nasa-nearest-earth-objects-1910-2024"
kaggle.api.dataset_download_files(neo, path="./datasets/neo", unzip=True)

print("Downloaded dataset: NEO")
