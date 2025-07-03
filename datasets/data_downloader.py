"""Utility for downloading Kaggle datasets used in this project."""

import os
import kaggle

DATASETS = {
    "credit_score": "parisrohan/credit-score-classification",
    "tweets": "mexwell/depressivenon-depressive-tweets-data",
    "ai_vs_human": "shanegerami/ai-vs-human-text",
    "neo": "ivansher/nasa-nearest-earth-objects-1910-2024",
    "fraud_detection": "mlg-ulb/creditcardfraud",
    "rice_type": "mssmartypants/rice-type-classification",
    "smoke_detection": "deepcontractor/smoke-detection-dataset"
}


def download_all() -> None:
    """Download each dataset and report any failures."""

    kaggle.api.authenticate()

    for folder, dataset_id in DATASETS.items():
        dest = os.path.join("./datasets", folder)
        try:
            kaggle.api.dataset_download_files(dataset_id, path=dest, unzip=True)
            if not any(os.scandir(dest)):
                raise RuntimeError("download produced an empty directory")
            print(f"Downloaded dataset: {folder} ({dataset_id})")
        except Exception as err:  # noqa: BLE001
            print(f"Failed to download {dataset_id}: {err}")


if __name__ == "__main__":
    download_all()
