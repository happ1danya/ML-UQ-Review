# ML-UQ-Review

Review and performance evaluation of uncertainty quantification in machine learning assisted measurements.

## Environment setup
1. Create and activate a Python virtual environment. One example using `venv`:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Downloading datasets
The project uses Kaggle to obtain benchmark datasets. To download them:
1. Install and configure the Kaggle command line interface. Follow the [Kaggle API instructions](https://www.kaggle.com/docs/api) to place your `kaggle.json` credentials in `~/.kaggle/`.
2. Run the downloader script:
   ```bash
   python datasets/data_downloader.py
   ```
   Each dataset will be extracted under `datasets/<name>`.

## Training models
After the datasets are downloaded, train the corresponding models:

```bash
bash train.sh
```

This script iterates over every dataset and stores the trained model in its
respective folder. Once training is complete, you can evaluate a dataset with
`main.py` as described below.

## Running `main.py`
`main.py` evaluates a trained model using MC Dropout and generates plots summarising uncertainty metrics.

Two ways to run it:
- Using a predefined dataset abbreviation:
  ```bash
  python main.py --dataset mnist
  ```
  The script will automatically locate the model and test arrays inside `datasets/mnist/`.
- Explicit paths to a model and test arrays:
  ```bash
  python main.py --model_path path/to/model.keras \
                 --X_test_path path/to/X_test.npy \
                 --y_test_path path/to/y_test.npy
  ```

Output plots (PNG and EPS) are written to the corresponding dataset folder.
In the same directory, a `*_results.json` file captures the arrays used to
generate the figures. It stores the uncertainty bins (`u_lists`, `m_lists`,
`c_lists`) and summaries of how each method differs from MP across ten
classification metrics.

## Evaluating all datasets
To run `main.py` for every dataset in a single command, use the helper script:

```bash
bash evaluate.sh
```

This will sequentially evaluate each dataset using its trained model.


## Directory overview
- `datasets/` – dataset downloader and downloaded data. After running the downloader, subfolders such as `mnist` or `credit_score` will appear here. Evaluation results from `main.py` are also saved in these folders.
- `models/` – pre-trained `.keras` models and example notebooks used during model development.
- `utils/` – helper modules for argument parsing, data loading, inference, metrics and plotting.
- `main.py` – entry point that performs MC Dropout inference and creates figures.
- `requirements.txt` – Python package dependencies.

Running `main.py` generates accuracy, precision/recall/F1, and count plots describing model uncertainty. Files are saved in the dataset directory used for evaluation.
