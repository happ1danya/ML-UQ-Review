https://ieeexplore.ieee.org/document/11204481

# ML-UQ-Review with EPRDS Extension

Review and performance evaluation of uncertainty quantification in machine-learning-assisted measurements. The `eprds-extension` branch adds the complete uncertainty-method set used in the EPRDS study while retaining the original dataset, training, MC-dropout, evaluation, and plotting workflow.

## Environment setup

1. Create and activate a Conda environment:

   ```bash
   conda create -n mluq python=3.10
   conda activate mluq
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Downloading datasets

The project uses Kaggle to obtain benchmark datasets.

1. Install and configure the Kaggle command-line interface. Place `kaggle.json` in `~/.kaggle/`.
2. Run:

   ```bash
   python datasets/data_downloader.py
   ```

Each dataset is extracted under `datasets/<name>`.

## Training models

```bash
bash train.sh
```

The script iterates over the configured datasets and stores each trained model in its corresponding dataset folder.

## Running the experiments

Using a predefined dataset abbreviation:

```bash
python main.py --dataset mnist
```

Using explicit paths:

```bash
python main.py \
  --model_path path/to/model.keras \
  --X_test_path path/to/X_test.npy \
  --y_test_path path/to/y_test.npy
```

To evaluate all configured datasets:

```bash
bash evaluate.sh
```

The evaluation uses MC dropout, calculates every uncertainty method listed below, builds uncertainty bins, evaluates accuracy/precision/recall/F1, and writes PNG, EPS, and JSON outputs to the corresponding dataset folder.

## Implemented uncertainty methods

| Code label | Paper notation / meaning | Prediction used |
|---|---|---|
| `MP` | Misclassification probability from the first pass | First-pass prediction |
| `MP_Mean` | Misclassification probability from the mean predictive distribution | MC-mean prediction |
| `Entropy` | Normalized categorical entropy from the first pass | First-pass prediction |
| `M_E` | Mean of entropy over repeated observations | MC-mean prediction |
| `E_M` | Entropy of the mean predictive distribution | MC-mean prediction |
| `Max_E` | Maximum entropy among repeated observations | MC-mean prediction |
| `DPkP` | Deviation of the probability of the stable predicted class; also called DPP in the earlier code | MC-mean prediction |
| `EPR` | Mean entropy of predicted class versus all remaining classes | MC-mean prediction |
| `EPRD` | Deviation of predicted-versus-rest entropy | MC-mean prediction |
| `EPRDS` | Mean predicted-versus-rest entropy multiplied by normalized entropy deviation | MC-mean prediction |

For `DPkP`, `EPR`, `EPRD`, and `EPRDS`, the important class is selected once as the class with the highest probability in the mean MC-dropout predictive distribution. The same class is then tracked across all stochastic observations.

### EPRDS calculation

For sample `i`, let `k_i` be the stable predicted class. For observation `t`, its predicted-versus-rest distribution is:

```text
[p_t(k_i), 1 - p_t(k_i)]
```

The binary entropy of this distribution is calculated for every observation. EPRDS is implemented as:

```text
EPRDS_i = mean_t(EPR_t,i) * minmax_i(std_t(EPR_t,i))
```

The deviation term is normalized across the evaluated dataset so the final values remain on the uncertainty scale used by the original experiment pipeline.

The implementation is in `utils/uncertainty_quantification.py`. Paper-style aliases such as `M_E`, `E_M`, `Max_E`, `DPkP`, and `EPRDS` are exported directly.

## Running numerical tests

The tests do not require TensorFlow models or downloaded datasets:

```bash
python -m unittest discover -s tests -v
```

The tests verify output shape and range, the EPRDS composition rule, zero deviation for identical stochastic observations, binary-class consistency, and input validation.

## Output files

For each model, the dataset directory receives:

- accuracy, precision, recall, F1, and uncertainty-count plots in PNG and EPS;
- `<model>_results.json`, containing:
  - raw uncertainty values for every method;
  - binned accuracy curves (`u_lists`);
  - binned precision/recall/F1 curves (`m_lists`);
  - bin counts (`c_lists`);
  - differences relative to MP (`diff_summary`).

## Directory overview

- `datasets/` — dataset downloader, downloaded data, trained models, and experiment outputs.
- `models/` — model-development notebooks and supporting model files.
- `utils/uncertainty_quantification.py` — baseline and EPRDS uncertainty methods.
- `utils/metrics.py` — generic binning and evaluation for any method dictionary.
- `utils/plotting.py` — dynamic plotting for all enabled methods.
- `tests/` — numerical unit tests for the uncertainty implementations.
- `main.py` — MC-dropout evaluation entry point.
- `requirements.txt` — Python dependencies.
