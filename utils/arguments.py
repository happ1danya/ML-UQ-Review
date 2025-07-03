import argparse
import os

DATASET_DIRS = {
    "avh":   "ai_vs_human",
    "cs":    "credit_score",
    "fm":    "fashion_mnist",
    "mnist": "mnist",
    "neo":   "neo",
    "tweets":"tweets",
    "rt":    "rice_type",
    "sd":    "smoke_detection",
    "gamma": "gamma",
    "fd":    "fraud_detection",
    "lr":    "letter_recognition",
    "ct":    "cover_type",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Load a model and test dataset")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        choices=DATASET_DIRS.keys(),
        help="Abbreviation of a predefined dataset"
    )
    group.add_argument(
        "--model_path",
        type=str,
        help="Full path to the trained model file"
    )

    # If using --model_path, these must also be provided
    parser.add_argument(
        "--X_test_path",
        type=str,
        help="Full path to the X_test dataset (required if --model_path is used)"
    )
    parser.add_argument(
        "--y_test_path",
        type=str,
        help="Full path to the y_test dataset (required if --model_path is used)"
    )

    parser.add_argument(
        "--num_observations",
        type=int,
        default=100,
        help="Number of observations to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of batch size during inference"
    )

    args = parser.parse_args()

    if args.dataset:
        # Build paths from abbreviation
        folder = DATASET_DIRS[args.dataset]
        base_dir = os.path.join("datasets", folder)
        args.model_path  = os.path.join(base_dir, f"{args.dataset}.keras")
        args.X_test_path = os.path.join(base_dir, f"X_test_{args.dataset}.npy")
        args.y_test_path = os.path.join(base_dir, f"y_test_{args.dataset}.npy")
    else:
        missing = []
        if not args.X_test_path:
            missing.append("--X_test_path")
        if not args.y_test_path:
            missing.append("--y_test_path")
        if missing:
            parser.error(
                f"When using --model_path, you must also provide {' and '.join(missing)}"
            )

    return args
