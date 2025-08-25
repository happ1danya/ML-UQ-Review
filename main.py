import numpy as np
import os 

from utils import (
    parse_args,
    load_data,
    run_mc_dropout,
    misclassification_probability,
    entropy,
    entropy_mean,
    mean_entropy,
    max_entropy,
    std_predicted_prob,
    compute_truth_flags,
    build_bins_from_arrays,
    summarise_metric_diffs,
    save_results_json,
    plot_all,
)


def main():
    args = parse_args()
    model, X, y = load_data(
        args.model_path,
        args.X_test_path,
        args.y_test_path
    )

    # If u have really limited VRAM u can set --batch_size 64 or others to fit ur GPU
    obs = run_mc_dropout(model, X, args.num_observations, args.batch_size)

    # Predictions and Truth flags for observations
    preds0 = obs[0].argmax(axis=1)
    preds_mc = obs.mean(axis=0).argmax(axis=1)
    flags0 = compute_truth_flags(y, preds0)
    flags_mc = compute_truth_flags(y, preds_mc)


    mp, mp_mean = misclassification_probability(obs)
    ent = entropy(obs)
    mean_ent = mean_entropy(obs)
    ent_mean = entropy_mean(obs)
    max_ent = max_entropy(obs)
    dpp = std_predicted_prob(obs)


    u_lists, m_lists, c_lists = build_bins_from_arrays(
        y,
        mp=mp, mp_mean=mp_mean, ent=ent, mean_ent=mean_ent, ent_mean=ent_mean, max_ent=max_ent, dpp=dpp,
        preds0=preds0, preds_mc=preds_mc,
        flags0=flags0, flags_mc=flags_mc,
        slices=10
    )

    # Summarise per-bin metric differences
    diff_summary = summarise_metric_diffs(u_lists, m_lists)

    # 5.  Plot everything
    output_dir = os.path.join(
        'datasets',
        os.path.basename(os.path.dirname(args.model_path))
    )
    prefix = os.path.splitext(os.path.basename(args.model_path))[0]

    plot_all(u_lists, m_lists, c_lists, output_dir, prefix)
    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    save_results_json(u_lists, m_lists, c_lists, diff_summary, json_path)
    print("Done.")

if __name__ == "__main__":
    main()
