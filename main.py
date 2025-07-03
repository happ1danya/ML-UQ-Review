import numpy as np
import os 

from utils import (
    parse_args,
    load_data,
    run_mc_dropout,
    misclassification_probability,
    entropy,
    std_predicted_prob,
    compute_truth_flags,
    build_bins_from_arrays,
    compute_overall_metrics,
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


    mp, mp_mc = misclassification_probability(obs)
    ent, ent_mc = entropy(obs)
    dpp = std_predicted_prob(obs)


    u_lists, m_lists, c_lists = build_bins_from_arrays(
        y,
        mp=mp, mp_mc=mp_mc, ent=ent, ent_mc=ent_mc, dpp=dpp,
        preds0=preds0, preds_mc=preds_mc,
        flags0=flags0, flags_mc=flags_mc,
        slices=10
    )

    # Compute overall metrics and store exact arrays
    metrics = compute_overall_metrics(y, preds0, preds_mc)
    diff_summary = summarise_metric_diffs(u_lists, m_lists)

    # 5.  Plot everything
    output_dir = os.path.join(
        'datasets',
        os.path.basename(os.path.dirname(args.model_path))
    )
    prefix = os.path.splitext(os.path.basename(args.model_path))[0]

    plot_all(u_lists, m_lists, c_lists, output_dir, prefix)
    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    save_results_json(u_lists, m_lists, c_lists, metrics, diff_summary, json_path)
    print("Done.")

if __name__ == "__main__":
    main()
