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
    predicted_vs_rest_entropy,
    predicted_vs_rest_entropy_deviation,
    eprds,
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
        args.y_test_path,
    )

    # If VRAM is limited, use --batch_size 64 (or another smaller value).
    obs = run_mc_dropout(model, X, args.num_observations, args.batch_size)

    # Predictions and truth flags.
    preds0 = obs[0].argmax(axis=1)
    preds_mc = obs.mean(axis=0).argmax(axis=1)
    flags0 = compute_truth_flags(y, preds0)
    flags_mc = compute_truth_flags(y, preds_mc)

    mp, mp_mean = misclassification_probability(obs)

    uncertainty = {
        "MP": mp,
        "MP_Mean": mp_mean,
        "Entropy": entropy(obs),
        "M_E": mean_entropy(obs),
        "E_M": entropy_mean(obs),
        "Max_E": max_entropy(obs),
        "DPkP": std_predicted_prob(obs),
        "EPR": predicted_vs_rest_entropy(obs),
        "EPRD": predicted_vs_rest_entropy_deviation(obs),
        "EPRDS": eprds(obs),
    }

    # MP uses the first-pass prediction; repeated-observation methods use the
    # stable prediction obtained from the MC-mean distribution.
    prediction_by_method = {
        name: (preds0 if name == "MP" else preds_mc)
        for name in uncertainty
    }
    flags_by_method = {
        name: (flags0 if name == "MP" else flags_mc)
        for name in uncertainty
    }

    u_lists, m_lists, c_lists = build_bins_from_arrays(
        y,
        uncertainty=uncertainty,
        predictions=prediction_by_method,
        truth_flags=flags_by_method,
        slices=10,
    )

    diff_summary = summarise_metric_diffs(u_lists, m_lists)

    output_dir = os.path.join(
        "datasets",
        os.path.basename(os.path.dirname(args.model_path)),
    )
    prefix = os.path.splitext(os.path.basename(args.model_path))[0]

    plot_all(u_lists, m_lists, c_lists, output_dir, prefix)
    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    save_results_json(
        u_lists,
        m_lists,
        c_lists,
        diff_summary,
        json_path,
        uncertainty=uncertainty,
    )
    print("Done.")


if __name__ == "__main__":
    main()
