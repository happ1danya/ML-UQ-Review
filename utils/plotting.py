import os

import matplotlib.pyplot as plt
import numpy as np

_DARK_COLS = [
    "#444444",
    "#8B4513",
    "#2E8B57",
    "#483D8B",
    "#556B2F",
    "#8B0000",
    "#006666",
    "#1F4E79",
    "#7F6000",
    "#7030A0",
]
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "p", "X", "h", "v", "<", ">"]
_HATCHES = ["/", "\\", "o", "+", "*", ".", "x", "-", "|", "O"]


def plot_metric(x, lines, labels, xlabel, ylabel, outpath_png, outpath_eps):
    """Draw one uncertainty-versus-performance line plot."""
    plt.figure(figsize=(7.5, 5.2), dpi=100)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xlim(0.1, 1.0)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0.1, 1.1, 0.1), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="gray")
    plt.axhline(y=0.9, color="gray", linestyle="--", linewidth=1)

    for index, (values, label) in enumerate(zip(lines, labels)):
        plt.plot(
            x,
            values,
            color=_DARK_COLS[index % len(_DARK_COLS)],
            linestyle=_LINESTYLES[index % len(_LINESTYLES)],
            marker=_MARKERS[index % len(_MARKERS)],
            markersize=7,
            linewidth=1.8,
            label=label,
        )

    plt.legend(loc="best", fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath_png, bbox_inches="tight")
    plt.savefig(outpath_eps, bbox_inches="tight")
    plt.close()


def plot_all(u_lists, m_lists, counts, output_dir, prefix):
    """Create all performance and count figures for the supplied methods."""
    os.makedirs(output_dir, exist_ok=True)
    order = list(u_lists)
    x_values = np.arange(0.1, 1.1, 0.1)

    plot_metric(
        x_values,
        [u_lists[key] for key in order],
        order,
        "Uncertainty",
        "Accuracy",
        os.path.join(output_dir, f"{prefix}_acc.png"),
        os.path.join(output_dir, f"{prefix}_acc.eps"),
    )

    metric_groups = [
        ("pre", 0, "Macro Precision"),
        ("pre", 1, "Micro Precision"),
        ("pre", 2, "Weighted Precision"),
        ("rec", 3, "Macro Recall"),
        ("rec", 4, "Micro Recall"),
        ("rec", 5, "Weighted Recall"),
        ("f1", 6, "Macro F1 Score"),
        ("f1", 7, "Micro F1 Score"),
        ("f1", 8, "Weighted F1 Score"),
    ]
    suffixes = ["mac", "mic", "wei"]
    for tag, index, ylabel in metric_groups:
        plot_metric(
            x_values,
            [m_lists[key][index] for key in order],
            order,
            "Uncertainty",
            ylabel,
            os.path.join(
                output_dir,
                f"{prefix}_{tag}_{suffixes[index % 3]}.png",
            ),
            os.path.join(
                output_dir,
                f"{prefix}_{tag}_{suffixes[index % 3]}.eps",
            ),
        )

    max_length = len(next(iter(counts.values())))
    x_bar = np.arange(0.1, 0.1 * max_length + 0.1, 0.1)
    total_methods = len(order)
    bar_width = min(0.08 / max(total_methods, 1), 0.015)
    offsets = (
        np.arange(total_methods) - (total_methods - 1) / 2
    ) * bar_width

    plt.figure(figsize=(9, 5.5), dpi=100)
    plt.xlabel("Uncertainty Bins", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.xticks(x_bar, fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, color="gray")

    for index, (method, offset) in enumerate(zip(order, offsets)):
        plt.bar(
            x_bar + offset,
            counts[method],
            width=bar_width,
            color=_DARK_COLS[index % len(_DARK_COLS)],
            hatch=_HATCHES[index % len(_HATCHES)],
            edgecolor="black",
            label=method,
        )

    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    for extension in ("png", "eps"):
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_count.{extension}"),
            format=extension,
            bbox_inches="tight",
        )
    plt.close()
