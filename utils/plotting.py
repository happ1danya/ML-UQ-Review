import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Helper: consistent colour / style palette
# ---------------------------------------------------------------------
_DARK_COLS   = ['#444444', '#8B4513', '#2E8B57', '#483D8B', '#556B2F',
                '#8B0000', '#006666']
_LINESTYLES  = ['-', '--', '-.', ':', '-', '--', '-']
_MARKERS     = ['o', 's', '^', 'D', 'p', 'X', 'h']
_HATCHES     = ['/', '\\', 'o', '+', '*', '.', 'x']


# ---------------------------------------------------------------------
# 1) Generic 2-D line plot
# ---------------------------------------------------------------------
def plot_metric(x, lines, labels, xlabel, ylabel, outpath_png, outpath_eps):
    """
    Draws a single panel (line plot) with the *exact* visual style
    used in the long, hand-written code block, but treats 'Ideal'
    specially (black, dotted, no markers).
    """
    plt.figure(figsize=(6, 4.5), dpi=100)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    # axis limits, ticks, grid
    plt.xlim(0.1, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0.1, 1.1, 0.1), fontsize=15)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')
    plt.axhline(y=0.9, color='gray', linestyle='--', linewidth=1)

    for i, (y, lab) in enumerate(zip(lines, labels)):
        if lab == 'Ideal Line':
            # special styling for the ideal line
            plt.plot(
                x, y,
                color='black',
                linestyle='--',
                linewidth=2,
                label=lab
            )
        else:
            plt.plot(
                x, y,
                color=_DARK_COLS[i % len(_DARK_COLS)],
                linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                marker=_MARKERS[i % len(_MARKERS)],
                markersize=10,
                linewidth=2,
                label=lab
            )

    plt.legend(loc='lower left', fontsize=15, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath_png, bbox_inches='tight')
    plt.savefig(outpath_eps, bbox_inches='tight')
    plt.close()



# ---------------------------------------------------------------------
# 2) Produce *all* figures in one call
# ---------------------------------------------------------------------
def plot_all(u_lists, m_lists, counts, output_dir, prefix):
    """
    Parameters
    ----------
    u_lists : dict[str, (10,)]           – accuracy curves
              keys expected: 'MP', 'MP_Mean', 'Entropy', 'Mean_Entropy', 'Entropy_Mean', 'Max_Entropy', 'DPP'
    m_lists : dict[str, (9, 10)]         – metric curves
              row-order: [Pre-Mac, Pre-Mic, Pre-Wei,
                          Rec-Mac, Rec-Mic, Rec-Wei,
                          F1-Mac,  F1-Mic,  F1-Wei]
    counts  : dict[str, (10,)]           – per-bin instance counts
    output_dir : str
    prefix     : str – file-name prefix without extension
    """
    os.makedirs(output_dir, exist_ok=True)
    x_vals = np.arange(0.1, 1.1, 0.1)          # same bin mid-points

    order = ['MP', 'MP_Mean', 'Entropy', 'Mean_Entropy', 'Entropy_Mean', 'Max_Entropy', 'DPP']

    # Compute ideal reference line: y = [0.95, 0.85, …, 0.05]
    #ideal_y = [0.95 - 0.1 * i for i in range(len(x_vals))]

    # ---- Accuracy (with Ideal) ------------------------------------
    # plot_metric(
    #     x_vals,
    #     [u_lists[k] for k in order] + [ideal_y],
    #     order + ['Ideal Line'],
    #     'Uncertainty',
    #     'Accuracy',
    #     os.path.join(output_dir, f"{prefix}_acc.png"),
    #     os.path.join(output_dir, f"{prefix}_acc.eps")
    # )
    plot_metric(
        x_vals,
        [u_lists[k] for k in order],
        order,
        'Uncertainty',
        'Accuracy',
        os.path.join(output_dir, f"{prefix}_acc.png"),
        os.path.join(output_dir, f"{prefix}_acc.eps")
    )
    plt.close()

    # ---- Precision / Recall / F1 (macro, micro, weighted), all with Ideal -----------
    metric_groups = [
        ('pre', 0, 'Macro Precision'),
        ('pre', 1, 'Micro Precision'),
        ('pre', 2, 'Weighted Precision'),
        ('rec', 3, 'Macro Recall'),
        ('rec', 4, 'Micro Recall'),
        ('rec', 5, 'Weighted Recall'),
        ('f1',  6, 'Macro F1 Score'),
        ('f1',  7, 'Micro F1 Score'),
        ('f1',  8, 'Weighted F1 Score')
    ]

    for tag, idx, ylabel in metric_groups:
        # curves = [m_lists[k][idx] for k in order] + [ideal_y]
        # labels = order + ['Ideal Line']
        curves = [m_lists[k][idx] for k in order]
        labels = order
        plot_metric(
            x_vals,
            curves,
            labels,
            'Uncertainty',
            ylabel,
            os.path.join(output_dir, f"{prefix}_{tag}_{['mac','mic','wei'][idx%3]}.png"),
            os.path.join(output_dir, f"{prefix}_{tag}_{['mac','mic','wei'][idx%3]}.eps")
        )
        plt.close()

    # ---- BAR plot: instance counts (styled like reference) ------------
    max_len   = len(next(iter(counts.values())))
    x_bar     = np.arange(0.1, 0.1 * max_len + 0.1, 0.1)
    bar_width = 0.015  # Same as your reference

    plt.figure(figsize=(8, 5), dpi=100)
    plt.xlabel('Uncertainty Bins', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(x_bar, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')

    # Compute max y
    ymax = max(max(c) for c in counts.values())
    plt.ylim(0, ymax + 1)

    # Plot bars with symmetric spacing
    bars = []
    total_methods = len(order)
    offsets = np.linspace(-2.5, 2.5, total_methods) * bar_width

    for i, (method, offset) in enumerate(zip(order, offsets)):
        bars.append(
            plt.bar(
                x_bar + offset,
                counts[method],
                width=bar_width,
                color=_DARK_COLS[i % len(_DARK_COLS)],
                hatch=_HATCHES[i % len(_HATCHES)],
                edgecolor='black',
                label=method
            )
        )

    # Legend inside top right
    plt.legend(
        handles=bars,
        loc='upper right',
        fontsize=15,
        ncol=2
    )

    # Layout & save
    plt.tight_layout()
    for ext in ['png', 'eps']:
        plt.savefig(os.path.join(output_dir, f"{prefix}_count.{ext}"),
                    format=ext, bbox_inches='tight')
    plt.close()
