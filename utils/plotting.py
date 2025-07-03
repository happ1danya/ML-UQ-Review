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
def plot_metric(x, lines, labels, xlabel, ylabel, outpath):
    """
    Draws a single panel (line plot) with the *exact* visual style
    used in the long, hand-written code block.

    Parameters
    ----------
    x        : (N,) 1-D array – bin centroids (0.1 … 1.0)
    lines    : list of (N,) arrays – one array per candidate method
    labels   : list of str        – legend entries (same order as *lines*)
    xlabel   : str
    ylabel   : str
    outpath  : str – full pathname (extension decides format)
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

    # draw every method with a dedicated colour / linestyle / marker
    for i, (y, lab) in enumerate(zip(lines, labels)):
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
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------
# 2) Produce *all* figures in one call
# ---------------------------------------------------------------------
def plot_all(u_lists, m_lists, counts, output_dir, prefix):
    """
    Parameters
    ----------
    u_lists : dict[str, (10,)]           – accuracy curves
              keys expected: 'MP', 'MP_MC', 'Entropy', 'Entropy_MC', 'DPP'
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

    order = ['MP', 'MP_MC', 'Entropy', 'Entropy_MC', 'DPP']

    # ---- Accuracy ----------------------------------------------------
    plot_metric(
        x_vals,
        [u_lists[k] for k in order],
        order,
        'Uncertainty',
        'Accuracy',
        os.path.join(output_dir, f"{prefix}_acc.png")
    )
    # EPS copy (unchanged style)
    plt.imread(os.path.join(output_dir, f"{prefix}_acc.png"))  # preload (matplotlib quirk)
    plt.savefig(os.path.join(output_dir, f"{prefix}_acc.eps"), format='eps',
                bbox_inches='tight')
    plt.close()

    # ---- Precision / Recall / F1 (macro, micro, weighted) -----------
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
        plot_metric(
            x_vals,
            [m_lists[k][idx] for k in order],
            order,
            'Uncertainty',
            ylabel,
            os.path.join(output_dir, f"{prefix}_{tag}_{['mac','mic','wei'][idx%3]}.png")
        )
        # matching EPS
        plt.imread(os.path.join(output_dir,
                                f"{prefix}_{tag}_{['mac','mic','wei'][idx%3]}.png"))
        plt.savefig(os.path.join(output_dir,
                                 f"{prefix}_{tag}_{['mac','mic','wei'][idx%3]}.eps"),
                    format='eps', bbox_inches='tight')
        plt.close()

    # ---- BAR plot: instance counts ----------------------------------
    max_len   = len(next(iter(counts.values())))
    x_bar     = np.arange(0.1, 0.1 * max_len + 0.1, 0.1)
    bar_width = 0.015

    plt.figure(figsize=(6, 4.5), dpi=100)
    plt.xlabel('Uncertainty Bins', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(x_bar, fontsize=15, rotation=45)
    plt.yticks(fontsize=15)
    ymax = max(max(v) for v in counts.values()) + 1
    plt.ylim(0, ymax)
    plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')

    bars = []
    for i, k in enumerate(order):
        shift = (i - 2) * bar_width         # centre the stacks
        bars.append(
            plt.bar(
                x_bar + shift,
                counts[k],
                width=bar_width,
                color=_DARK_COLS[i],
                hatch=_HATCHES[i],
                edgecolor='black',
                label=k
            )
        )

    plt.legend(
        handles=bars,
        loc='upper right',
        fontsize=12,
        ncol=3,
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, f"{prefix}_count.png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{prefix}_count.eps"), format='eps',
                bbox_inches='tight')
    plt.close()
