# ---------------------------------------------------------------------
#  Binning helpers (vectorised, safe for empty bins)
# ---------------------------------------------------------------------
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def cal_bin(uc, truth_flags, slices=10, return_rate=True):
    """Per-bin accuracy *or* count, NaN for empty bins."""
    edges   = np.linspace(0, 1, slices + 1)
    results = []
    for k in range(slices):
        sel = [truth_flags[i] for i, u in enumerate(uc)
               if edges[k] <= u < edges[k + 1]]
        if return_rate:
            results.append(sel.count('T') / len(sel) if sel else np.nan)
        else:
            results.append(len(sel))
    return np.asarray(results, dtype=float)

def cal_bin_counts(uc, truth_flags, slices=10):
    return cal_bin(uc, truth_flags, slices, return_rate=False)

def grouper(uc, truth, preds, slices=10):
    """Return *lists* of y_true / y_pred arrays, one per bin."""
    edges = np.linspace(0, 1, slices + 1)
    y_tr, y_pr = [], []
    for k in range(slices):
        idx = np.where((uc >= edges[k]) & (uc < edges[k + 1]))[0]
        y_tr.append(truth[idx])
        y_pr.append(preds[idx])
    return y_tr, y_pr

def cal_bin_others(y_true_bins, preds_bins):
    """Nine metrics (P/R/F1 × macro/micro/weighted) for every bin."""
    out = np.full((9, len(y_true_bins)), np.nan)          # default NaN
    for i in range(len(y_true_bins)):
        if len(y_true_bins[i]) == 0:
            continue
        out[0, i] = precision_score(y_true_bins[i], preds_bins[i], average='macro')
        out[1, i] = precision_score(y_true_bins[i], preds_bins[i], average='micro')
        out[2, i] = precision_score(y_true_bins[i], preds_bins[i], average='weighted')
        out[3, i] = recall_score   (y_true_bins[i], preds_bins[i], average='macro')
        out[4, i] = recall_score   (y_true_bins[i], preds_bins[i], average='micro')
        out[5, i] = recall_score   (y_true_bins[i], preds_bins[i], average='weighted')
        out[6, i] = f1_score       (y_true_bins[i], preds_bins[i], average='macro')
        out[7, i] = f1_score       (y_true_bins[i], preds_bins[i], average='micro')
        out[8, i] = f1_score       (y_true_bins[i], preds_bins[i], average='weighted')
    return out
# ---------------------------------------------------------------------
#  One-stop function that prepares EVERYTHING for plot_all()
# ---------------------------------------------------------------------
# utils.py  (append near the other helpers)
# -------------------------------------------
def build_bins_from_arrays(
    y_onehot,
    *,
    mp, mp_mc, ent, ent_mc, dpp,
    preds0, preds_mc,
    flags0, flags_mc,
    slices: int = 10
):
    """
    Consume the already-computed uncertainty arrays + flags/preds
    and return the three dicts that `plot_all()` expects.
    """
    # --- accuracy curves & instance counts --------------------------
    u_lists = {
        'MP'        : cal_bin(mp,        flags0, slices),
        'MP_MC'     : cal_bin(mp_mc,     flags_mc, slices),
        'Entropy'   : cal_bin(ent,       flags0, slices),
        'Entropy_MC': cal_bin(ent_mc,    flags_mc, slices),
        'DPP'       : cal_bin(dpp,       flags_mc, slices),
    }
    c_lists = {
        'MP'        : cal_bin_counts(mp,        flags0, slices),
        'MP_MC'     : cal_bin_counts(mp_mc,     flags_mc, slices),
        'Entropy'   : cal_bin_counts(ent,       flags0, slices),
        'Entropy_MC': cal_bin_counts(ent_mc,    flags_mc, slices),
        'DPP'       : cal_bin_counts(dpp,       flags_mc, slices),
    }

    # --- precision / recall / F1 curves -----------------------------
    gt_labels = y_onehot.argmax(axis=1)           # shape (N,)
    m_lists   = {}
    for name, (u_arr, preds_arr) in {
        'MP'        : (mp,        preds0),
        'MP_MC'     : (mp_mc,     preds_mc),
        'Entropy'   : (ent,       preds0),
        'Entropy_MC': (ent_mc,    preds_mc),
        'DPP'       : (dpp,       preds_mc)
    }.items():
        y_true_bins, y_pred_bins = grouper(u_arr, gt_labels, preds_arr, slices)
        m_lists[name] = cal_bin_others(y_true_bins, y_pred_bins)

    return u_lists, m_lists, c_lists

