# metrics.py
import numpy as np

def misclassification_probability(obs):
    # obs: (num_obs, N, C)
    baseline = obs[0]
    argmax0 = baseline.argmax(axis=1)
    mp = 1 - baseline[np.arange(len(argmax0)), argmax0]
    mc_mean = obs.mean(axis=0)
    argmax_mc = mc_mean.argmax(axis=1)
    mp_mc = 1 - mc_mean[np.arange(len(argmax_mc)), argmax_mc]
    return mp, mp_mc

def entropy(obs):
    base = obs.shape[2]
    p = obs[0]
    ent = -(p * np.log(p) / np.log(base)).sum(axis=1)
    mc_mean = obs.mean(axis=0)
    ent_mc = -(mc_mean * np.log(mc_mean) / np.log(base)).sum(axis=1)
    return ent, ent_mc

def std_predicted_prob(obs):
    preds_mc = obs.mean(axis=0).argmax(axis=1)
    stds = obs[:, np.arange(obs.shape[1]), preds_mc].std(axis=0)
    # normalize
    return (stds - stds.min()) / (stds.max() - stds.min())
