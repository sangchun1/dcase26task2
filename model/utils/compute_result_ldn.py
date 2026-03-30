import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def _get_distance_fn(dist):
    if dist == "cosine":
        return cosine_distances
    if dist == "l2":
        return euclidean_distances
    raise ValueError(f"Unsupported dist metric: {dist}. Use 'cosine' or 'l2'.")


def _safe_knn_sum(self_dist, k, eps=1e-8):
    """Sum K nearest neighbor distances excluding self-distance."""
    if self_dist.shape[0] == 0:
        return np.array([], dtype=np.float32)

    # Exclude self-distance (first sorted element ~= 0)
    sorted_dist = np.sort(self_dist, axis=1)[:, 1:]
    k_eff = max(1, min(int(k), sorted_dist.shape[1] if sorted_dist.shape[1] > 0 else 1))

    if sorted_dist.shape[1] == 0:
        # Single-sample bank case
        return np.full((self_dist.shape[0],), eps, dtype=np.float32)

    return (np.sum(sorted_dist[:, :k_eff], axis=1) + eps).astype(np.float32)


def compute_result_ldn(result_train, result_test, ks_kt=(16, 9), dist="cosine", test=False):
    # Data load (기존 compute_result.py와 동일한 경로 스타일 유지)
    eval_train = pd.read_csv("/home/user/PSC/ASD/2026/data/dev_train.csv")
    if not test:
        eval_test = pd.read_csv("/home/user/PSC/ASD/2026/data/dev_test.csv")
        eval_test["label"] = eval_test["label"].map({"normal": 0, "anomaly": 1})
    else:
        eval_test = pd.read_csv("/mnt/storage1/asd/2025_eval/eval.csv")

    # Convert tensor to numpy if needed
    result_train_np = result_train.cpu().numpy() if hasattr(result_train, "cpu") else result_train
    result_test_np = result_test.cpu().numpy() if hasattr(result_test, "cpu") else result_test

    K_s, K_t = int(ks_kt[0]), int(ks_kt[1])
    dist_fn = _get_distance_fn(dist)

    source_refs = {}
    target_refs = {}
    source_denoms = {}
    target_denoms = {}

    for machine in eval_train["machine"].unique():
        machine_mask = eval_train["machine"] == machine
        src_idx = eval_train[machine_mask & (eval_train["domain"] == "source")].index.to_numpy()
        tgt_idx = eval_train[machine_mask & (eval_train["domain"] == "target")].index.to_numpy()

        Fs = result_train_np[src_idx].astype(np.float32)
        Ft = result_train_np[tgt_idx].astype(np.float32)

        source_refs[machine] = Fs
        target_refs[machine] = Ft

        src_self_dist = dist_fn(Fs, Fs) if Fs.shape[0] > 0 else np.zeros((0, 0), dtype=np.float32)
        tgt_self_dist = dist_fn(Ft, Ft) if Ft.shape[0] > 0 else np.zeros((0, 0), dtype=np.float32)

        source_denoms[machine] = _safe_knn_sum(src_self_dist, K_s)
        target_denoms[machine] = _safe_knn_sum(tgt_self_dist, K_t)

    # Compute anomaly scores
    test_scores = np.zeros(len(result_test_np), dtype=np.float32)

    for i in range(len(result_test_np)):
        machine = eval_test["machine"].iloc[i]
        y = result_test_np[i].reshape(1, -1).astype(np.float32)

        # Source score
        Fs = source_refs[machine]
        if Fs.shape[0] == 0:
            s_s = np.inf
        else:
            d_src = dist_fn(y, Fs).reshape(-1)
            s_s = float(np.min(d_src / source_denoms[machine]))

        # Target score
        Ft = target_refs[machine]
        if Ft.shape[0] == 0:
            s_t = np.inf
        else:
            d_tgt = dist_fn(y, Ft).reshape(-1)
            s_t = float(np.min(d_tgt / target_denoms[machine]))

        test_scores[i] = min(s_s, s_t)

    if test:
        return test_scores

    # Metrics (기존 compute_result.py 포맷 유지)
    eval_test["anomaly_score"] = test_scores
    p_aucs = []
    aucs_source = []
    aucs_target = []
    machine_results = {}

    for machine in eval_test["machine"].unique():
        machine_results[machine] = []
        temp = eval_test[eval_test["machine"] == machine]

        # pAUC
        true = temp["label"].values
        pred = temp["anomaly_score"].values
        p_auc = roc_auc_score(true, pred, max_fpr=0.1)
        p_aucs.append(p_auc)
        machine_results[machine].append(p_auc)

        # AUC source
        temp_source = temp[temp["domain"] == "source"]
        true_source = temp_source["label"].values
        pred_source = temp_source["anomaly_score"].values
        auc_source = roc_auc_score(true_source, pred_source)
        aucs_source.append(auc_source)
        machine_results[machine].append(auc_source)

        # AUC target
        temp_target = temp[temp["domain"] == "target"]
        true_target = temp_target["label"].values
        pred_target = temp_target["anomaly_score"].values
        auc_target = roc_auc_score(true_target, pred_target)
        aucs_target.append(auc_target)
        machine_results[machine].append(auc_target)

    mean_p_auc = hmean(p_aucs)
    mean_auc_source = hmean(aucs_source)
    mean_auc_target = hmean(aucs_target)
    score = hmean(aucs_source + aucs_target + p_aucs)

    return machine_results, mean_auc_source, mean_auc_target, mean_p_auc, score
