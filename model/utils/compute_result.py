import numpy as np
import pandas as pd
from scipy.stats import hmean
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

def compute_result(result_train, result_test, sampling_strategy, method='kmeans', test=False):
    # Data load
    eval_train = pd.read_csv('/home/user/PSC/ASD/2026/data/dev_train.csv')
    
    if not test:
        eval_test = pd.read_csv('/home/user/PSC/ASD/2026/data/dev_test.csv')
        eval_test['label'] = eval_test['label'].map({'normal': 0, 'anomaly': 1})
    else:
        eval_test = pd.read_csv('/mnt/storage1/asd/2025_eval/eval.csv')
    
    # Convert CUDA tensor to numpy if necessary
    result_train_np = result_train.cpu().numpy() if hasattr(result_train, 'cpu') else result_train
    result_test_np = result_test.cpu().numpy() if hasattr(result_test, 'cpu') else result_test

    # KMeans Method
    if method == 'kmeans':
        source_centers = {}
        target_refs = {}
        
        for machine in eval_train['machine'].unique():
            # Split result_train by domain(source/target for each machine type)
            machine_mask = eval_train['machine'] == machine
            source_mask = machine_mask & (eval_train['domain'] == 'source')
            target_mask = machine_mask & (eval_train['domain'] == 'target')

            source_indices = eval_train[source_mask].index.to_numpy()
            target_indices = eval_train[target_mask].index.to_numpy()

            machine_source_emb = result_train_np[source_indices]
            machine_target_emb = result_train_np[target_indices]

            # Run KMeans on source domain embeddings of each machine type
            k = min(sampling_strategy, len(machine_source_emb))

            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(machine_source_emb)
            centers = kmeans.cluster_centers_

            source_centers[machine] = centers.astype(np.float32)

            # Store all 10 target normal embeddings
            target_refs[machine] = machine_target_emb.astype(np.float32)


        # Compute anomaly scores
        test_scores = np.zeros(len(result_test_np))

        for i in range(len(result_test_np)):
            machine = eval_test['machine'].iloc[i]
            test_embedding = result_test_np[i].reshape(1, -1).astype(np.float32)

            # Compute cosine distance to source centers
            src_centers = source_centers[machine]
            src_dist = cosine_distances(test_embedding, src_centers)
            src_min = float(np.min(src_dist))

            # Compute cosine distance to target refs
            refs = target_refs[machine]
            tgt_dist = cosine_distances(test_embedding, refs)
            tgt_min = float(np.min(tgt_dist))

            # Return the smallest distance
            test_scores[i] = min(src_min, tgt_min)
            

    # SMOTE Method
    elif method == 'smote':
        # Normalize result_train per machine and store statistics
        machine_stats = {}
        result_train_normalized = np.zeros_like(result_train_np)
        for machine in eval_train['machine'].unique():
            train_indices = np.array(eval_train[eval_train['machine'] == machine].index)
            machine_train = result_train_np[train_indices]
            mean = machine_train.mean(axis=0)
            std = machine_train.std(axis=0) + 1e-8  # Add small epsilon to avoid division by zero
            machine_stats[machine] = {'mean': mean, 'std': std}
            result_train_normalized[train_indices] = (machine_train - mean) / std

        # SMOTE on normalized result_train (single application)
        result_train_whole = []
        domain_train_whole = []
        machine_train_indices = {}  # Track indices in result_train_whole for each machine
        start_idx = 0
        for machine in eval_train['machine'].unique():
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            train_indices = np.array(eval_train[eval_train['machine'] == machine].index)
            result_train_res, domain_train = smote.fit_resample(
                result_train_normalized[train_indices],
                eval_train[eval_train['machine'] == machine]['domain'].values
            )
            result_train_whole.append(result_train_res)
            domain_train_whole.append(domain_train)
            
            # Store indices for this machine's SMOTE results
            smote_samples = result_train_res.shape[0]
            machine_train_indices[machine] = np.arange(start_idx, start_idx + smote_samples)
            start_idx += smote_samples

        result_train_whole = np.concatenate(result_train_whole, axis=0)
        domain_train_whole = np.concatenate(domain_train_whole, axis=0)

        # KNN per machine
        test_scores = np.zeros(len(result_test_np))
        knn_models = {}
        for machine in eval_train['machine'].unique():
            # Fit KNN for this machine
            knn = NearestNeighbors(n_neighbors=1, metric='cosine')
            knn.fit(result_train_whole[machine_train_indices[machine]])
            knn_models[machine] = knn

        # Compute anomaly scores for result_test
        for i in range(len(result_test_np)):
            machine = eval_test['machine'].iloc[i]
            # Normalize test embedding using the machine's mean and std
            test_embedding = (result_test_np[i] - machine_stats[machine]['mean']) / machine_stats[machine]['std']
            # Compute distance using the machine's KNN model
            distance, _ = knn_models[machine].kneighbors([test_embedding])
            test_scores[i] = distance[0, 0]
    
    
    # Return values
    if test:
        return test_scores
    else:
        # Compute metrics
        eval_test['anomaly_score'] = test_scores
        p_aucs = []
        aucs_source = []
        aucs_target = []
        machine_results = {}

        machine_list = eval_test['machine'].unique()
        for machine in machine_list:
            machine_results[machine] = []
            # pAUC
            temp = eval_test[eval_test['machine'] == machine]
            true = temp['label'].values
            cos = temp['anomaly_score'].values
            p_auc = roc_auc_score(true, cos, max_fpr=0.1)
            p_aucs.append(p_auc)
            machine_results[machine].append(p_auc)
            
            # AUCSource    
            temp_source = temp[temp['domain'] == 'source']
            true_source = temp_source['label'].values
            cos_source = temp_source['anomaly_score'].values
            auc_score = roc_auc_score(true_source, cos_source)
            aucs_source.append(auc_score)
            machine_results[machine].append(auc_score)
            
            # AUCTarget            
            temp_target = temp[temp['domain'] == 'target']
            true_target = temp_target['label'].values
            cos_target = temp_target['anomaly_score'].values
            auc_score = roc_auc_score(true_target, cos_target)
            aucs_target.append(auc_score)
            machine_results[machine].append(auc_score)

        mean_p_auc = hmean(p_aucs)
        mean_auc_source = hmean(aucs_source)
        mean_auc_target = hmean(aucs_target)
        score = hmean(aucs_source + aucs_target + p_aucs)
        
        return machine_results, mean_auc_source, mean_auc_target, mean_p_auc, score