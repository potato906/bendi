
class AISDataset(Dataset):
    def __init__(self, l_data, max_seqlen=96, dtype=torch.float32, device=torch.device("cpu")):
        self.max_seqlen = max_seqlen  
        self.device = device          
        self.l_data = l_data          
        
        self.alpha = 0.7             
        self.lambda_dtw = 0.1         
        
        self.prev_cluster_centers = None  
        self.kshape_clusters = self.perform_kshape_clustering()  

    def estimate_density_and_select_centers(self, data, n_clusters):
        data_flattened = data.reshape(-1, data.shape[-1])
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data_flattened)
        densities = kde.score_samples(data_flattened)  
        top_density_indices = np.argsort(densities)[-n_clusters:]
        initial_centers = data_flattened[top_density_indices].reshape(
            n_clusters, data.shape[1], data.shape[2]
        )
        return initial_centers
    def perform_kshape_clustering(self):
        all_trajectories = [
            item["traj"][:, :2] for item in self.l_data 
            if "traj" in item and isinstance(item["traj"], np.ndarray)
        ]
        all_trajectories = [
            traj[:self.max_seqlen] if len(traj) > self.max_seqlen 
            else np.pad(traj, ((0, self.max_seqlen - len(traj)), (0, 0)), 'constant') 
            for traj in all_trajectories
        ]
        all_trajectories = np.array(all_trajectories)  # 形状：(n_traj, max_seqlen, 2)

        def dynamic_time_warping_entropy(traj1, traj2):         
            n, m = len(traj1), len(traj2)
            DTW = np.zeros((n + 1, m + 1))  
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = np.linalg.norm(traj1[i - 1] - traj2[j - 1])
                    DTW[i, j] = cost + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])
            dtw_distance = DTW[n, m]  

            def trajectory_entropy(traj):             
                if len(traj) <= 1:
                    return 0.0  
                displacements = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
                displacements_norm = displacements / (displacements.sum() + 1e-10)  
                entropy = -np.sum(displacements_norm * np.log2(displacements_norm + 1e-10))
                return entropy

            entropy1 = trajectory_entropy(traj1)
            entropy2 = trajectory_entropy(traj2)
            avg_entropy = (entropy1 + entropy2) / 2  
            dtw_entropy = dtw_distance + self.lambda_dtw * avg_entropy  

            return dtw_entropy

        n_clusters = 5  
        if self.prev_cluster_centers is not None:
            kshape = KShape(n_clusters=n_clusters, max_iter=10, init=self.prev_cluster_centers)
        else:
            initial_centers = self.estimate_density_and_select_centers(all_trajectories, n_clusters)
            kshape = KShape(n_clusters=n_clusters, max_iter=10, init=initial_centers)


        if self.prev_cluster_centers is not None:
            self.prev_cluster_centers = self.alpha * self.prev_cluster_centers + (1 - self.alpha) * current_cluster_centers
        else:
            self.prev_cluster_centers = current_cluster_centers

        return cluster_labels


