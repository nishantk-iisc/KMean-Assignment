import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # # otherwise you might see that none of the pixels are assigned to some
        # # of the clusters, which will result in a division by zero error)
        n_samples, n_features = X.shape
        idx = np.random.choice(n_samples, size=self.num_clusters, replace=False)
        self.cluster_centers = X[idx]

        for _ in range(max_iter):
            # Assign each sample to the closest prototype
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)
        
            # Update prototypes
            new_centers = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else self.cluster_centers[i] for i in range(self.num_clusters)])
        
            # Check for convergence
            if np.linalg.norm(new_centers - self.cluster_centers) < self.epsilon:
                break
        
            self.cluster_centers = new_centers

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
        # raise NotImplementedError
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
        labels = np.argmin(distances, axis=1)
        cluster_centers = self.cluster_centers[labels]
        return cluster_centers
        # raise NotImplementedError