import numpy as np
from sklearn.cluster import KMeans
from heapq import heappush, heappop
import os
from utils import *

class IVF:
    def __init__(self, n_clusters: int, n_probs: int, dimension: int, data_size: int):
        self.data_size = data_size 
        self.n_clusters = n_clusters
        self.n_probs = n_probs
        self.dimension = dimension
        self.clusters = {i: [] for i in range(n_clusters)}  # Dict to store cluster vectors
        self.centroids = []  # Centroids of the clusters
        self.clusters_file_path = "Data_" + str(data_size) + ".bin"
        self.cluster_start_pos_file_path = "ClusterStartPos_" + str(data_size) + ".bin"
        self.centroids_file_path = "Centroids_" + str(data_size) + ".bin"
        
        folder_name='./Databases'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        size_folder = os.path.join(folder_name, str(data_size))
        if not os.path.exists(size_folder):
            os.makedirs(size_folder)

        self.main_directory_path = size_folder


    def train(self, original_data_path):
        print("Training IVF index...")
        data = read_database(original_data_path,self.data_size,self.dimension)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(data)
        self.centroids = kmeans.cluster_centers_
        write_centroids_file(self.centroids,os.path.join(self.main_directory_path, self.centroids_file_path),self.dimension)

        # Assign vectors to the nearest cluster
        labels = kmeans.predict(data)
        for i, label in enumerate(labels):
            self.clusters[label].append(i)
        write_clusters_file(self.clusters,os.path.join(self.main_directory_path, self.clusters_file_path),os.path.join(self.main_directory_path, self.cluster_start_pos_file_path))

        print("Training complete. Clusters created.")
        return
    
    def retrieve(self, query,top_k):
        self.centroids=read_centroids_file(os.path.join(self.main_directory_path, self.centroids_file_path),self.dimension)
        print(self.centroids,query)

        query_dot_centroids = np.argsort(self.centroids.dot(query.T).T / (np.linalg.norm(self.centroids) * np.linalg.norm(query))).squeeze().tolist()[::-1]
       
        top_scores = query_dot_centroids[:self.n_probs]

        top_k_embeddings = []
        for score in top_scores:
            embeddings = []
            vec_indexes = read_one_cluster(score,os.path.join(self.main_directory_path, self.clusters_file_path),os.path.join(self.main_directory_path, self.cluster_start_pos_file_path),self.n_clusters,self.data_size)
            for id in vec_indexes:
                embeddings.append(read_one_embedding(id))
            query_dot_embedding = np.argsort(embeddings.dot(query.T).T / (np.linalg.norm(embeddings) * np.linalg.norm(query))).squeeze().tolist()[::-1]
            top_k_embeddings.append(query_dot_embedding[:top_k])
        
        return sorted(top_k_embeddings, reverse=True)[:top_k]



    # def retrieve(self, query: np.ndarray, top_k: int, n_probe: int):
    #     # Step 1: Find the closest centroids (clusters)
    #     cluster_distances = [
    #         (self._euclidean_distance(query, centroid), cluster_id)
    #         for cluster_id, centroid in enumerate(self.centroids)
    #     ]
    #     cluster_distances.sort()  # Sort by distance to query
    #     probed_clusters = [cluster_id for _, cluster_id in cluster_distances[:n_probe]]

    #     # Step 2: Collect candidates from the probed clusters
    #     candidates = []
    #     for cluster_id in probed_clusters:
    #         for vector in self.clusters[cluster_id]:
    #             similarity = self._cosine_similarity(query, vector)
    #             heappush(candidates, (-similarity, vector))  # Use max heap for top-k

    #     # Step 3: Extract top_k results
    #     top_k_results = []
    #     while candidates and len(top_k_results) < top_k:
    #         _, vector = heappop(candidates)
    #         top_k_results.append(vector)

    #     return top_k_results

    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray):
        return np.linalg.norm(vec1 - vec2)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)
