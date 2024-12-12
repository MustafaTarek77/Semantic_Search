import numpy as np
from sklearn.cluster import KMeans
import os
from utils import *

class IVF:
    def __init__(self, original_data_path: str, n_clusters: int, n_probs: int, dimension: int, data_size: int):
        self.data_size = data_size 
        self.n_clusters = n_clusters
        self.n_probs = n_probs
        self.dimension = dimension
        self.original_data_path = original_data_path
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


    def train(self):
        print("Training IVF index...")
        data = read_database(self.original_data_path, self.data_size,self.dimension)
        kmeans = KMeans(n_clusters=self.n_clusters)
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
    
    def retrieve(self, query,top_k, index_path=None):
        self.centroids = read_centroids_file(os.path.join(index_path, self.centroids_file_path), self.dimension)        
        query_dot_centroids = np.argsort(self.centroids.dot(query.T).T / (np.linalg.norm(self.centroids) * np.linalg.norm(query))).squeeze().tolist()[::-1]
       
        top_scores = query_dot_centroids[:self.n_probs]

        top_k_embeddings = []
        for score in top_scores:
            vec_indexes_batch_generator = read_one_cluster(score, os.path.join(index_path, self.clusters_file_path), os.path.join(index_path, self.cluster_start_pos_file_path), self.n_clusters,self.data_size)            
            for id in vec_indexes:
                embedding = read_one_embedding(self.original_data_path,id,self.dimension)
                query_dot_embedding = embedding.dot(query.T) / (np.linalg.norm(embedding) * np.linalg.norm(query))
                top_k_embeddings.append((query_dot_embedding,id))
        result = sorted(top_k_embeddings, reverse=True)[:top_k]
        ids = [score[1] for score in result]

        return ids
