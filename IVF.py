import numpy as np
from sklearn.cluster import KMeans
import os
from utils import *
import heapq
import gc

class IVF:
    def __init__(self, original_data_path: str, n_clusters: int, n_probs: int, dimension: int, data_size: int):
        self.data_size = data_size 
        self.n_clusters = n_clusters
        self.n_probs = n_probs
        self.dimension = dimension
        self.original_data_path = original_data_path
        self.clusters = {i: [] for i in range(n_clusters)}  # Dict to store cluster vectors
        self.centroids = []  # Centroids of the clusters
        self.clusters_file_path = "Data.bin"
        self.cluster_start_pos_file_path = "ClusterStartPos.bin"
        self.centroids_file_path = "Centroids.bin"
        
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
        
    def retrieve(self, query, top_k, index_path=None, batch_size=None):
        if self.data_size == 20*10**6 or self.data_size == 10**6:
            # centroids_generator = read_khra_centroids_file(os.path.join(index_path, self.centroids_file_path), self.dimension,1000)
            # self.centroids = np.vstack([batch for batch in centroids_generator]) 

            self.centroids= read_whole_centroids_file(os.path.join(index_path, self.centroids_file_path), self.dimension)  

            # centroids_generator = read_centroids_file(os.path.join(index_path, self.centroids_file_path), self.dimension)
            # self.centroids = np.array([centroid for centroid in centroids_generator]) # Load centroids
        else:
            centroids_generator = read_centroids_file(os.path.join(index_path, self.centroids_file_path), self.dimension)
            self.centroids = np.array([centroid for centroid in centroids_generator]) # Load centroids

        # Compute similarities with centroids
        query_dot_centroids = np.argsort(
            self.centroids.dot(query.T).T / (np.linalg.norm(self.centroids, axis=1) * np.linalg.norm(query))
        ).squeeze().tolist()[::-1]

        top_scores = query_dot_centroids[:self.n_probs]
        
        # Use a min-heap to store only the top-k embeddings
        heap = []

        for score in top_scores:
            if self.data_size == 20*10**6:
                vec_indexes = list(read_whole_cluster(
                    score,
                    os.path.join(index_path, self.clusters_file_path),
                    os.path.join(index_path, self.cluster_start_pos_file_path),
                    self.n_clusters,
                    self.data_size
                ))
            else:
                vec_indexes = list(read_one_cluster(
                    score,
                    os.path.join(index_path, self.clusters_file_path),
                    os.path.join(index_path, self.cluster_start_pos_file_path),
                    self.n_clusters,
                    self.data_size
                ))

            # Process in batches
            for i in range(0, len(vec_indexes), min(batch_size,len(vec_indexes))):
                batch = vec_indexes[i:i+batch_size]
                embeddings = []
                embeddings = read_embeddings(self.original_data_path, batch, self.dimension)
                del vec_indexes
                # Compute similarity for the batch
                for embedding, id in embeddings:
                    query_dot_embedding = embedding.dot(query.T) / (
                        np.linalg.norm(embedding) * np.linalg.norm(query)
                    )

                    # Push to heap
                    if len(heap) < top_k:
                        heapq.heappush(heap, (query_dot_embedding, id))
                    else:
                        heapq.heappushpop(heap, (query_dot_embedding, id))
                    

        # Extract sorted results from the heap
        result = sorted(heap, reverse=True)
        ids = [score[1] for score in result]
        del query_dot_embedding
        del result
        del query_dot_centroids
        del heap
        del top_scores
        gc.collect()

        return ids