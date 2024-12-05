import numpy as np
import struct


def read_database(original_data_path, data_size, vec_size):
    data = []
    with open(original_data_path, 'rb') as file:
        # Calculate the position of the row where the size of one row (vec_size * float_size)
        for row_id in range(data_size):
            position = row_id * (vec_size * 4)
            # Seek to the position of the row
            file.seek(position)
            # Read the row with size of one row (ID + vec_size * floats)s
            packed_data = file.read(vec_size * 4)
            data.append(struct.unpack(f'{vec_size}f', packed_data))
            
    return np.array(data, dtype=np.float32)


def write_centroids_file(centroids, centroids_file_path, vec_size):
    with open(centroids_file_path, 'wb') as file:
        for centroid in centroids:
            packed_data = struct.pack(f'{vec_size}f', *centroid)
            file.write(packed_data)

    return 


def write_clusters_file(clusters, clusters_file_path, cluster_begin_file_path):
    with open(clusters_file_path, 'wb') as cluster_file:
        with open(cluster_begin_file_path, 'wb') as pos_file:
            start = 0
            for _, cluster in clusters.items(): 
                cluster_packed_data = struct.pack(f'{len(cluster)}i', *cluster)
                cluster_file.write(cluster_packed_data)
                pos_packed_file = struct.pack(f'i',start)
                pos_file.write(pos_packed_file)
                start += len(cluster)

    return 


def read_centroids_file(centroids_file_path, vec_size):
    centroids = []
    with open(centroids_file_path, 'rb') as file:
        while True:
            packed_data = file.read(vec_size * 4)

            if packed_data == b'':
                break

            data = struct.unpack(f'{vec_size}f', packed_data)
            centroids.append(data)

    return np.array(centroids)


def read_one_cluster(cluster_id, clusters_file_path, cluster_begin_file_path, n_clusters, data_size):
    with open(clusters_file_path, 'rb') as cluster_file:
        with open(cluster_begin_file_path, 'rb') as pos_file:
            start_pos = cluster_id * 4
            pos_file.seek(start_pos) 
            start_packed_data = pos_file.read(4)
            start = struct.unpack('i',start_packed_data)[0]

            if (cluster_id + 1 < n_clusters):
                end_pos = (cluster_id + 1) * 4
                pos_file.seek(end_pos) 
                end_packed_data = pos_file.read(4)
                end = struct.unpack('i',end_packed_data)[0]
            else:
                end = data_size

            vec_indexes=[]
            cluster_file.seek(start * 4)
            while cluster_file.tell() < (end * 4):
                packed_data = cluster_file.read(4)
                if packed_data == b'':
                    break
                data = struct.unpack('i', packed_data)[0]
                vec_indexes.append(data)

    return vec_indexes

def read_one_embedding(original_data_path,row_id,vec_size):
    with open(original_data_path, 'rb') as file:
        position = row_id * (vec_size*4)
        file.seek(position)
        packed_data = file.read(vec_size * 4)
        embedding= struct.unpack(f'{vec_size}f', packed_data)
            
    return np.array(embedding, dtype=np.float32)
