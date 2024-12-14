from typing import Dict, List, Annotated
import numpy as np
import os
from IVF import *
import re

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
# NCLUSTERS = 10000
# NPROBS = 110
# BATCH_SIZE = 50000
def extract_db_size(input_string):
    match = re.search(r'_(\d+)m', input_string)
    if match:
        return int(match.group(1)) * 10**6
    return None  

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "saved_db_20m", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.db_size= extract_db_size(index_file_path)
        if self.db_size == 10**6:
            self.NCLUSTERS = 500
            self.NPROBS = 13
            self.BATCH_SIZE = 30000
        elif self.db_size == 10*10**6:
            self.NCLUSTERS = 10000
            self.NPROBS = 50
            self.BATCH_SIZE = 5000
        elif self.db_size == 15*10**6:
            self.NCLUSTERS = 15000
            self.NPROBS = 80
            self.BATCH_SIZE = 30000
        elif self.db_size == 20*10**6:
            self.NCLUSTERS = 15000
            self.NPROBS = 110
            self.BATCH_SIZE = 30000
            
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        ivf = IVF(self.db_path,self.NCLUSTERS,self.NPROBS,DIMENSION,self.db_size)

        return ivf.retrieve(query,top_k,index_path=self.index_path, batch_size=self.BATCH_SIZE)

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        ivf = IVF(self.db_path,self.NCLUSTERS,self.NPROBS,DIMENSION,self.db_size)
        ivf.train()
        
        return
