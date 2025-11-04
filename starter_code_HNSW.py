import faiss
import h5py
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
import time

# get the SIFT dataset from github
def get_sift_data(file_path='sift1m.h5'):
    url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded SIFT1M dataset to {file_path}")

def build_hnsw_index(file_path='sift1m.h5', M=16, efConstruction=200, efSearch=200):
    with h5py.File(file_path, 'r') as f:
        # print(list(f.keys()))
        x_train = f['train'][:]
        x_query = f['test'][:]

    d = x_train.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    index.add(x_train)
    return index, x_query

def evaluate_hnsw():
    # build hnsw index
    index, xq = build_hnsw_index()

    # run query to search for 10 nearest neighbors
    D, I = index.search( xq[0:1], 10)  
    
    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    with open('output.txt', 'w') as f:
        for neighbors in I:
            for idx in neighbors:
                f.write(f"{idx}\n")
    

if __name__ == "__main__":
    # download sift data
    get_sift_data()

    # Q1: Evaluate HNSW and write output to output.txt
    evaluate_hnsw()

