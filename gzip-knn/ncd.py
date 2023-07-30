import gzip 
import numpy as np

def calculate_ncd_row(data_row, trainset):
    i, x = data_row
    x_compressed = np.array([len(gzip.compress(x.encode()))])
    x2_compressed = np.array([len(gzip.compress(x2.encode())) for x2 in trainset])
    xx2_compressed = np.array([len(gzip.compress((" ".join([x, x2])).encode())) for x2 in trainset])
    ncd_values = (xx2_compressed - np.minimum(x_compressed, x2_compressed)) / np.maximum(x_compressed, x2_compressed)
    return i, ncd_values.tolist()