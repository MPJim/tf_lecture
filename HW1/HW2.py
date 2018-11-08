import numpy as np
import tensorflow as tf
import struct
import matplotlib#.pyplot as plt

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

training_data = read_idx('./MNIST/train-images.idx3-ubyte')
traing_label  = read_idx('./MNIST/train-labels.idx1-ubyte')
test_data     = read_idx('./MNIST/t10k-images.idx3-ubyte')
test_label    = read_idx('./MNIST/t10k-labels.idx1-ubyte')



'''
fig, ax = plt.subplot(1,10)
for i in range(10): ax[i]
'''