import sys
print(sys.path)

print(123)
print(123)
print(123)


def test1():
    print(1)
    print(2)
    print(3)
    return

test1()

import tensorflow as tf
import glob
import numpy as np
from keras.utils import to_categorical

def _load_data(file_path):
    print('aaaaaaa')
    print(file_path)
    x = np.load(file_path)
    print(f"这个是从文件加载出来的{x.shape}")
    label_file = file_path.replace('features', 'labels')
    y = np.load(label_file)
    return x.astype('float32'), to_categorical(y.astype(int), num_classes)

file1 = './data/KGS-2011-19-19099-test_features_0.npy'
_load_data(file1)