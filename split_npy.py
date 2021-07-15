import pandas
import numpy as np


# npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints.npy'
# data = np.load(npy_path, allow_pickle=True)
# print(data.shape)
# split = np.zeros_like(data[0:1000])
# for i in range(0,9):
#     split = data[i:i+1000]
#     np.save('split%d' % i , split)

data = np.load('split0.npy',allow_pickle=True)
print(data.shape)