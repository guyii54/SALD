import cv2
import numpy as np
import os
import fnmatch
import data_process as data
import config


# ----------------------airplane dataset----------------------------------
# # result: 186.43622315941903
# img_dir = r'D:\Airplane Keypart\Dataset\FRVC\data\images'
# npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_train_new.npy'
# data = data.Data(img_dir=img_dir, npy_path=npy_path)
# data.readnpy()
# params = config.parser_config('config.cfg')
#
# generator = data.generate(params,mode='test')
# test_num = 1500
# pos_list = []
# for i in range(1,test_num):
#     img, gtmap, visi_weights, ske, name = next(generator)
#     img = img[0]
#     ske = ske[0]
#     Np = np.where(ske>0.5)
#     Np = len(Np[0])
#     Nn = 256*256 - Np
#     pos = Nn/Np
#     pos_list.append(pos)
#     aver_pos = np.average(pos_list)
#     print(aver_pos)

# --------------other dataset----------------
img_dir = r'F:\Projects\Airplane Keypoint\hed-tf-master\data\dataset\SymPASCAL\image_gt\train'
img_list = os.listdir(img_dir)
pic_list = fnmatch.filter(img_list, '*.png')
pos_list = []
for name in pic_list:
    ske = cv2.imread(os.path.join(img_dir, name))
    ske = cv2.cvtColor(ske, cv2.COLOR_BGR2GRAY)
    height, width = ske.shape
    Np = np.where(ske>0.5)
    Np = len(Np[0])
    Nn = height * width - Np
    pos = Nn/Np
    pos_list.append(pos)
    aver_pos = np.average(pos_list)
    print(aver_pos)

