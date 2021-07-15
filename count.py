import numpy as np
import data_process


img_dir = r'D:\Airplane Keypart\Dataset\FRVC\data\images'
npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_test_new.npy'
data = data_process.Data(img_dir=img_dir, npy_path=npy_path)
data.readnpy()
aver_length = np.zeros((1,2))
length = []

for name in data.pic_list:
    raw_box = data.bbox_dict[name]
    raw_box = data.str2num_list(raw_box)
    raw_height = raw_box[3] - raw_box[1]
    raw_width = raw_box[2] - raw_box[0]
    agu_box = max(raw_height,raw_width)
    ratio = np.asarray([raw_height/agu_box, raw_width/agu_box])
    heat_length = ratio * 64
    length.append(heat_length)

length = np.asarray(length)
aver_length = np.average(length,axis=0)
print(aver_length)