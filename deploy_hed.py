import data_process
import hed_net_inter as hed
import os
import config

img_dir = r'D:\Airplane Keypart\Dataset\FRVC\data\images'
npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_test_new.npy'
params = config.parser_config('config.cfg')
data = data_process.Data(img_dir=img_dir, npy_path=npy_path)
data.readnpy()

hed= hed.HED(params,data)
out_path = r'D:\Airplane Keypart\skey_data\HED\output_1230'
# model_path = r'D:\Airplane Keypart\skey_data\HED\weights_1218_2'
test_num = 1577
# gt_path = r'D:\Airplane Keypart\Dataset\FRVC\data\HED_GT'
# data.Gen_Skeval_GT(params,gt_path,test_num)
hed.pred(out_path,test_num,os.path.join(r'D:\Airplane Keypart\skey_data\HED\weights_1230','hed_1212-166'))