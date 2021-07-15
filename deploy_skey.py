import config
import skey_net as skey
import os
import data_process as data


npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_test_new.npy'
print('--Parsing Config File--')
params = config.parser_config('config.cfg')
# print(params['gpu_memory_fraction'])
data = data.Data(img_dir=params['img_dir'],npy_path=npy_path)
data.readnpy()

skey_net = skey.SKEY(params,data)
test_num = 1577
out_path = r'D:\Airplane Keypart\skey_data\SKEY\output_0102'
load_path = os.path.join(r'D:\Airplane Keypart\skey_data\SKEY\weights_0102','skey_1219-5')
skey_net.pred(test_num,out_path,load_path)
