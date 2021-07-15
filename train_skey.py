import os
import skey_net as skey
import config
import data_process as data

print('--Parsing Config File--')
params = config.parser_config('config.cfg')
# print(params['gpu_memory_fraction'])
data = data.Data(img_dir=params['img_dir'],npy_path=params['npy_path'])
data.readnpy()

skey_net = skey.SKEY(params,data)
hed_load = os.path.join(r'D:\Airplane Keypart\skey_data\HED\weights_1230','hed_1212-166')
hg_load = os.path.join(r'D:\Airplane Keypart\skey_data\HG\weights_1228','tiny_hourglass_14')
# skey_load = os.path.join(r'D:\Airplane Keypart\skey_data\SKEY\weights_1226','skey_1219-2')
skey_net.train(summary=True, hg_load=hg_load,hed_load=hed_load)