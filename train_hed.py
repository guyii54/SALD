import hed_net_inter as hed
import config
import data_process

print('--Parsing Config File--')
params = config.parser_config('config.cfg')
# print(params['gpu_memory_fraction'])
data = data_process.Data(img_dir=params['img_dir'],npy_path=params['npy_path'])
data.readnpy()
data.create_sets()

# npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_train.npy'
hed = hed.HED(params,data)
# load = r'D:\Airplane Keypart\skey_data\HED\modelweights_1216/hed_1212-154'
hed.CustomSet_Train()

