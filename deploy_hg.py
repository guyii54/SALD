import data_process
import config
import hourglass_new as hg
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_dir', type=str, default=r'D:\Airplane Keypart\Dataset\FRVC\data\images',help='which picture will be predicted')
parser.add_argument('--npy_path', type=str, default=r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_test_new.npy')
parser.add_argument('--out_path', type=str, default=r'D:\Airplane Keypart\skey_data\HG\TEST_EXE')
parser.add_argument('--model_path', type=str, default=r'D:\Airplane Keypart\skey_data\HG\weights_1228')

opt = parser.parse_args()

img_dir = opt.img_dir
if not os.path.exists(img_dir):
    print(img_dir,'not found!!!')
    raise FileNotFoundError
else:
    print('--Image files Found.--')
# csv_path = r'D:\Airplane Keypart\hourglasstensorlfow\hourglass-branch/via_export_csv.csv'
# bbox_path = r'D:\Airplane Keypart\hourglasstensorlfow\data\Dataset\FRVC\data\images_box.txt'
# npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_train.npy'
npy_path = opt.npy_path
if not os.path.exists(npy_path):
    print(npy_path,'not found!!!')
    raise FileNotFoundError
else:
    print('--Test list Found.--')

model_path = opt.model_path
if not os.path.exists(model_path):
    print(model_path,'not found!!!')
    raise FileNotFoundError
else:
    print('--Model file Found.--')

# pic_list = r'D:\Airplane Keypart\Dataset\FRVC\data\ske_pic_list.txt'
params = config.parser_config('hgconfig.cfg')
data = data_process.Data(img_dir=img_dir, npy_path=npy_path)
data.readnpy()

hg= hg.HourglassModel(params,dataset=data,mode='hg')

out_path = opt.out_path
if not os.path.exists(out_path):
    os.makedirs(out_path)

hg.pred(out_path,1500,os.path.join(model_path,'tiny_hourglass_14'))

print('Prediction Done! Results saved in %s'%out_path)

