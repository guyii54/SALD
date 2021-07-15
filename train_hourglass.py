import hourglass_new
import data_process
import cv2
import numpy as np
import config
import os

print('--Parsing Config File')
params = config.parser_config('hgconfig.cfg')

print('--Creating Dataset')
data = data_process.Data(img_dir=params['img_dir'], npy_path=params['npy_path'])
data.readnpy()
data.create_sets()

model = hourglass_new.HourglassModel(params=params,dataset=data,mode='hg')

model.generate_model()
# load_path = os.path.join(r'D:\Airplane Keypart\skey_data\HG\weights_1214','tiny_hourglass_28')
model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = data)