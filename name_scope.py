import tensorflow as tf
import tensorflow.contrib.slim as slim
import config
import hourglass_new as hg
import hed_net_inter as hed
import data_process

print('--Parsing Config File--')
params = config.parser_config('config.cfg')
# print(params['gpu_memory_fraction'])
data = data_process.Data(img_dir=params['img_dir'],npy_path=params['npy_path'])
data.readnpy()

hed_net = hed.HED(params,data)
hg_net = hg.HourglassModel(params)
img = tf.placeholder(tf.float32,(None, params['img_size'], params['img_size'], params['channel']))
ske = tf.placeholder(tf.float32, (None, params['img_size'], params['img_size'], 1))
key_out = hg_net._graph_hourglass(img)
ske_out =  hed_net.vgg_hed(img)
var = slim.get_variables_to_restore()
for v in var:
    print(v)
