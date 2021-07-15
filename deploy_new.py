# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import yaml
from hed_net_inter import HED
from loss import HedLoss
import os
import cv2
import argparse
import gc
import config

def img_pre_process(img, **kwargs):
    # normalization
    def stretch(bands, lower_percent=2, higher_percent=98, bits=8):
        if bits not in [8, 16]:
            print('error ! dest image must be 8bit or 16bits !')
            return
        out = np.zeros_like(bands, dtype=np.float32)
        n = bands.shape[2]
        for i in range(n):
            a = 0
            b = 1
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            if d-c == 0:
                out[:, :, i] = 0
                continue
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            out[:, :, i] = np.clip(t, a, b)
        if bits == 8:
            return out.astype(np.float32)*255
        else:
            return np.uint16(out.astype(np.float32)*65535)
    # reduce mean
    img = stretch(img)
    img -= kwargs['mean']
    return img

# 把400*400的大图分成四个小块，分别进行预测，再拼接起来
def predict_big_map(img_path, out_shape=(448, 448), inner_shape=(224, 224), out_channel=1, pred_fun=None, **kwargs):
    """
    :param img_path: big image path
    :param out_shape: (height, width)
    :param inner_shape: (height, width)
    :param out_channel: predicted results' channel num
    :param pred_fun: forward model
    :return: predicted image
    """
    image = cv2.imread(img_path, )
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        gc.collect()
    pd_up_h, pd_lf_w = np.int64((np.array(out_shape)-np.array(inner_shape)) / 2)

    print(image.shape)
    ori_shape = image.shape
    pd_bm_h = (out_shape[0]-pd_up_h) - (image.shape[0] % inner_shape[0])
    pd_rt_w = (out_shape[1]-pd_lf_w) - (image.shape[1] % inner_shape[1])

    # np.ceil向上取整
    it_h = np.int64(np.ceil(1.0*image.shape[0] / inner_shape[0]))
    it_w = np.int64(np.ceil(1.0*image.shape[1] / inner_shape[1]))

    image_pd = np.pad(image, ((pd_up_h, pd_bm_h), (pd_lf_w, pd_rt_w), (0, 0)), mode='reflect').astype(np.float32)  # the image is default a color one
    print(image_pd.shape)
    print((pd_up_h, pd_bm_h), (pd_lf_w, pd_rt_w))
    gc.collect()

    tp1 = np.array(inner_shape[0] - ori_shape[0] % inner_shape[0])
    tp2 = np.array(inner_shape[1] - ori_shape[1] % inner_shape[1])
    if ori_shape[0] % inner_shape[0] == 0:
        tp1 = 0
    if ori_shape[1] % inner_shape[0] == 0:
        tp2 = 0

    out_img = np.zeros((ori_shape[0]+tp1, ori_shape[1]+tp2, out_channel), np.float32)

    image = None  # release memory
    # main loop
    for ith in range(0, it_h):
        h_start = ith * inner_shape[0]
        count = 1
        for itw in range(0, it_w):
            w_start = itw*inner_shape[1]
            tp_img = image_pd[h_start:h_start+out_shape[0], w_start:w_start+out_shape[1], :]

            tp_img = img_pre_process(tp_img.copy(), **kwargs)

            tp_out = pred_fun(tp_img[np.newaxis, :])
            tp_out = np.squeeze(tp_out, axis=0)

            out_img[h_start:h_start+inner_shape[0], w_start:w_start+inner_shape[1], :] = tp_out[pd_up_h:pd_up_h+inner_shape[0], pd_lf_w:pd_lf_w+inner_shape[1], :]


            count += 1
    return out_img[0:ori_shape[0], 0:ori_shape[1], :]

if __name__ == '__main__':
    print('--Parsing Config File')
    params = config.parser_config('config.cfg')
    # print(params['gpu_memory_fraction'])
    print('--INIT SESSION')
    # for device
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=params['gpu_memory_fraction'],
                                allow_growth=params['allow_growth'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params['gpu'])
    device_config = tf.ConfigProto(log_device_placement=params['log_device_placement'],
                                   allow_soft_placement=params['allow_soft_placement'],
                                   gpu_options=gpu_options)
    # for pic
    height = params['height']
    width = params['width']
    channel = params['channel']
    mean = params['mean']
    # for network
    hed_class = HED(params)
    hed_class.vgg_hed()
    sides = [tf.sigmoid(hed_class.side1),
             tf.sigmoid(hed_class.side2),
             tf.sigmoid(hed_class.side3),
             tf.sigmoid(hed_class.side4),
             tf.sigmoid(hed_class.side5),
             tf.sigmoid(hed_class.fused_side)]
    aver_sides = 1.0 * tf.add_n(sides) / len(sides)
    weighted_side = tf.zeros_like(sides[0])
    for i in range(5):
        weighted_side = weighted_side + params['deploy_weights'][i] * sides[i]
    weighted_side = weighted_side / len(sides)

    sess = tf.Session(config=device_config)
    print('--Session creat done.')
    saver = tf.train.Saver()
    # load weights
    saver.restore(sess, params['model_weights_path'] + 'vgg16_hed-300')
    print('--Testing Begin')
    with open(params['test_dir']) as f:
        for line in f:
            cur_name = line
            cur_name = cur_name.replace('\n', '')
            # print('cur_name:', cur_name)
            if params['is_complex_pre']:
                output_img = predict_big_map(img_path=cur_name, out_shape=(height, width), inner_shape=(224, 224), out_channel=1,
                                         pred_fun=(lambda ipt: sess.run(weighted_side, feed_dict={hed_class.x: ipt})), mean=mean)
            else:
                cur_img = cv2.imread(cur_name)
                input_img = img_pre_process(cur_img, mean=mean)
                out = sess.run(weighted_side, feed_dict={hed_class.x: input_img[np.newaxis, :]})
                output_img = np.squeeze(out, axis=0)
            output_img = np.squeeze((output_img * 255).astype(np.uint8))
            output_img = 255 * (output_img > params['binary_thresh'])
            write_name = os.path.join(params['ske_out_dir'],
                                      os.path.split(cur_name)[1])
            # cv2.imwrite('./data/result/tb_gray_img.png', output_img)
            # cv2.imwrite('./data/result/tb_black_img.png', 255 * (output_img > 127))
            cv2.imwrite(write_name, output_img)
            if params['is_visual']:
                ske = cv2.imread(write_name)
                visual = 0.5 * ske + 0.5 * cur_img
                write_name = os.path.join(params['visu_out_dir'],os.path.split(cur_name)[1])
                cv2.imwrite(write_name, visual)
        print('generate done.')
        sess.close()