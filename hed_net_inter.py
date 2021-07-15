# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import cv2
import config
import random
import os
import tensorflow.contrib.slim as slim
from tensorflow import keras


class HED(object):
    def __init__(self, params,data):
        self.dataset = data
        self.params = params
        self.height = params['height']
        self.width = params['width']
        self.channel = params['channel']
        self.img_dir = params['img_dir']
        self.ske_dir = params['ske_dir']
        self.outchannel = params['outchannel']
        # self.x = tf.placeholder(tf.float32, (None, self.height, self.width, self.channel))
        # self.label = tf.placeholder(tf.float32, (None, self.height, self.width, 1))
        # with open('cfg.yml') as file:
        #     self.cfg = yaml.load(file)

    # 前向传播
    def vgg_hed(self, input):
        self.x = input
        # with tf.name_scope('HED'):
        # bn1: None x 256 x 256 x 64
        bn1, relu1 = self.block(input_tensor=self.x, filters=64, iteration=2, dilation_rate=[(4, 4), (1, 1)], name='block1')
        mp1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool1')
        # bn2: None x 128 x 128 x 128
        bn2, relu2 = self.block(input_tensor=mp1, filters=128, iteration=2, name='block2')
        mp2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool2')
        # bn3: None x 64 x 64 x 256
        bn3, relu3 = self.block(input_tensor=mp2, filters=256, iteration=3, name='block3')
        mp3 = tf.layers.max_pooling2d(inputs=relu3, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool3')
        # bn4: None x 32 x 32 x 512
        bn4, relu4 = self.block(input_tensor=mp3, filters=512, iteration=3, name='block4')
        mp4 = tf.layers.max_pooling2d(inputs=relu4, pool_size=(2, 2), strides=(2, 2), padding='same', name='max_pool4')
        # bn5: None x 16 x 16 x 512
        bn5, relu5 = self.block(input_tensor=mp4, filters=512, iteration=3, name='block5')
        if self.outchannel ==1:
            self.side1 = self.side(input_tensor=bn1, stride=(1, 1), name='side1', deconv=False)
            self.side2 = self.side(input_tensor=bn2, stride=(2, 2), name='side2')
            self.side3 = self.side(input_tensor=bn3, stride=(4, 4), name='side3')
            self.side4 = self.side(input_tensor=bn4, stride=(8, 8), name='side4')
            self.side5 = self.side(input_tensor=bn5, stride=(16, 16), name='side5')
            sides = tf.concat(values=[self.side1, self.side2, self.side3, self.side4, self.side5], axis=3)
            self.fused_side = tf.layers.conv2d(inputs=sides, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                               use_bias=False, kernel_initializer=tf.constant_initializer(0.2),
                                               name='fused_side')
        else:
            self.side1 = self.side_multi(input_tensor=bn1, stride=(1, 1), name='side1', deconv=False,outchannel=self.outchannel)
            self.side2 = self.side_multi(input_tensor=bn2, stride=(2, 2), name='side2', outchannel=self.outchannel)
            self.side3 = self.side_multi(input_tensor=bn3, stride=(4, 4), name='side3', outchannel=self.outchannel)
            self.side4 = self.side_multi(input_tensor=bn4, stride=(6, 6), name='side4', outchannel=self.outchannel)
            self.side5 = self.side_multi(input_tensor=bn5, stride=(8, 8), name='side5', outchannel=self.outchannel)
            sides = tf.concat(values=[self.side1, self.side2, self.side3, self.side4, self.side5], axis=3)
            self.fused_side = tf.layers.conv2d(inputs=sides, filters=self.outchannel, kernel_size=(1, 1), strides=(1, 1),
                                               use_bias=False, kernel_initializer=tf.constant_initializer(0.2),
                                               name='fused_side')
        # side1 = tf.layers.conv2d(inputs=bn1, filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same',
        #                          activation=tf.nn.relu, name='side1')
        # side2 = tf.layers.conv2d_transpose(inputs=bn2, filters=1, kernel_size=(3, 3),
        #                                    strides=(2, 2), padding='same', activation=tf.nn.relu, name='side2')
        # side3 = tf.layers.conv2d_transpose(inputs=bn3, filters=1, kernel_size=(3, 3),
        #                                    strides=(4, 4), padding='same', activation=tf.nn.relu, name='side3')
        # side4 = tf.layers.conv2d_transpose(inputs=bn4, filters=1, kernel_size=(3, 3),
        #                                    strides=(8, 8), padding='same', activation=tf.nn.relu, name='side4')
        # side5 = tf.layers.conv2d_transpose(inputs=bn5, filters=1, kernel_size=(3, 3),
        #                                    strides=(16, 16), padding='same', activation=tf.nn.relu, name='side5')

        return self.side1, self.side2, self.side3, self.side4, self.side5, self.fused_side

    def block(self, input_tensor, filters, iteration, dilation_rate=None, name=None):
        if dilation_rate is None:
            dilation_rate = [(1, 1)]
        if len(dilation_rate) == 1:
            dilation_rate *= iteration

        regularizer = tf.contrib.layers.l2_regularizer(self.params['weight_decay_ratio'])
        with tf.variable_scope(name):
            relu = input_tensor
            for it in range(iteration):
                tp_dilation_rate = dilation_rate.pop(0)
                # print(tp_dilation_rate)
                conv = tf.layers.conv2d(inputs=relu, filters=filters,
                                        kernel_size=(3, 3), strides=(1, 1), padding='same',
                                        activation=None, use_bias=True,
                                        kernel_regularizer=regularizer,
                                        dilation_rate=tp_dilation_rate,
                                        # kernel_initializer=tf.truncated_normal_initializer(stddev=0.5),
                                        name='conv{:d}'.format(it))
                # bn = tf.layers.batch_normalization(inputs=conv, axis=-1, name='bn{:d}'.format(it))
                bn = conv
                relu = tf.nn.relu(bn, name='relu{:d}'.format(it))
        return relu, relu

    def side(self, input_tensor, stride, name, deconv=True):
        with tf.variable_scope(name):
            side = tf.layers.conv2d(inputs=input_tensor, filters=1, kernel_size=(1, 1), strides=(1, 1),
                                    padding='same',
                                    activation=None,
                                    bias_initializer=tf.constant_initializer(value=0),
                                    kernel_initializer=tf.constant_initializer(value=0),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))
            if deconv:
                side = tf.layers.conv2d_transpose(inputs=side, filters=1, kernel_size=(2*stride[0], 2*stride[1]),
                                                  strides=stride, padding='same',
                                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  bias_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params['weight_decay_ratio']),
                                                  activation=None)
            side = tf.image.resize_images(images=side, size=(self.height, self.width),
                                          method=tf.image.ResizeMethod.BILINEAR)
        return side


    def side_multi(self,input_tensor,stride, name,deconv=True,outchannel=1):
        with tf.variable_scope(name):
            side = tf.layers.conv2d(inputs=input_tensor, filters=outchannel, kernel_size=(1, 1), strides=(1, 1),
                                    padding='same',
                                    activation=None,
                                    bias_initializer=tf.constant_initializer(value=0),
                                    kernel_initializer=tf.constant_initializer(value=0),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0002))
            if deconv:
                side = tf.layers.conv2d_transpose(inputs=side, filters=outchannel, kernel_size=(2*stride[0], 2*stride[1]),
                                                  strides=stride, padding='same',
                                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  bias_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params['weight_decay_ratio']),
                                                  activation=None)
            side = tf.image.resize_images(images=side, size=(self.height, self.width),
                                          method=tf.image.ResizeMethod.BILINEAR)
        return side


    def evaluate(self):
        # evaluation criteria
        # accuracy

        # precision

        # recall

        # F1 score
        pass



    def assign_init_weights(self, sess=None):
        with open(self.params['init_weights'], 'rb') as file:
            weights = np.load(file, encoding='latin1', allow_pickle=True).item()
        with tf.variable_scope('block1', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv1_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv1_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv1_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv1_2'][1]))
        print('assign first block done !')
        with tf.variable_scope('block2', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv2_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv2_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv2_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv2_2'][1]))
        print('assign second block done !')
        with tf.variable_scope('block3', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv3_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv3_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv3_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv3_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv3_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv3_3'][1]))
        print('assign third block done !')
        with tf.variable_scope('block4', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv4_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv4_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv4_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv4_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv4_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv4_3'][1]))
        print('assign fourth block done !')
        with tf.variable_scope('block5', reuse=True):
            k = tf.get_variable(name='conv0/kernel')
            sess.run(tf.assign(k, weights['conv5_1'][0]))
            k = tf.get_variable(name='conv0/bias')
            sess.run(tf.assign(k, weights['conv5_1'][1]))

            k = tf.get_variable(name='conv1/kernel')
            sess.run(tf.assign(k, weights['conv5_2'][0]))
            k = tf.get_variable(name='conv1/bias')
            sess.run(tf.assign(k, weights['conv5_2'][1]))

            k = tf.get_variable(name='conv2/kernel')
            sess.run(tf.assign(k, weights['conv5_3'][0]))
            k = tf.get_variable(name='conv2/bias')
            sess.run(tf.assign(k, weights['conv5_3'][1]))
        weights = None  # gc
        print('assign fifth block done !')
        print('--net initializing successfully with vgg16 weights trained by imagenet data')

    # 计算loss
    def calc_loss(self, sideoutputs):
        self.loss = 0
        self.CollectionLoss = {}
        l = []
        # print(type(sideoutputs))
        # print(type(self.label))
        # for side1~side5
        if self.params['is_deep_supervised']:
            for n in range(len(sideoutputs) - 1):
                l.append(tf.nn.weighted_cross_entropy_with_logits(targets=self.label,logits=sideoutputs[n],pos_weight=self.params['pos_weights']))
                tp_loss = self.params['side_weights'][n] * l[n]
                self.loss += tf.reduce_mean(tp_loss)
        # for fuse_side
        self.loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=sideoutputs[-1],
                                                                             pos_weight=self.params['pos_weights']))
        self.CollectionLoss['l'] = l
        self.CollectionLoss['tp_loss'] = tp_loss
        self.CollectionLoss['loss1']=self.loss
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if self.params['use_weight_regularizer']:
            self.CollectionLoss['RegLoss'] = tf.add_n(reg_loss)
            re_loss = tf.add_n(reg_loss)
            # print('loss2:',re_loss)
            self.loss = re_loss + self.loss
        self.CollectionLoss['TotalLoss'] = self.loss
        return self.loss,self.CollectionLoss  # 1.0*tf.shape(self.label)[0]

    def load_data(self):
        print('--LOAD IMAGE--')
        self.imgs = None
        self.labels = None
        self.sample_num = None
        img_names = []
        label_names = []
        with open(self.params['hed_train_file'], 'r') as f:
            for line in f:
                cur_line = line.strip()
                img_name = os.path.join(self.img_dir, cur_line)
                label_name = os.path.join(self.ske_dir, cur_line)
                img_names.append(img_name)
                label_names.append(label_name)
        self.sample_num = len(img_names)
        self.imgs = np.zeros((len(img_names), self.params['height'], self.params['width'], self.params['channel']), np.float32)
        self.labels = np.zeros((len(img_names), self.params['height'], self.params['width'], 1), np.float32)
        for it in range(self.sample_num):
            tmp_img = cv2.imread(img_names[it])
            tmp_label = cv2.imread(label_names[it], cv2.IMREAD_GRAYSCALE)
            try:
                self.imgs[it, :, :, :] = tmp_img.astype(np.float32)
                self.labels[it, :, :, 0] = (tmp_label / 255).astype(np.float32)
            except:
                print('wrong imgname:', img_names[it])
        self.imgs = self.imgs - self.params['mean']
        print('Dataset Size: %d' % self.sample_num)
        print('--LOAD IMAGE DONE--')


    # run this function each epoch
    def generate_data(self):
        print('---generate new batch---')
        batchsize = self.params['batch_size']
        idx = list(range(self.sample_num))
        if self.params['if_shuffle']:
            random.shuffle(idx)
        for i in range(0, self.sample_num, batchsize):
            imgs = self.imgs[idx[i:min(i+ batchsize, self.sample_num)], :, :, :]
            labels = self.labels[idx[i:min(i + batchsize, self.sample_num)], :, :, :]
            # print('batch_size: ', labels.shape[0])
            yield imgs, labels


    def summary(self):
        # for structure
        max_outputs = 1
        tf.summary.image(name='orig_image_sm', tensor=self.x, max_outputs=max_outputs)
        tf.summary.image(name='side1_im', tensor=tf.sigmoid(self.side1), max_outputs=max_outputs, )
        tf.summary.image(name='side2_im', tensor=tf.sigmoid(self.side2), max_outputs=max_outputs, )
        tf.summary.image(name='side3_im', tensor=tf.sigmoid(self.side3), max_outputs=max_outputs, )
        tf.summary.image(name='side4_im', tensor=tf.sigmoid(self.side4), max_outputs=max_outputs, )
        tf.summary.image(name='side5_im', tensor=tf.sigmoid(self.side5), max_outputs=max_outputs, )
        tf.summary.image(name='fused_side_im', tensor=tf.sigmoid(self.fused_side), max_outputs=max_outputs, )

        tf.summary.histogram(name='side1_hist', values=tf.sigmoid(self.side1))
        tf.summary.histogram(name='side2_hist', values=tf.sigmoid(self.side2))
        tf.summary.histogram(name='side3_hist', values=tf.sigmoid(self.side3))
        tf.summary.histogram(name='side4_hist', values=tf.sigmoid(self.side4))
        tf.summary.histogram(name='side5_hist', values=tf.sigmoid(self.side5))
        tf.summary.histogram(name='fused_side_hist', values=tf.sigmoid(self.fused_side))
        # for loss
        tf.summary.scalar(name='loss_sm', tensor=self.loss)
        # tf.summary.scalar(name='floss_sm', tensor=self.floss)
        tf.summary.image(name='label_sm', tensor=self.label, max_outputs=max_outputs, )

    def CustomSet_Train(self,load = None, begin = 0):
        '''
        defference with self.train: data generated by self.dataset
        :return:
        '''
        # graph
        self.x = tf.placeholder(tf.float32, (None, self.height, self.width, self.channel))
        self.label = tf.placeholder(tf.float32, (None, self.height, self.width, self.outchannel))
        sideoutputs = self.vgg_hed(self.x)

        # loss
        loss,CollectLoss = self.calc_loss(sideoutputs)

        # train params set
        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(learning_rate=self.params['hed_base_lr'],
                                        global_step=global_step,
                                        decay_steps=self.params['hed_decay_steps'],
                                        decay_rate=self.params['hed_decay_rate'],
                                        staircase=self.params['hed_staircase'])
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

        print('--INIT SESSION--')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['gpu_memory_fraction'],
                                    allow_growth=self.params['allow_growth'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.params['gpu'])
        device_config = tf.ConfigProto(log_device_placement=self.params['log_device_placement'],
                                       allow_soft_placement=self.params['allow_soft_placement'],
                                       gpu_options=gpu_options)

        self.Session = tf.Session(config=device_config)
        print('Session creat done.')

        # data
        generator = self.dataset.generate(self.params)

        self.summary()
        tf.summary.scalar(name='lr', tensor=lr)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.params['hed_log_dir_train'], graph=self.Session.graph, flush_secs=15)

        print('--TRAINING--')
        # load weights
        saver = tf.train.Saver()
        self.Session.run(tf.global_variables_initializer())
        if load is not None:
            saver.restore(self.Session, load)
            print('--load done--')
        else:
            begin = 0
            self.assign_init_weights(self.Session)
        step = begin * self.params['hed_epoch_size']
        for epoch in range(begin, self.params['hed_max_epochs']):
            for i in range(self.params['hed_epoch_size']):
                total = self.params['hed_epoch_size']
                block = int(i / total * 20)

                img_train, gt_train, weight_train, ske = next(generator)
                cur_lr, out, cur_loss, merged_summary, _ = self.Session.run([lr, sideoutputs, CollectLoss, merged_summary_op, train_op],
                                                         feed_dict={self.x: img_train, self.label: ske})

                print('\rEpoch %d'%epoch + '    Training:|{0}{1}|'.format('=' * block, ' ' * (20 - block)) + '%0.3s%%' % (
                            i / total * 100) + '    Loss:%f' % (cur_loss['TotalLoss']) + '    lr:%0.8f'% (cur_lr),
                      end='')
                if cur_loss['TotalLoss'] <0:
                    print(cur_loss)
                step += 1
                if i % self.params['hed_save_step'] == 0:
                    summary_writer.add_summary(merged_summary, global_step=step)

            saver.save(sess=self.Session,
                       save_path=os.path.join(self.params['hed_weights_path'], 'hed_1212'),
                       global_step=epoch)
            print(' Save a snapshoot !')
        summary_writer.close()
        saver.save(sess=self.Session, save_path=os.path.join(self.params['hed_weights_path'], 'hed_1212'),
                   global_step=self.params['hed_max_epochs'])
        print('save final model')

    def pred(self,out_path,test_num,weight_path):
        # graph
        testimg = tf.placeholder(tf.float32, (None, self.height, self.width, self.channel))
        sideoutputs = self.vgg_hed(testimg)
        sides = [self.side1,self.side2,self.side3,self.side4,self.side5,self.fused_side]
        weighted_side = tf.zeros_like(sides[0])
        for i in range(5):
            weighted_side = weighted_side + self.params['deploy_weights'][i] * sides[i]
        weighted_side = weighted_side / len(sides)

        # create
        sess = tf.Session()
        print('--Session creat done.')
        # saver = tf.train.Saver()
        # # load weights
        # saver.restore(sess, weight_path)
        # ------load part-------
        AllWeights = slim.get_variables_to_restore()
        HEDWeights = [v for v in AllWeights if (
                'block' in v.name.split('/')[0] or
                'side' in v.name.split('/')[0] or
                'fused_side' in v.name.split('/')[0])]
        hedsaver = tf.train.Saver(HEDWeights)
        hedsaver.restore(sess, weight_path)

        # test
        print('--Testing Begin')
        ske_output = {}
        generator = self.dataset.generate(mode='test',params=self.params)
        pic_count = 0
        for i in range(1, test_num):
            block = int(i/test_num * 20)
            print('\rDeploying:|{0}{1}|'.format('='*block,' '*(20-block)) + '%f%%' % (i/test_num*100), end=' ')
            img, gtmap, visi_weights, ske, name = next(generator)
            out,sideout,fuseside = sess.run([sideoutputs,sides,self.fused_side],feed_dict={testimg:img})
            # out Dim: 1 x 256 x 256 x 1
            # img Dim 1 x 256 x 256 x 3
            ske = self.hed_post_prosess(out)
            side5 = sideout[4]
            side5 = self.hed_post_prosess(side5)
            # cv2.normalize()
            # cv2.imwrite(os.path.join(out_path,'ske',name),ske)
            # cv2.imwrite(os.path.join(out_path,'side5',name),side5)
            # ske_output[name] = fuseside
            fuseside = np.squeeze(fuseside)
            for c in range(self.params['outchannel']):
                cv2.imwrite(os.path.join(out_path,'c_%d_%s'% (c, name)), self.hed_post_prosess(fuseside[:,:,c]))
            pic_count += 1
        # np.save(os.path.join(out_path,'ske.npy'),ske_output)
        print('\n%d was inferenced' % pic_count)

    def hed_post_prosess(self,ske):
        '''
        ske Dim: 256 x 256
        :param ske:
        :return:
        '''
        out = np.squeeze(ske)
        ske_thresh = cv2.threshold(out, 0, maxval=256, type=cv2.THRESH_TOZERO)
        ske_thresh = ske_thresh[1]
        ske_norm = (ske_thresh - ske_thresh.min()) / (ske_thresh.max() - ske_thresh.min())
        ske = ske_norm * 256
        ske = np.expand_dims(ske, axis=2)
        ske = ske.astype(np.uint8)
        return ske


    def train(self):
        # graph
        self.x = tf.placeholder(tf.float32, (None, self.height, self.width, self.channel))
        self.label = tf.placeholder(tf.float32, (None, self.height, self.width, 1))
        sideoutputs = self.vgg_hed(self.x)

        # loss
        loss = self.calc_loss(sideoutputs)

        # train params set
        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(learning_rate=self.params['base_lr'],
                                        global_step=global_step,
                                        decay_steps=self.params['decay_steps'],
                                        decay_rate=self.params['decay_rate'],
                                        staircase=self.params['staircase'])
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

        print('--INIT SESSION--')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['gpu_memory_fraction'],
                                    allow_growth=self.params['allow_growth'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.params['gpu'])
        device_config = tf.ConfigProto(log_device_placement=self.params['log_device_placement'],
                                       allow_soft_placement=self.params['allow_soft_placement'],
                                       gpu_options=gpu_options)

        self.Session = tf.Session(config=device_config)
        print('Session creat done.')

        # load data
        self.dataset = self.load_data()

        # write summary
        self.summary()
        tf.summary.scalar(name='lr', tensor=lr)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.params['log_dir_train'], graph=self.Session.graph, flush_secs=15)

        print('--TRAINING--')
        # load weights
        saver = tf.train.Saver()
        self.Session.run(tf.global_variables_initializer())
        self.assign_init_weights(self.Session)
        print('Initial weights done.')

        step = 0
        for epoch in range(1, self.params['max_epochs']+1):
            for imgs, labels in self.generate_data():
                merged_summary, _ = self.Session.run([merged_summary_op, train_op],
                                                feed_dict={self.x: imgs, self.label: labels})
                if not (step % 1):
                    summary_writer.add_summary(merged_summary, global_step=step)
                    print('save a merged summary !')
                step += 1

                print('global_step:', self.Session.run(global_step), 'epoch: ', epoch)

            if not epoch % self.params['snapshot_epochs']:
                saver.save(sess=self.Session, save_path=os.path.join(self.params['model_weights_path'], 'vgg16_hed'), global_step=epoch)
                print('save a snapshoot !')
        summary_writer.close()
        saver.save(sess=self.Session, save_path=os.path.join(self.params['model_weights_path'], 'vgg16_hed'), global_step=epoch)
        print('save final model')




# test structer
if __name__ == "__main__":
    print('--Parsing Config File--')
    params = config.parser_config('config.cfg')




    # #-----------dataset test-------------#
    # hed = HED(params)
    # hed.load_data()
    # for imgs, labels in hed.generate_data():
    #     print(imgs.shape)

    #------------train test---------------#
    # hed = HED(params,data=)
    # hed.train()

    # ipt = np.ones((12, 112, 112, 3))
    # merged_summary = tf.summary.merge_all()
    # for s in s1:
        # print('s:', s)
    # print('trainable variables:', tf.trainable_variables())
    # with tf.variable_scope('block1', reuse=True):
    #     k = tf.get_variable(name='conv0/kernel')
    #     print(k)
    # # tf.get_default_graph()
    # with tf.Session() as sess:
    #     # `sess.graph` provides access to the graph used in a `tf.Session`.
    #     summary_writer = tf.summary.FileWriter('test_log', sess.graph)
    #
    #     # Perform your computation...
    #     # for i in range(1000):
    #     #     sess.run(train_op)
    #     #     # ...
    #     # summary_writer.add_summary(merged_summary)
    #
    #     summary_writer.close()
    # print(s1)
    # with tf.variable_scope('block1', reuse=True):
    #     print(tf.get_variable('conv1/kernel'))
    # summary = tf.summary.image(name='hah', tensor='block1/side1/kernel')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     a = sess.run(s1, feed_dict={hed.x: ipt})
    #     print(a.shape)
    #     print(tf.trainable_variables())
    #     with tf.variable_scope('block1', reuse=True):
    #         print(sess.run(tf.get_variable('conv1/kernel')))


