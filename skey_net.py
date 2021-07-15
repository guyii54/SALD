import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
import config
from tensorflow import keras
import data_process
import hourglass_new as HG
import hed_net_inter as HED
import tensorflow.contrib.slim as slim
import cv2

class SKEY():
    def __init__(self, params, Data):
        self.params = params
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:0'
        self.joint_names = params['joint_list']
        self.dataset = Data


    def skey_graph(self):
        self.hed = HED.HED(self.params,self.dataset)
        self.hg = HG.HourglassModel(params=self.params)
        self.ske_out =  self.hed.vgg_hed(self.img)  #tuple
        # ske_out type: tuple
        # fuse_ske_out Dim: None x 256 x 256 x 1 or None x 256 x 256 x 4
        fuse_ske_out = self.ske_out[5]
        # self.hg_input Dim: None x 256 x 256 x 4 or None x 256 x 256 x 7
        self.hg_input = tf.concat([fuse_ske_out, self.img], axis=3)
        # key_out Dim: None x 4 x 64 x 64 x 12
        self.key_out = self.hg._graph_hourglass(self.hg_input)
        self.AllWeights = slim.get_variables_to_restore()
        return self.ske_out, self.key_out

    def cal_loss(self, ske_out, key_out, bias_weight):
        self.key_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=key_out, labels= self.htmap), name= 'cross_entropy_loss')
        self.ske_loss = 0
        if self.params['is_deep_supervised']:
            for n in range(len(ske_out) -1 ):
                tmp_loss = self.params['side_weights'][n]*tf.nn.weighted_cross_entropy_with_logits(targets=self.ske, logits=ske_out[n],pos_weight=self.params['pos_weights'])
                self.ske_loss += tf.reduce_mean(tmp_loss)
        self.ske_loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.ske, logits=ske_out[-1],
                                                                             pos_weight=self.params['pos_weights']))
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if self.params['use_weight_regularizer']:
            self.regu_loss = tf.add_n(reg_loss)
        return self.ske_loss + bias_weight * self.key_loss + self.regu_loss, self.ske_loss, self.key_loss

    def generate_model(self):
        startTime = time.time()
        print('CREAT MODEL')
        with tf.name_scope('input'):
            self.img = tf.placeholder(tf.float32,(None, self.params['img_size'], self.params['img_size'], self.params['channel']))
            self.ske = tf.placeholder(tf.float32, (None, self.params['img_size'], self.params['img_size'], self.params['outchannel']))
            self.htmap = tf.placeholder(tf.float32, (None, self.params['nstacks'], 64, 64, self.params['outdim']))
        inputTime = time.time()
        print('---Inputs : Done (' + str(int(abs(inputTime-startTime))) + ' sec.)')
        ske, key = self.skey_graph()
        graphTime = time.time()
        print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')
        with tf.name_scope('loss'):
            self.total_loss, ske_l, key_l = self.cal_loss(ske_out=ske,key_out=key,bias_weight=self.params['bias_weight'])
        lossTime = time.time()
        print('---Loss : Done (' + str(int(abs(graphTime - lossTime))) + ' sec.)')

        with tf.device(self.cpu):
            with tf.name_scope('steps'):
                self.train_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.name_scope('lr'):
                self.lr = tf.train.exponential_decay(self.params['learning_rate'], self.train_step, decay_rate=self.params['learning_rate_decay'], decay_steps=self.params['decay_step'], staircase=True, name='learning_rate')

        with tf.device(self.gpu):
            with tf.name_scope('rmsprop'):
                self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            optimTime = time.time()
            print('---Optim : Done (' + str(int(abs(optimTime - lossTime))) + ' sec.)')
            with tf.name_scope('minimizer'):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    self.train_rmsprop = self.rmsprop.minimize(self.total_loss, self.train_step)
            minimTime = time.time()
            print('---Minimizer : Done (' + str(int(abs(optimTime - minimTime))) + ' sec.)')
            self.init = tf.global_variables_initializer()
            initTime = time.time()
            print('---Init : Done (' + str(int(abs(initTime - minimTime))) + ' sec.)')
            self.summary()
            endTime = time.time()
            print('Model created (' + str(int(abs(endTime - startTime))) + ' sec.)')
            del endTime, startTime, initTime, optimTime, minimTime, lossTime, graphTime, inputTime

    def summary(self):
        # image:
        # self.ske_out  #list
        key_out = self.key_out  #None x 4 x 64 x 64 x 12
        with tf.name_scope('Skeleton'):
            for i in range(len(self.ske_out)):
                # ske_out[i] Dim: None x 256 x 256 x 1
                tf.summary.image('sideout%s'% i, self.ske_out[i], collections=['ske'])
            tf.summary.image('ske_gt:', self.ske, collections=['ske'])
        with tf.name_scope('KeyDetect'):
            key_out = key_out[:, 3, :, :, :]    # None x 64 x 64 x 12
            gt = self.htmap[:, 3, :, :, :]  #  None x 64 x 64 x 12
            gt = tf.reduce_sum(gt, axis=3) # None x 64 x 64
            gt = tf.expand_dims(gt, axis=3)     # None x 64 x 64 x 1
            for i in range(key_out.shape[3]):
                tf.summary.image(self.joint_names[i], tf.expand_dims(key_out[:,:,:,i],axis=3), collections=['key'])
            tf.summary.image('key_gt', gt, collections=['key'])
        # scalar
        with tf.name_scope('loss'):
            tf.summary.scalar('Total', self.total_loss, collections=['train'])
            tf.summary.scalar('Ske', self.ske_loss, collections=['train'])
            tf.summary.scalar('Key', self.key_loss, collections=['train'])
        tf.summary.scalar('learning_rate', self.lr, collections=['train'])
        self.loss_summary = tf.summary.merge_all('train')
        self.ske_summary = tf.summary.merge_all('ske')
        self.key_summary = tf.summary.merge_all('key')

    def creat_session(self):
        """ Initialize weights
        """
        print('Session initialization')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['gpu_memory_fraction'],
                                    allow_growth=self.params['allow_growth'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.params['gpu'])
        device_config = tf.ConfigProto(log_device_placement=self.params['log_device_placement'],
                                       allow_soft_placement=self.params['allow_soft_placement'],
                                       gpu_options=gpu_options)
        self.Session = tf.Session(config=device_config)
        t_start = time.time()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def train(self,summary=True,hg_load=None,hed_load = None, skey_load =None):
        self.generate_model()
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self.creat_session()

            # summary
            if (self.params['log_dir_train'] == None):
                raise ValueError('Train/Test directory not assigned')
            else:
                with tf.device(self.cpu):
                    self.HEDWeights = [v for v in self.AllWeights if (
                                'block' in v.name.split('/')[0] or
                                'side' in v.name.split('/')[0] or
                                'fused_side' in v.name.split('/')[0])]
                    self.HGWeights = [v for v in self.AllWeights if ('model' in v.name or 'BatchNorm' in v.name)]
                    self.HGWeights = self.HGWeights[1:]  # delete model/preprocessing/conv_256_to_128/weights, whose shape is (6,6,7,64)
                    # print(HGWeights)
                    # print(HEDWeights)
                    print('---Creat Saver---')
                    self.HGSaver = tf.train.Saver(self.HGWeights)
                    self.HEDSaver = tf.train.Saver(self.HEDWeights)
                    self.saver = tf.train.Saver()
                    print('---Creat Svaer Done---')
                if summary:
                    with tf.device(self.gpu):
                        print('---Add Log File---')
                        self.train_summary = tf.summary.FileWriter(self.params['log_dir_train'], tf.get_default_graph())
                        print('---Add Log File Done')
                        # self.test_summary = tf.summary.FileWriter(self.logdir_test)

            # load
            if hg_load is not None:
                self.HGSaver.restore(self.Session, hg_load)
                print('--load Hourglass weights DONE--')
            if hed_load is not None:
                self.HEDSaver.restore(self.Session, hed_load)
                print('--load HED weights DONE--')
            if skey_load is not None:
                self.saver.restore(self.Session, skey_load)
                print('--load SKEY weights DONE--')


            # train
            nEpochs = self.params['nepochs']
            epoch_size = self.params['epoch_size']
            with tf.name_scope('Train'):
                self.generater = self.dataset.generate(self.params)
                startTime = time.time()
                for epoch in range(nEpochs):
                    epochstartTime = time.time()
                    avg_loss = 0.
                    sum_loss = 0.
                    print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
                    for i in range(epoch_size):
                        block = int(i / epoch_size * 20)
                        # tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                        sys.stdout.flush()
                        img_train, gt_train, weight_train, ske = next(self.generater)
                        _, loss, skeLoss, keyLoss, ske_sum, key_sum, loss_sum= \
                            self.Session.run([self.train_rmsprop,
                                              self.total_loss,self.ske_loss,self.key_loss,
                                              self.ske_summary, self.key_summary,self.loss_summary],
                                             feed_dict={self.img: img_train, self.htmap:gt_train, self.ske:ske})
                        # hedout = self.Session.run(self.ske_out, feed_dict={self.img: img_train, self.htmap:gt_train, self.ske:ske})
                        # print(hedout)
                        sum_loss += loss

                        if i % self.params['save_step'] == 0:
                            self.train_summary.add_summary(key_sum,epoch*epoch_size + i)
                            self.train_summary.add_summary(ske_sum,epoch*epoch_size + i)
                            self.train_summary.add_summary(loss_sum,epoch*epoch_size + i)
                            self.train_summary.flush()

                        print('\rTraining:|{0}{1}|'.format('=' * block, ' ' * (20 - block))
                              + '%0.3s%%' % (i / epoch_size * 100)
                              + '    loss: %0.3f+%d*%0.3f+%0.3f=%0.3f'% (skeLoss,self.params['bias_weight'], keyLoss, loss-skeLoss-self.params['bias_weight']*keyLoss, loss), end='')
                    avg_loss = sum_loss/epoch_size
                    print('\navrage loss: %0.3f'% avg_loss)
                    self.saver.save(self.Session,os.path.join(self.params['weight_saved'],'skey_1219'),global_step=epoch)

    def pred(self,test_num, out_path=None,load_path=None):
        self.generate_model()
        key_out = self.key_out
        pred_sigmoid = tf.nn.sigmoid(key_out[:, self.params['nstacks'] - 1], name='sigmoid_final_prediction')
        joint_tensor_final = self._create_joint_tensor(key_out[0, -1], name='joint_tensor_final')
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self.creat_session()

        self.saver = tf.train.Saver()
        if load_path is None:
            raise ValueError('No weights input')
        else:
            self.saver.restore(self.Session, load_path)
            print('---load weights done----')

        generator = self.dataset.generate(self.params, mode='test')

        heatmaps = {}
        joints = {}
        skeletons = {}
        pic_count = 0

        for i in range(1, test_num):
            num = np.int(20 * i /test_num)

            print('\rDeploying in Test Sets:{0}{1}'.format('='*num,' '*(20-num)) + '|%d%%'%(i/test_num*100), end='')
                # print('deployed %d' % i)
            # heatmap
            img, gtmap, visi_weights, ske, name = next(generator)
            # if img.shape == (256, 256, 3):
            out, j,skeout = self.Session.run([pred_sigmoid, joint_tensor_final,self.ske_out], feed_dict={self.img: img})
            # else:
            #     print('Image Size does not match placeholder shape')
            #     continue
            # skeletons[name] = skeout
            fuse_side = skeout[5]
            fuse_side = np.squeeze(fuse_side)
            # fuse_side: 256 x 256 x 4
            for c in range(fuse_side.shape[2]):
                cv2.imwrite(os.path.join(out_path,'ske','c%d_%s'%(c,name)),self.hed.hed_post_prosess(fuse_side[:,:,c]))
            # print(os.path.join(out_path,'ske',name),fuse_side.shape)
            heatmaps[name] = out
            joints[name] = j
            pic_count += 1
        # np.save(os.path.join(out_path,'heatmaps.npy'),heatmaps)
        # np.save(os.path.join(out_path,'joints.npy'),joints)
        # np.save(os.path.join(out_path,'ske.npy'),skeletons)
        print('\n%d was inferenced' % pic_count)


    def _create_joint_tensor(self, tensor, name='joint_tensor', debug=False):
        """ TensorFlow Computation of Joint Position
        Args:
            tensor		: Prediction Tensor Shape [nbStack x 64 x 64 x outDim] or [64 x 64 x outDim]
            name		: name of the tensor
        Returns:
            out			: Tensor of joints position

        Comment:
            Genuinely Agreeing this tensor is UGLY. If you don't trust me, look at
            'prediction' node in TensorBoard.
            In my defence, I implement it to compare computation times with numpy.
        """
        with tf.name_scope(name):
            shape = tensor.get_shape().as_list()
            if debug:
                print(shape)
            if len(shape) == 3:
                resh = tf.reshape(tensor[:, :, 0], [-1])
            elif len(shape) == 4:
                resh = tf.reshape(tensor[-1, :, :, 0], [-1])
            if debug:
                print(resh)
            arg = tf.arg_max(resh, 0)
            if debug:
                print(arg, arg.get_shape(), arg.get_shape().as_list())
            joints = tf.expand_dims(tf.stack([arg // tf.to_int64(shape[1]), arg % tf.to_int64(shape[1])], axis=-1), axis=0)
            for i in range(1, shape[-1]):
                if len(shape) == 3:
                    resh = tf.reshape(tensor[:, :, i], [-1])
                elif len(shape) == 4:
                    resh = tf.reshape(tensor[-1, :, :, i], [-1])
                arg = tf.arg_max(resh, 0)
                j = tf.expand_dims(tf.stack([arg // tf.to_int64(shape[1]), arg % tf.to_int64(shape[1])], axis=-1), axis=0)
                joints = tf.concat([joints, j], axis=0)
            return tf.identity(joints, name='joints')








if __name__ == '__main__':
    params = config.parser_config('config.cfg')
    img_dir = params['img_dir']
    npy_path = params['npy_path']
    data = data_process.Data(img_dir=img_dir, npy_path=npy_path)
    data.readnpy()
    skey = SKEY(params, data)
    skey.train(summary=False)