import pandas
import os
import cv2
import numpy as np
import config
import random
from skimage import transform
import time

class Data:

    '''
    INPUT:
    img_dir: directory where img is put
    txt_path: if format is written in .txt, input the path
    csv_path: if format is written in .csv, input the path
    '''
    def __init__(self, img_dir, txt_path='', csv_path='', npy_path='',pic_list=None):
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.csv_path = csv_path
        self.npy_path = npy_path
        if pic_list is not None:
            piclist = []
            with open(pic_list, 'r') as f:
                for line in f:
                    cur = line.strip()
                    piclist.append(cur)
            self.pic_list = piclist
            print('Picture list lenghth: %d' % len(self.pic_list))
        else:
            self.pic_list = None
            print('Picture list lenghth: auto')
        # self.bbox_path = bbox_path
        self.joints_list = ['head', 'left_wing_tip', 'left_wing_front', 'right_wing_front', 'right_wing_tip', 'right_wing_back',
                            'left_wing_back','tail_tip', 'tail_left', 'tail_up', 'tail_right', 'tail_front']
        self.rgbcolor = [(234,4,55),    #红
                        (183,211,0),   #绿
                        (255,209,0),   #黄
                        (79,31,145),
                        (162,76,200),
                        (215,0,108),
                        (255,144,0),
                        (0,157,217),
                        (120,199,232),
                        (188,168,230),
                        (124,34,48),
                        (0,123,99),
                        (242,147,209),
                        (127,120,0),
                        (187,167,134)]
        self.bgr_color = [(55,4,234),
                        (0,211,183),
                        (0,209,255),
                        (145,31,79),
                        (200,76,162),
                        (108,0,215),
                        (0,144,255),
                        (217,157,0),
                        (232,199,120),
                        (230,168,188),
                        (48,34,124),
                        (99 ,123,0),
                        (209,147,242),
                        (0,120,127),
                        (134,167,187)]

    '''
    DESCRIPTION:
    - read csv format label file
    INPUT:
    - bbox_path: bbox label txt path in FGVR
    OUTPUT:
    - self.point_dict
    - self.attr_dictt
    - self.pic_list
    - self.bbox_dict
    - self.pic_list: picnames in csv file
    
    # data structure:
    # point_dict: key named by image name, value is 12 point location(24 values) saved in it
    # attr_dict: key named by image name, value is 12 point attribution saved in it.
    # attr value: 0 for visible, 1 for invisible, 2 for out of image
    '''
    def readcsv(self):
        csv_path = self.csv_path
        data = pandas.read_csv(csv_path)    # dataFrame
        point_dict = {}
        attr_dict = {}
        total_dict = {}
        lines_num = len(data.index)
        print('--Loading csvfile.')
        pic_list = []
        for index in range(lines_num):
            name = data.loc[index, 'filename']
            region_id = data.loc[index,'region_id']
            if region_id == 0:
                point_list = []
                point_dict[name] = point_list
                attr_list = []
                attr_dict[name] = attr_list
                total_dict[name] = {'joints': point_list, 'weights': attr_list}
                pic_list.append(name)
            else:
                point_list = point_dict[name]
                attr_list = attr_dict[name]
            point = eval(data.loc[index, 'region_shape_attributes'])
            px = str(point['cx'])
            py = str(point['cy'])
            point_list.append(px)
            point_list.append(py)

            point_attr = eval(data.loc[index,'region_attributes'])
            keys = point_attr.keys()
            attr = '0'
            if 'type' in keys:
                if point_attr['type'] == 'unknown':
                    attr = '1'
            if 'name' in keys:
                if point_attr['name'] == 'out':
                    attr = '2'
            attr_list.append(attr)
            # print(attr_dict)
            # print(point_dict)
        pic_num1 = len(point_dict.keys())
        pic_num2 = len(attr_dict.keys())
        if pic_num1 == pic_num2:
            print('%d pictures detected.' % pic_num1)
        else:
            print('point is not equal to attrs.')
        self.point_dict = point_dict
        self.attr_dict = attr_dict
        self.pic_list = pic_list
        self.bbox_dict = self.read_bbox(self.bbox_path)


    def readnpy(self):
        print('READ DATA')
        data = np.load(self.npy_path, allow_pickle=True)
        data_num = data.shape[0]
        point_dict = {}
        attr_dict = {}
        box_dict = {}
        total_dict = {}
        pic_list = []
        wrong_pic = []
        for index in range(data_num):
            point_list = []
            attr_list = []
            pic_data = data[index]
            keypoints = pic_data['keypoint']
            name = keypoints[0]['img_name']
            if len(keypoints) != 12:
                wrong_pic.append(name)
                continue
            for i in range(len(keypoints)):
                data_dict = keypoints[i]
                if data_dict['point_name'] != self.joints_list[i]:
                    print('wrong sequnce')
                    wrong_pic.append(name)
                    continue
                cx = str(data_dict['x'])
                cy = str(data_dict['y'])
                attr = '0'
                if data_dict['visible'] == 0:
                    attr = '1'
                if data_dict['outside'] == 1:
                    attr = '2'
                point_list.append(cx)
                point_list.append(cy)
                attr_list.append(attr)
            box = pic_data['box']
            xmin = str(int(box['xmin']))
            ymin = str(int(box['ymin']))
            xmax = str(int(box['xmax']))
            ymax = str(int(box['ymax']))
            box_n = [xmin, ymin, xmax, ymax]
            pic_list.append(name)
            point_dict[name] = point_list
            attr_dict[name] = attr_list
            box_dict[name] = box_n
            total_dict[name] = {'joints': point_list, 'weights': attr_list}
        self.point_dict = point_dict
        self.attr_dict = attr_dict
        self.bbox_dict = box_dict
        if self.pic_list is None:
            self.pic_list = pic_list
        print('%d image totally' % data.shape[0])
        print('%d image saved in dict' % len(pic_list))
        print('%d wrong image' % len(wrong_pic))
        if len(wrong_pic) != 0:
            print(wrong_pic)
        # print(box_dict)
        # print(data)



    # Load bounding box from FGVC
    def read_bbox(self, bbox_path):
        bbox_dict = {}
        with open(bbox_path, 'r') as f:
            for line in f:
                cur_line = line.split(' ')
                name = cur_line[0] + '.jpg'
                xmin = cur_line[1]
                ymin = cur_line[2]
                xmax = cur_line[3]
                ymax = cur_line[4].strip()
                bbox_dict[name] = [xmin, ymin, xmax, ymax]
        return bbox_dict

    # check if keypoints are all visible
    def check_visible(self, pic_name):
        attrs = self.attr_dict[pic_name]
        for i in attrs:
            if i != 0:
                return False
        return True

    '''
    generate txt file from self.dicts
    INPUT:
    - save_path, file_name
    - pic_list: if not specifically assigned, pic_list assigned by csv file
    
    !!! ONLY filename in pic_list will be saved in the txtfile
    '''
    def totxt(self, save_path,file_name ='total', pic_list='default'):
        if pic_list == 'default':
            pic_list = self.pic_list
        point_dict = self.point_dict
        attr_dict = self.attr_dict
        bbox_dict = self.bbox_dict
        print('--Wring txt file')
        file_name = file_name + '.txt'
        point_path = os.path.join(save_path, file_name)
        with open(point_path, 'w') as f:
            for index in pic_list:
                name = index
                points = point_dict[name]
                bbox = bbox_dict[name]
                # check
                if len(points) != 24:
                    print('file %s point number wrong' % name)
                    break
                line = name + ' ' + ' '.join(bbox) + ' ' +' '.join(points) + '\n'
                f.write(line)

        # attr_path = os.path.join(save_path, 'attr.txt')
        # with open(attr_path, 'w') as f:
        #     for index in pic_list:
        #         name = index
        #         attrs = attr_dict[name]
        #         bbox = bbox_dict[name]
        #         if len(attrs) != 12:
        #             print('file %s point number wrong' % name)
        #             break
        #         line = name + ' ' + ' '.join(points) + '\n'
        #         f.write(line)
        #         line = name + ' ' +' '.join(bbox) + ' '.join(attrs) + '\n'
        #         f.write(line)
        print('%d items written successfully.' % len(pic_list))

    def create_sets(self,validation_rate = 0.1):
        """ Create train and validation set
            Select Elements to feed training and validation set
        		Args:
        			validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        		"""
        sample = len(self.pic_list)
        valid_sample = int(sample * validation_rate)
        self.train_set = self.pic_list[:sample - valid_sample]
        self.valid_set = []
        preset = self.pic_list[sample - valid_sample:]
        print('START SET CREATION')
        # if some points are invisible in validation set, put it in train set
        for elem in preset:
            if self.check_visible(elem):
                self.valid_set.append(elem)
            else:
                self.train_set.append(elem)
        print('SET CREATED')
        np.save('Dataset-Validation-Set', self.valid_set)
        np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

    def attr2visi_weights(self, attrs):
        visi_weights = []
        for i in attrs:
            if i == '0':
                visi_weights.append(1)
            else:
                visi_weights.append(0)
        return visi_weights

    def _makeGaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        # y = np.arange(0, height, 1, float)
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def generate_hm(self,height, width, joints, max_length, weights, if_occlusion):
        '''
        :param height:
        :param width:
        :param joints:
        :param max_length:
        :param weights:
        :param if_occlusion: if true, generate a heatmap even if point is occluded
        :return:
        '''
        num_joints = len(self.joints_list)
        hm = np.zeros((height, width, num_joints), dtype=np.float32)
        for i in range(num_joints):
            if if_occlusion:
                s = int(np.sqrt(max_length) * max_length * 10 / 4096) + 2
                hm[:, :, i] = self._makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
            else:
                if weights[i] == 1:
                    s = int(np.sqrt(max_length) * max_length * 10 / 4096) + 2
                    # a = self._makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
                    # cv2.imshow('heatmap',a)
                    # cv2.waitKey()
                    hm[:, :, i] = self._makeGaussian(height, width, sigma=s, center=(joints[i,0], joints[i,1]))
                else:
                    hm[:, :, i] = np.zeros((height, width))
        return hm

    def _crop_data(self, height, width, box, joints, boxp=0):
        """ Automatically returns a padding vector and a bounding box given
        the size of the image and a list of joints.
        Args:
            height		: Original Height
            width		: Original Width
            box			: Bounding Box
            joints		: Array of joints
            boxp		: Box percentage (Use 20% to get a good bounding box)
        """
        padding = [[0, 0], [0, 0], [0, 0]]
        j = np.copy(joints)
        if box[0:2] == [-1, -1]:
            j[joints == -1] = 1e5
            box[0], box[1] = min(j[:, 0]), min(j[:, 1])
        crop_box = [box[0] - int(boxp * (box[2] - box[0])), box[1] - int(boxp * (box[3] - box[1])),
                    box[2] + int(boxp * (box[2] - box[0])), box[3] + int(boxp * (box[3] - box[1]))]
        if crop_box[0] < 0: crop_box[0] = 0
        if crop_box[1] < 0: crop_box[1] = 0
        if crop_box[2] > width - 1: crop_box[2] = width - 1
        if crop_box[3] > height - 1: crop_box[3] = height - 1
        new_h = int(crop_box[3] - crop_box[1])
        new_w = int(crop_box[2] - crop_box[0])
        crop_box = [crop_box[0] + new_w // 2, crop_box[1] + new_h // 2, new_w, new_h]
        if new_h > new_w:
            bounds = (crop_box[0] - new_h // 2, crop_box[0] + new_h // 2)
            if bounds[0] < 0:
                padding[1][0] = abs(bounds[0])
            if bounds[1] > width - 1:
                padding[1][1] = abs(width - bounds[1])
        elif new_h < new_w:
            bounds = (crop_box[1] - new_w // 2, crop_box[1] + new_w // 2)
            if bounds[0] < 0:
                # padding[0][1] = abs(bounds[0])
                padding[0][0] = abs(bounds[0])        #old
            # if bounds[1] > width - 1:
            if bounds[1] > height - 1:
                # padding[0][0] = abs(height - bounds[1])
                padding[0][1] = abs(height - bounds[1])   #old
        '''
        padding[0][0]: up pad
        padding[0][1]: down pad
        padding[1][0]: left pad
        padding[1][1]: right pad
        box center will shifts after padding
        '''
        crop_box[0] += padding[1][0]
        crop_box[1] += padding[0][0]
        return padding, crop_box

    def _crop_img(self, img, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            img			: Source Image
            padding	: Padding
            crop_box	: Bounding Box
        """
        img_origin = img
        img = np.pad(img, padding, mode='constant')
        # in np.pad, up and down is converse with picture
        max_lenght = max(crop_box[2], crop_box[3])
        # img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
        #       crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]    #old
        img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
              crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        return img


    '''
    - new_j_big: joints in big map, for visualization
    - new_j_small: joints to generate heatmap
    '''
    def _relative_joints(self, box, padding, joints, to_size=64):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        Args:
            box			: Bounding Box
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(joints)
        max_l = max(box[2], box[3])
        new_j = new_j + [padding[1][0], padding[0][0]]
        new_j_big = new_j - [box[0] - max_l // 2, box[1] - max_l // 2]
        new_j_small = new_j * to_size / (max_l + 0.0000001)
        return new_j_small.astype(np.int32),new_j_big.astype(np.int32)




    def _augment(self, img, hm, max_rotation=30):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        if random.choice([0, 1]):
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            img = transform.rotate(img, r_angle, preserve_range=True)
            hm = transform.rotate(hm, r_angle)
        return img, hm

    def str2num_list(self, list):
        ans = []
        for item in list:
            x = int(item)
            ans.append(x)
        return ans


    def generate(self,params,mode='train'):
        '''
        output:
        img: batch_size x 256 x 256 x 3
        heatmap: batch_size x stacks x 64 x 64 x point_num(12)
        visual_weights: batch_szie x point_num   1 for visible, 0 for invisible
        ske: batch_size x 256 x 256 x 1
        name: if mode is test, return batch_size=1, name for this pic
        '''
        if mode == 'train':
            batch_size = params['batch_size']
        else:
            batch_size = 1
        stacks = params['nstacks']
        normalize = True
        sample_set = 'Train'
        test_index = 0
        while True:
            train_img = np.zeros((batch_size, 256, 256, 3), dtype='float32')
            train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.joints_list)), np.float32)
            visi_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
            ske = np.zeros((batch_size,256,256,params['outchannel']), dtype='float32')
            i = 0
            for i in range(batch_size):
                if mode == 'train':
                    name = random.choice(self.pic_list)
                else:
                    name = self.pic_list[test_index]
                    test_index += 1
                # try:
                joints = self.str2num_list(self.point_dict[name])
                joints = np.reshape(joints, (-1,2))
                attrs = self.attr_dict[name]
                bbox = self.str2num_list(self.bbox_dict[name])
                visi_weights[i] = self.attr2visi_weights(attrs)
                img = cv2.imread(os.path.join(self.img_dir, name))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # --------------------old-------------------------
                # padd, cbox = self._crop_data(img.shape[0], img.shape[1], bbox, joints, boxp=0)
                # new_j, vi_j = self._relative_joints(cbox, padd, joints, to_size=64)
                # # new_j = self.newj_resize(joints,img.shape, tosize=64)
                # hm = self.generate_hm(64, 64, new_j, 64, visi_weights[i])
                # img = self._crop_img(img, padd, cbox)
                # img = img.astype(np.uint8)
                # img = cv2.resize(img, (256, 256))
                # img, hm = self._augment(img, hm)
                # ---------------------new 1: resize---------------------------------------------
                # crop = self.crop_new(img,bbox)
                # crop = cv2.resize(crop,(256,256))
                # new_j = self.Raw2CropResize_joints(name,bbox,tosize=64)
                # ske_j = self.Raw2CropResize_joints(name,bbox,tosize=256)

                # --------------------new 2: pad and resize------------------------
                crop = self.padcrop_new(img,bbox)
                crop = cv2.resize(crop,(256,256))
                cv2.imwrite(os.path.join(r'D:\Airplane Keypart\Dataset\FRVC\data\resize', name), crop)
                new_j = self.Raw2PadCrop_joints(name,bbox,tosize=64)
                ske_j = self.Raw2PadCrop_joints(name, bbox, 256, debug=False)

                # hm Dim: 64 x 64 x 12
                hm = self.generate_hm(64, 64, new_j, 64, visi_weights[i],if_occlusion=params['if_occluded'])

                # -------------flip-------------
                # if mode == 'train':
                #     crop, hm = self.random_flip(crop,hm)
                hm = np.expand_dims(hm, axis=0)
                hm = np.repeat(hm, stacks, axis=0)
                if normalize:
                    train_img[i] = crop.astype(np.float32) / 255
                else:
                    train_img[i] = crop.astype(np.float32)
                train_gtmap[i] = hm

                # ------------generate by algorithm--------------
                ske[i] = self.generate_ske(name,ske_j,outchannel=params['outchannel'],method='method2')
                # ------------generate by reading val pics------------
                # ske_j = cv2.imread(os.path.join(r'D:\Airplane Keypart\skey_data\HED\output_1218\ske',name))
                # ske_j = cv2.cvtColor(ske_j, cv2.COLOR_BGR2GRAY)
                # ske[i] = np.expand_dims(ske_j,axis=2)

                ske[i] = ske[i]/255
                # except:
                #     print('error file:', name)
                if mode == 'test':
                    yield train_img, train_gtmap, visi_weights, ske, name
            if mode == 'train':
                yield train_img, train_gtmap, visi_weights, ske

    def random_flip(self,img,hm):
        # cv2.imshow('raw',img)
        flag = random.choice([0,1])
        # flag = 1
        # print(flag)
        if flag:
            img = np.flip(img,axis=1)
            hm = np.flip(hm,axis=1)
            hm[:,:,[2,3]] = hm[:,:,[3,2]]
            hm[:,:,[1,4]] = hm[:,:,[4,1]]
            hm[:,:,[6,5]] = hm[:,:,[5,6]]
            hm[:,:,[8,10]] = hm[:,:,[10,8]]
        # cv2.imshow('new',img)
        # cv2.waitKey()
        return img, hm

    '''
    生成关键点热力图并可视化或存储
    cbox: coordinate in raw image
    '''
    def test_hm(self,tosize=700, mode='visual',save_path = None):
        if mode == 'save':
            if save_path == None:
                print('need saved path')
                return 0
            print('%d pictures to be saved' % len(self.pic_list))

        save_count = 0
        fail_save = []
        for i in self.pic_list:
            name = i
            joints = self.str2num_list(self.point_dict[name])
            joints = np.reshape(joints, (-1, 2))
            attrs = self.attr_dict[name]
            bbox = self.str2num_list(self.bbox_dict[name])
            visi_weights = self.attr2visi_weights(attrs)
            img = cv2.imread(os.path.join(self.img_dir, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #crop and pad
            padd, cbox = self._crop_data(img.shape[0], img.shape[1], bbox, joints, boxp=0)
            new_j, vi_j = self._relative_joints(cbox, padd, joints, to_size=tosize)
            hm = self.generate_hm(tosize, tosize, new_j, tosize, visi_weights)
            img = self._crop_img(img, padd, cbox)
            img = img.astype(np.uint8)
            visual = np.zeros_like(img)
            for i in range(len(joints)):
                (x,y) = vi_j[i]
                if visi_weights[i] == 0:
                    # visual = cv2.circle(img, (x, y), radius=3, color=(255, 0, 0), thickness=8)
                    continue
                visual = cv2.circle(img, (x,y), radius=3, color=(0, 0, 255),thickness=8)
            visual = cv2.resize(visual, (tosize, tosize))
            heat_maps = np.sum(hm, axis=2)

            if mode == 'visual':
                cv2.imshow('heatmaps', heat_maps)
                cv2.imshow('image', visual)
                cv2.imshow('raw',img/255)
                # Wait
                time.sleep(2)
                if cv2.waitKey(1) == 27:
                    print('Ended')
                    cv2.destroyAllWindows()
                    break
            elif mode == 'save':
                # heat = os.path.join(save_path['heat'],name)
                visu = os.path.join(save_path['visu'],name)
                # cv2.imwrite(heat, heat_maps)
                flag = cv2.imwrite(visu, visual)
                if flag == False:
                    fail_save.append(name)
                else:
                    save_count = save_count + 1
                print(flag, save_count)
        print(fail_save)


    def data_wash(self, tosize = 700):
        '''
        visualize picture, heatmaps, skeletion
        - press esc to quit
        - press enter to save this pic
        - press others to skip this pic
        :param tosize:
        :return:
        '''
        pick_count = 0
        print('Press Esc to quit')
        print('Press Enter to save this pic')
        print('Press others to skip this pic')
        name_list = []
        for name in self.pic_list:
            joints = self.str2num_list(self.point_dict[name])
            joints = np.reshape(joints, (-1, 2))
            attrs = self.attr_dict[name]
            bbox = self.str2num_list(self.bbox_dict[name])
            visi_weights = self.attr2visi_weights(attrs)
            img = cv2.imread(os.path.join(self.img_dir, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # crop and pad
            padd, cbox = self._crop_data(img.shape[0], img.shape[1], bbox, joints, boxp=0)
            new_j, vi_j = self._relative_joints(cbox, padd, joints, to_size=tosize)
            hm = self.generate_hm(tosize, tosize, new_j, tosize, visi_weights)
            img = self._crop_img(img, padd, cbox)
            img = img.astype(np.uint8)
            visual = np.zeros_like(img)
            for i in range(len(joints)):
                (x, y) = vi_j[i]
                if visi_weights[i] == 0:
                    # visual = cv2.circle(img, (x, y), radius=3, color=(255, 0, 0), thickness=8)
                    continue
                visual = cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=8)
            visual = cv2.resize(visual, (tosize, tosize))
            heat_maps = np.sum(hm, axis=2)
            ske = self.visual_ske(name,tosize=tosize)
            cv2.imshow('heatmaps', heat_maps)
            cv2.imshow('image', visual)
            cv2.imshow('skeleton',ske)
            # cv2.imshow('raw', img / 255)
            key = cv2.waitKey()
            # print(key)
            # esc
            if key == 27:
                print('End')
                break
            # enter
            if key == 13:
                name_list.append(name)
                pick_count = pick_count +1
                # cv2.destroyAllWindows()
            else:
                # cv2.destroyAllWindows()
                continue
            print('%d/1000 pics picked' % pick_count)
        print('%d pics picked' % pick_count)
        with open('pic_list.txt', 'w') as f:
            for name in name_list:
                f.write(name+'\n')

    def crop_new(self,img,bbox):
        '''
        directly crop img, no padding, no making it square
        :param img:
        :param bbox:
        :return:
        '''
        return img[bbox[1]:bbox[3],bbox[0]:bbox[2]]

    def padcrop_new(self,img,inbbox):
        # TODO: debug:0103328.jpg minus pad
        bbox = inbbox.copy()
        rawheight, rawwidth, ch = img.shape
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        box_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        if width > height:
            bbox[1] = int(box_center[1] - width / 2)
            bbox[3] = int(box_center[1] + width / 2)
        else:
            bbox[0] = int(box_center[0] - height / 2)
            bbox[2] = int(box_center[0] + height / 2)
        pad = np.zeros((6), dtype='int32')
        if bbox[0] < 0:
            pad[2] = abs(bbox[0])
            bbox[0] += pad[2]
            bbox[2] += pad[2]
        if bbox[1] < 0:
            pad[0] = abs(bbox[1])
            bbox[1] += pad[0]
            bbox[3] += pad[0]
        if bbox[2] > (rawwidth + pad[2]):
            pad[3] = bbox[2] - pad[2] - rawwidth
        if bbox[3] > (rawheight + pad[0]):
            pad[1] = bbox[3] - pad[0] - rawheight
        pad = pad.reshape((3,2))
        # print(pad)
        paded = np.pad(img, pad, mode='mean')
        croped = self.crop_new(paded, bbox)
        return croped




    def Raw2CropResize_joints(self,name,bbox=None,tosize=256,debug=False):
        '''
        get relative joints coordinate after crop and resize
        coordinate is in Cartesian mode
        :param name:
        :param bbox:
        :param tosize:
        :param debug:
        :return:
        '''
        img = cv2.imread(os.path.join(self.img_dir, name))
        height, width, c = img.shape
        raw_j = self.str2num_list(self.point_dict[name])
        raw_j = np.reshape(raw_j, (-1, 2))
        new_j = np.zeros_like(raw_j)
        if debug:
            rawDraw = img.copy()
            re = cv2.resize(img, (tosize, tosize))
            cv2.circle(rawDraw,tuple(raw_j[0]),radius=2,color=self.bgr_color[0],thickness=3)
            cv2.imshow('raw',rawDraw)
            cv2.waitKey()
        if bbox == None:
            for index in range(raw_j.shape[0]):
                new_j[index][0] = int(raw_j[index][0] * tosize / width)
                new_j[index][1] = int(raw_j[index][1] * tosize / height)
        else:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            for index in range(raw_j.shape[0]):
                # coord after crop
                new_j[index][0] = int(raw_j[index][0] - bbox[0])
                new_j[index][1] = int(raw_j[index][1] - bbox[1])
                new_j[index][0] = int(new_j[index][0] * tosize / width)
                new_j[index][1] = int(new_j[index][1] * tosize / height)
        return new_j

    def Raw2PadCrop_joints(self,name,inbbox=None,tosize=256,debug=False):
        '''
        Crop after padding, return joints coord, joints origin is left-up
        :param name:
        :param bbox:
        :param tosize:
        :param debug:
        :return:
        '''
        bbox = inbbox.copy()
        img = cv2.imread(os.path.join(self.img_dir, name))
        rawheight, rawwidth, c = img.shape
        raw_j = self.str2num_list(self.point_dict[name])
        raw_j = np.reshape(raw_j, (-1, 2))
        new_j = raw_j.copy()

        # compute pad and joints coord
        if bbox == None:
            if rawheight > rawwidth:
                pad_w = int((rawheight - rawwidth) / 2)
                res = rawheight - pad_w * 2
                pad = ((0, 0), (pad_w, pad_w + res), (0, 0))
                new_j[:, 0] = raw_j[:, 0] + pad_w
                new_j[:, 1] = raw_j[:, 1]
                if bbox is not None:
                    bbox[0] += pad_w
                    bbox[2] += pad_w
            else:
                pad_w = int((rawwidth - rawheight) / 2)
                res = rawwidth - pad_w * 2
                pad = ((pad_w, pad_w + res), (0, 0), (0, 0))
                new_j[:, 0] = raw_j[:, 0]
                new_j[:, 1] = raw_j[:, 1] + pad_w
                if bbox is not None:
                    bbox[1] += pad_w
                    bbox[3] += pad_w
            for index in range(new_j.shape[0]):
                new_j[index][0] = int(new_j[index][0] * tosize / rawwidth)
                new_j[index][1] = int(new_j[index][1] * tosize / rawheight)
        else:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            box_center = (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))
            if width > height:
                bbox[1] = int(box_center[1] - width/2)
                bbox[3] = int(box_center[1] + width/2)
            else:
                bbox[0] = int(box_center[0] - height/2)
                bbox[2] = int(box_center[0] + height/2)
            pad = np.zeros((6),dtype='int32')
            if bbox[0]<0:
                pad[2] = abs(bbox[0])
                bbox[0] += pad[2]
                bbox[2] += pad[2]
                new_j[:,0] += pad[2]
            if bbox[1]<0:
                pad[0] = abs(bbox[1])
                bbox[1] += pad[0]
                bbox[3] += pad[0]
                new_j[:,1] += pad[0]
            if bbox[2]>(rawwidth+pad[2]):
                pad[3] = bbox[2]- pad[2] - rawwidth
            if bbox[3]>(rawheight+pad[0]):
                pad[1] = bbox[3] -pad[0] - rawheight
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            pad = pad.reshape((3,2))
            for index in range(new_j.shape[0]):
                # coord after crop
                new_j[index][0] = int(new_j[index][0] - bbox[0])
                new_j[index][1] = int(new_j[index][1] - bbox[1])
                # coord after resize
                new_j[index][0] = int(new_j[index][0] * tosize / width)
                new_j[index][1] = int(new_j[index][1] * tosize / height)
        if debug:
            print('---Debug:Raw2PadCrop_joints')
            paded = np.pad(img,pad,mode='mean')
            croped = self.crop_new(paded,bbox)
            resize = cv2.resize(croped,(256,256))
            # cv2.rectangle(paded,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=(0,0,255),thickness=3)
            cv2.circle(resize,tuple(new_j[0]),radius=2,color=(0,0,255),thickness=2)
            cv2.imshow('p',resize)
            cv2.waitKey()
        return new_j

    def generate_ske(self, name, joints,outchannel, tosize=256, debug=True,method='method1'):
        '''
        joints: shape(12,2), coordinate in tosize img
        :param name:
        :param joints:
        :param tosize:
        :param debug:
        :param method:
        :return:
        '''
        if method == 'method1':
            ske = self.generate_ske_method1(name,joints,outchannel,tosize,debug)
        elif method == 'method2':
            ske = self.generate_ske_method2(name,joints,outchannel,tosize,debug)
        elif method == 'method3':
            ske = self.generate_ske_method3(name,joints,outchannel,tosize,debug)
        else:
            ske = None
            print('error: input correct method')
        # ske = np.expand_dims(ske,axis=2)
        # ske Dim: 256 x 256 x 1
        return ske


    # enter a single name and return the skeleton
    def generate_ske_method1(self,name,joints,outchannel,tosize=256,debug=True):
        ske_color = (255, 255, 255)
        ske_joints = joints
        ske_attr = self.attr_dict[name]
        # for attr in ske_attr:
        #     if attr == '2':
        #         return -1
        # img = cv2.imread(os.path.join(self.img_dir, name))
        total = np.zeros((tosize, tosize), np.uint8)
        left_wing = np.zeros((tosize, tosize), np.uint8)
        right_wing = np.zeros((tosize, tosize), np.uint8)
        body = np.zeros((tosize, tosize), np.uint8)
        center = self.get_plane_center(ske_joints)
        # ske = cv2.circle(ske, center, radius=2, color=ske_color, thickness=5)

        total = cv2.line(total, center, tuple(ske_joints[0]), color=ske_color, thickness=1)
        total = cv2.line(total, center, tuple(ske_joints[1]), color=ske_color, thickness=1)
        total = cv2.line(total, center, tuple(ske_joints[4]), color=ske_color, thickness=1)
        total = cv2.line(total, center, tuple(ske_joints[7]), color=ske_color, thickness=1)
        total = np.expand_dims(total, axis=2)
        if outchannel == 4:
            body = cv2.line(body, center, tuple(ske_joints[0]), color=ske_color, thickness=1)
            body = cv2.line(body, center, tuple(ske_joints[7]), color=ske_color, thickness=1)
            body = np.expand_dims(body, axis=2)

            left_wing = cv2.line(left_wing, center, tuple(ske_joints[1]), color=ske_color, thickness=1)
            left_wing = np.expand_dims(left_wing, axis=2)
            right_wing = cv2.line(right_wing, center, tuple(ske_joints[4]), color=ske_color, thickness=1)
            right_wing = np.expand_dims(right_wing, axis=2)
            ske = np.concatenate((total, body, left_wing, right_wing), axis=2)
            return ske
        elif outchannel == 1:
            ske = total
            return ske
        else:
            print('only support 1 or 4 channels output')
            return None

    def generate_ske_method2(self,name,joints,outchannel,tosize=256,debug=True):
        ske_color = (255, 255, 255)
        ske_joints = joints
        # ske_attr = self.attr_dict[name]
        # for attr in ske_attr:
        #     if attr == '2':
        #         return -1
        # img = cv2.imread(os.path.join(self.img_dir, name))

        total = np.zeros((tosize, tosize), np.uint8)
        left_wing = np.zeros((tosize, tosize), np.uint8)
        right_wing = np.zeros((tosize, tosize), np.uint8)
        body = np.zeros((tosize, tosize), np.uint8)
        body_joints = np.asarray([ske_joints[2],ske_joints[3],ske_joints[5],ske_joints[6]])
        center = np.average(body_joints,axis=0)
        center = tuple(center.astype(np.int))
        # ske = cv2.circle(ske, center, radius=2, color=ske_color, thickness=5)

        total = cv2.line(total, center, tuple(ske_joints[0]), color=ske_color, thickness=1)
        total = cv2.line(total, center, tuple(ske_joints[1]), color=ske_color, thickness=1)
        total = cv2.line(total, center, tuple(ske_joints[4]), color=ske_color, thickness=1)
        total = cv2.line(total, center, tuple(ske_joints[7]), color=ske_color, thickness=1)
        total = np.expand_dims(total,axis=2)
        if outchannel == 4:
            body = cv2.line(body, center, tuple(ske_joints[0]), color=ske_color, thickness=1)
            body = cv2.line(body, center, tuple(ske_joints[7]), color=ske_color, thickness=1)
            body = np.expand_dims(body,axis=2)

            left_wing = cv2.line(left_wing, center, tuple(ske_joints[1]), color=ske_color, thickness=1)
            left_wing = np.expand_dims(left_wing,axis=2)
            right_wing = cv2.line(right_wing, center, tuple(ske_joints[4]), color=ske_color, thickness=1)
            right_wing = np.expand_dims(right_wing,axis=2)
            ske = np.concatenate((total, body, left_wing, right_wing), axis=2)
            return ske
        elif outchannel == 1:
            ske = total
            return ske
        else:
            print('only support 1 or 4 channels output')
            return None



    def generate_ske_method3(self, name, joints, tosize=256, debug=True):
        gt_path = r'D:\Airplane Keypart\Dataset\FRVC\data\gaussan_ske'
        ske = cv2.imread(os.path.join(gt_path,name))
        ske = cv2.cvtColor(ske, cv2.COLOR_BGR2GRAY)
        ske = np.squeeze(ske)
        return ske



    def generate_GuassSke(self,name,joints,tosize=256,debug=True,mode='wings'):
        SavePath = r'D:\Airplane Keypart\Dataset\FRVC\data\gaussan_ske'
        ske_color = (255,255,255)
        ske_joints = joints
        GaussSke = np.zeros((tosize, tosize), np.float)
        ske = np.zeros((tosize, tosize), np.uint8)
        body_joints = np.asarray([ske_joints[2], ske_joints[3], ske_joints[5], ske_joints[6]])
        center = np.average(body_joints, axis=0)
        center = tuple(center.astype(np.int))
        if mode == 'total':
            ske = cv2.line(ske, center, tuple(ske_joints[0]), color=ske_color, thickness=1)
            ske = cv2.line(ske, center, tuple(ske_joints[1]), color=ske_color, thickness=1)
            ske = cv2.line(ske, center, tuple(ske_joints[4]), color=ske_color, thickness=1)
            ske = cv2.line(ske, center, tuple(ske_joints[7]), color=ske_color, thickness=1)
        elif mode == 'body':
            ske = cv2.line(ske, center, tuple(ske_joints[0]), color=ske_color, thickness=1)
            ske = cv2.line(ske, center, tuple(ske_joints[7]), color=ske_color, thickness=1)
        else:
            ske = cv2.line(ske, center, tuple(ske_joints[1]), color=ske_color, thickness=1)
            ske = cv2.line(ske, center, tuple(ske_joints[4]), color=ske_color, thickness=1)
        for i in range(ske.shape[0]):
            for j in range(ske.shape[1]):
                if ske[i,j] == 255:
                    x = j
                    y = i
                    tmp = self._makeGaussian(tosize,tosize,sigma=6,center=(x,y))
                    GaussSke += tmp
        # normalize
        GaussSke = (GaussSke - GaussSke.min())/(GaussSke.max()-GaussSke.min())
        GaussSke = GaussSke * 255
        GaussSke = GaussSke.astype(np.uint8)
        cv2.imwrite(os.path.join(SavePath,name),GaussSke)
        # cv2.imshow('gauss',GaussSke)
        # cv2.waitKey()



    def ifske(self,coord,joints):
        i,j = coord
        center = joints[4,:]
        target = (joints[:,0]- center[0]) * j - (joints[:,1] - center[1]) * i + joints[:,0] * joints[:,1] - center[0] * center[1]
        target = target[0:4]
        part = np.where(abs(target) < 0.1)
        part = part[0]
        if len(part) != 0:
            return True
        else:
            return False







    def visual_ske(self,name=None, name_visual=False ,tosize=700,ifdebug=False):
        '''
        if name is None, run all the pic in pic_list to generate and visualize skeleton
        otherwise visualize the given name pic
        given name mode is access for inner use to generate and visualize
        :param name:
        :param tosize:
        :return:
        '''
        method = 2
        ske_color = (50, 30, 33)
        HEAD_COLOR = (255,0,0)
        L_W_COLOR = (0,255,0)
        R_W_COLOR = (0,0,255)
        TAIL_COLOR = (255,255,0)
        ske_joints = self.str2num_list(self.point_dict[name])
        ske_joints = np.reshape(ske_joints, (-1, 2))
        ske_attr = self.attr_dict[name]
        if_ske = True
        for attr in ske_attr:
            if attr == '2':
                if_ske = False
        # print(ske_attr)
        img = cv2.imread(os.path.join(img_dir, name))
        ske = cv2.resize(img, (tosize,tosize))
        visiweights = self.str2num_list(self.attr_dict[name])
        ske_joints = self.Raw2Resize_joints(ske_joints, img.shape, tosize)
        if method == 1:
            center = self.get_plane_center(ske_joints)
        elif method == 2:
            body_joints = np.asarray([ske_joints[2], ske_joints[3], ske_joints[5], ske_joints[6]])
            center = np.average(body_joints, axis=0)
            center = tuple(center.astype(np.int))
        else:
            print('no method')
            return -1
        if ifdebug == True:
            points = ske.copy()
            p =[]
            p.append(ske_joints[2])
            p.append(ske_joints[3])
            p.append(ske_joints[5])
            p.append(ske_joints[6])
            for i in range(4):
                points = cv2.circle(points, tuple(p[i]), radius=2, color=self.bgr_color[i], thickness=5)
            points = cv2.line(points, tuple(p[0]), tuple(p[2]), color=self.bgr_color[4], thickness=2)
            points = cv2.circle(points, center, radius=2, color=self.bgr_color[5], thickness=2)
            cv2.imshow('points',points)
            cv2.waitKey()
        ske = cv2.circle(ske, center, radius=2, color=(255, 255, 255), thickness=2)
        ske = cv2.line(ske, center, tuple(ske_joints[0]), color=HEAD_COLOR, thickness=2)
        ske = cv2.line(ske, center, tuple(ske_joints[1]), color=L_W_COLOR, thickness=2)
        ske = cv2.line(ske, center, tuple(ske_joints[4]), color=R_W_COLOR, thickness=2)
        ske = cv2.line(ske, center, tuple(ske_joints[7]), color=TAIL_COLOR, thickness=2)
        if name_visual ==True:
            cv2.imshow('img',ske)
            cv2.waitKey()
        return ske


    def Raw2Resize_joints(self, joints, rawsize, tosize):
        '''
        Return new coordinate in new size
        :param joints:int coordinate
        :param tosize:
        :return:
        '''
        height = rawsize[0]
        width = rawsize[1]
        new_j = np.zeros_like(joints)
        for index in range(joints.shape[0]):
            new_j[index][0] = int(joints[index][0] * tosize / width)
            new_j[index][1] = int(joints[index][1] * tosize / height)
        return new_j


    def get_plane_center(self, joints):
        ske_joints = joints
        if len(joints) != 12:
            print('Wrong ske points')
            return False
        p1 = ske_joints[2]
        p2 = ske_joints[3]
        p3 = ske_joints[5]
        p4 = ske_joints[6]
        """
        A1*x + B1*y = C1
        A2*x + B2*y = C2
        """
        A1 = p3[1]-p1[1]
        B1 = p1[0]-p3[0]
        C1 = p3[1]*p1[0]-p1[1]*p3[0]
        A2 = p4[1] - p2[1]
        B2 = p2[0] - p4[0]
        C2 = p4[1] * p2[0] - p2[1] * p4[0]
        A = np.array([[A1,B1],[A2,B2]])
        B = np.array([C1,C2])
        try:
            c_x, c_y = np.linalg.solve(A,B)
        except:
            c_x, c_y = np.array([0,0])
        return (int(c_x), int(c_y))

    # def gen_hed_data(self,pic_path, ske_path):
    #     '''
    #     save pic for training hed alone
    #     :param pic_path:
    #     :param ske_path:
    #     :return:
    #     '''
    #     for name in self.pic_list:
    #         img = cv2.imread(os.path.join(self.img_dir,name))
    #         ske = self.generate_ske(name)
    #         img_resize = cv2.resize(img, (256,256))
    #         cv2.imwrite(os.path.join(pic_path,name), img_resize)
    #         cv2.imwrite(os.path.join(ske_path,name), ske)


    # old method(no use)
    # def generate_ske_new(self,name,tosize=256):
    #     '''
    #     NOT USE, WRONG!!!
    #     get skeleton using new way
    #     :param name:
    #     :param tosize:
    #     :return:
    #     '''
    #     ske_color = (255, 255, 255)
    #     ske_joints = self.str2num_list(self.point_dict[name])
    #     ske_joints = np.reshape(ske_joints, (-1, 2))
    #     ske_attr = self.attr_dict[name]
    #     for attr in ske_attr:
    #         if attr == '2':
    #             return -1
    #     img = cv2.imread(os.path.join(self.img_dir, name))
    #     height, width, c = img.shape
    #     ske = img.copy()
    #     line1 = self.get_line(ske_joints[0], ske_joints[7])
    #     # use visible side as chosen side
    #     if (ske_attr[3] == '0') & (ske_attr[4] == '0') & (ske_attr[5] == '0'):
    #         chosen_side = 'right'
    #         wing_center = (ske_joints[3] + ske_joints[5]) / 2
    #         line2 = self.get_line(wing_center,ske_joints[1])
    #     else:
    #         chosen_side = 'left'
    #         wing_center = (ske_joints[2] + ske_joints[6]) / 2
    #         line2 = self.get_line(wing_center, ske_joints[4])
    #     center = self.get_line_cross(line1,line2)
    #     ske = np.zeros((height, width), np.uint8)
    #     ske = cv2.line(ske, center, tuple(ske_joints[0]), color=ske_color, thickness=1)
    #     ske = cv2.line(ske, center, tuple(ske_joints[1]), color=ske_color, thickness=1)
    #     ske = cv2.line(ske, center, tuple(ske_joints[4]), color=ske_color, thickness=1)
    #     ske = cv2.line(ske, center, tuple(ske_joints[7]), color=ske_color, thickness=1)
    #     return ske
    #
    # # y = kx +b     line: [k,b]
    # def get_line(self,pt1, pt2):
    #     x1 = pt1[0]
    #     y1 = pt1[1]
    #     x2 = pt2[0]
    #     y2 = pt2[1]
    #     k = (y2-y1)/(x2-x1)
    #     b = (x2*y1-x1*y2) / (x2-x1)
    #     return [k,b]
    #
    # def get_line_cross(self,line1, line2):
    #     k1 = -line1[0]
    #     b1 = line1[1]
    #     k2 = -line2[0]
    #     b2 = line2[1]
    #     A = np.array([[k1, 1], [k2, 1]])
    #     B = np.array([b1, b2])
    #     try:
    #         c_x, c_y = np.linalg.solve(A,B)
    #     except:
    #         c_x, c_y = np.array([0,0])
    #     return (int(c_x), int(c_y))


    def plusjpg(self,dir,savename):
        list = []
        with open(dir,'r') as f:
            for line in f:
                cur = line.strip()
                cur = cur+'.jpg'
                list.append(cur)
        path,_ = os.path.split(dir)
        name = os.path.join(path,savename)
        print(name)
        with open(name, 'w') as s:
            for item in list:
                s.write(item+'\n')

    def show_bbox(self,name):
        bbox = self.bbox_dict[name]
        bbox = self.str2num_list(bbox)
        img = cv2.imread(os.path.join(img_dir,name))
        img_copy = img.copy()
        cv2.rectangle(img_copy, pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]),color=(0,0,255),thickness=5)
        cv2.imshow('img',img_copy)
        cv2.waitKey()


    def gen_eval(self,name):
        '''
        return visual joints index and count
        :param name:
        :return:
        '''
        attr_j = self.attr_dict[name]
        visi_count = 0
        visi_joint = []
        for i in range(len(self.joints_list)):
            if attr_j[i] == '0':
                visi_joint.append(i)
                visi_count += 1
        return visi_joint, visi_count

    def read_manufac(self,manu_txt):
        manufac_dict = {}
        with open(manu_txt, 'r') as f:
            for line in f:
                line = line.strip()
                cur = line.split(' ')
                name = cur[0] + '.jpg'
                manu = cur[1]
                manufac_dict[name] = manu
        print(len(manufac_dict.keys()))
        self.manu_dict = manufac_dict

    def Gen_Skeval_GT(self,params,GTPath,test_num):
        generator = self.generate(params=params,mode='test')
        for i in range(test_num):
            img, gtmap, visi_weights, ske, name = next(generator)
            skeleton = np.squeeze(ske,axis=0)
            skeleton = skeleton * 255
            skeleton = skeleton.astype(np.uint8)
            cv2.imwrite(os.path.join(GTPath,name),skeleton)

    def read_img(self,name):
        img = cv2.imread(os.path.join(img_dir,name))
        return img



if __name__ == '__main__':
    img_dir = r'D:\Airplane Keypart\Dataset\FRVC\data\images'
    # csv_path = r'D:\Airplane Keypart\hourglasstensorlfow\hourglass-branch/via_export_csv.csv'
    # bbox_path = r'D:\Airplane Keypart\hourglasstensorlfow\data\Dataset\FRVC\data\images_box.txt'
    # npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_train.npy'
    npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_test_new.npy'
    pic_list = r'D:\Airplane Keypart\Dataset\FRVC\data\ske_pic_list.txt'
    params = config.parser_config('config.cfg')
    data = Data(img_dir=img_dir, npy_path=npy_path)
    data.readnpy()


    # data.readcsv()

    ## -------write in txt---------#
    # data.totxt(save_path)

    # ------validation heatmap generating--------#
    # save_path = {'heat':r'D:\Airplane Keypart\hourglasstensorlfow\data\Dataset\FRVC\visual\heat',
    #              'visu':r'D:\Airplane Keypart\hourglasstensorlfow\data\Dataset\FRVC\visual\visible_only'}
    # data.test_hm(mode='visual')

    # ---------pic data-----------#
    # data.plusjpg('D:\Airplane Keypart\Dataset\FRVC\data\images_train.txt',savename='train')
    # data.data_wash()

    # -----genrate gaussan skeleton--------------
    # index = 0
    # for img, heat, weights,ske, name in data.generate(params,'test'):
    #     print(index)
    #     index += 1

    # ------generate a batchsize of pic----------#
    # data.create_sets()
    generator = data.generate(params,mode='test')
    # # for img, heat, weights,ske in data.generate(params):
    while True:
        begin = time.time()
        img, heat, weights, ske, name = next(generator)
        end = time.time()
        TimeUsed = int(abs(begin-end))
        print('time:',TimeUsed)
        # img = np.squeeze(img)
        cv2.imwrite(os.path.join(r'D:\Airplane Keypart\Dataset\FRVC\data\resize',name),img)
        # print(heat.shape)
        # heat_ex = heat[0,0,:,:,:]
        # heat_ex = np.sum(heat_ex,axis=2)
        # ske_ex = ske[0]
        # print(ske_ex.min(),ske_ex.max())
        # cv2.imshow('ske',ske_ex)
        # cv2.imshow('img',img_ex)
        # print(img_ex.dtype)
        # print(ske_ex.dtype)
        # cv2.imshow('heat',heat_ex)
        # cv2.waitKey()


    # ------visual skeleton---------------
    # '''
    # 0062131: true index, wrong loc
    # 0062857: wrong index
    # 0063926: four point nearly distributing in one line
    # '''
    # save_path = r'D:\Airplane Keypart\Dataset\FRVC\visual\ske'
    # for name in data.pic_list:
    #     ske = data.visual_ske(name=name,name_visual=False,tosize=700)
    #     cv2.imwrite(os.path.join(save_path,name),ske)

    # -------test function generate_ske--------------
    # name = '0056589.jpg'
    # data.visual_ske()
    # for name in data.pic_list:
    #     joints  = data.point_dict[name]
    #     joints = data.str2num_list(joints)
    #     joints = np.reshape(joints,(-1,2))
    #     ske = data.generate_ske(name,joints,tosize=700,method='method2')
    #     cv2.imshow('ske',ske)
    #     cv2.waitKey()

    # ------------gen_hed_data------------
    # pic_path = r'D:\Airplane Keypart\Dataset\FRVC\data\HED\img_raw'
    # ske_path = r'D:\Airplane Keypart\Dataset\FRVC\data\HED\img_gt'
    # data.gen_hed_data(pic_path,ske_path)

    # -----------test resize---------------
    # data.relative_j(name='', tosize=800)

    # ---------manu---------------------
    # manu_path = r'D:\Airplane Keypart\Dataset\FRVC\data\images_manufacturer_train.txt'
    # out = r'D:\Airplane Keypart\Dataset\FRVC\data\HED\img_manu'
    # data.read_manufac(manu_path)
    # print(len(data.pic_list))
    # wrong_pic = []
    # for name in data.pic_list:
    #     # if data.manu_dict[name] == 'Boeing':
    #     manu = data.manu_dict[name]
    #     print(data.manu_dict[name])
    #     try:
    #         ske = data.visual_ske(name)
    #         cv2.imwrite(os.path.join(out,'%s_%s'%(manu,name)),ske)
    #     except:
    #         wrong_pic.append(name)
    # print(wrong_pic)
        # cv2.imshow('ske',ske)
        # cv2.waitKey()


    # # # --------------test Raw2PadCrop_joints------------
    # for name in data.pic_list:
    # # name = '0103328.jpg'
    #     bbox = data.str2num_list(data.bbox_dict[name])
    #     data.Raw2PadCrop_joints(name,bbox,debug=True)
