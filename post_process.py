import numpy as np
import cv2
import config
import data_process
import os
import fnmatch

class PostProcessor():

    def __init__(self, out_dir, joints_dir ,dataset,ske_dir=None):
        '''
        heatmap Dim 64 x 64 x12
        joints Dim 12 x 2 in 256 x 256 resized img
        :param out_dir:
        :param joints_dir:
        '''
        self.data = dataset
        self.heatmap = np.load(out_dir,allow_pickle=True).item()
        self.joints = np.load(joints_dir,allow_pickle=True).item()
        self.pic_list = list(self.joints.keys())
        # self.pic_list = self.data.pic_list
        if ske_dir is not None:
            self.ske = np.load(ske_dir,allow_pickle=True).item()
        print('load %d results' % len(self.pic_list) )
        self.img_dir = img_dir
        self.joints_list = ['head', 'left_wing_tip', 'left_wing_front', 'right_wing_front', 'right_wing_tip',
                            'right_wing_back',
                            'left_wing_back', 'tail_tip', 'tail_left', 'tail_up', 'tail_right', 'tail_front']


    def concate_heat(self):
        '''
        :return:
        '''
        for name in self.pic_list:
            # Dim 1 x 64 x 64 x 12
            heats = self.heatmap[name]
            heats = np.squeeze(heats)
            sum = heats[:,:,0]
            # sum = np.sum(heats, axis=2)
            heat = cv2.resize(sum,(256,256))
            img = cv2.imread(os.path.join(self.img_dir, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(256,256))
            cv2.imshow('img',img)
            cv2.imshow('heatmap',heat)
            cv2.waitKey()

    def get_joints(self):
        '''
        using heatmap to get peak value loc to get prediction
        visualize prediction, gt and heatmap
        :return:
        '''
        for name in self.pic_list:
            # Dim 1 x 64 x 64 x 12
            heats = self.heatmap[name]
            # Dim 64 x 64 x 12
            heats = np.squeeze(heats)
            # Dim 64 x 64
            small_heat = np.sum(heats, axis=2)
            joints = []
            img = cv2.imread(os.path.join(self.img_dir,name))
            height, width, c = img.shape
            print(height,width)
            draw = img.copy()
            draw = cv2.resize(draw,(256,256))
            gt = draw.copy()
            gtj = self.data.relative_j(name,tosize=256)
            for i in range(heats.shape[2]):
                map = heats[:,:,i]
                # pos: (row, col)
                pos = np.where(map == np.max(map))
                print('pos:',pos)
                joints.append(pos[0])
                joints.append(pos[1])
                x1 = int(pos[1]* 256 /64)
                y1 = int(pos[0]* 256/64)
                print('new_j:',(x1,y1))
                # usual coordinate
                cv2.circle(draw, center=(x1,y1), radius=2, thickness=2, color=(0, 0, 255))
            for i in range(gtj.shape[0]):
                cv2.circle(gt, center=tuple(gtj[i]), radius=2, thickness=2, color=(0, 0, 255))
            # small_heat = cv2.normalize(small_heat,small_heat)
            cv2.imshow('small',small_heat)
            cv2.imshow('gt', gt)
            cv2.imshow('a',draw)
            cv2.waitKey()



    def draw_points(self,tosize=700):
        '''
        draw self.joint in pic in size (256,256)
        :return:
        '''
        for name in self.pic_list:
            img = cv2.imread(os.path.join(self.img_dir, name))
            attr = self.data.attr_dict[name]
            bbox = self.data.bbox_dict[name]
            bbox = data.str2num_list(bbox)
            img = data.crop_new(img, bbox)
            draw = img.copy()
            draw = cv2.resize(draw, (tosize, tosize))
            heat_joints = self.joints[name]
            raw_j = np.zeros_like(heat_joints)
            # new_j
            for index in range(heat_joints.shape[0]):
                raw_j[index][0] = int(heat_joints[index][1] * tosize / 64)
                raw_j[index][1] = int(heat_joints[index][0] * tosize / 64)
            for i in range(raw_j.shape[0]):
                if attr[i] == '0':
                    cv2.putText(draw,'%s'%(i+1),tuple(raw_j[i]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(183,211,0),thickness=int(tosize /250))
                    cv2.circle(draw, center=tuple(raw_j[i]), radius=int(tosize / 256), thickness=int(tosize / 128),
                               color=(0, 0, 255))
                elif attr[i] == '1':
                    cv2.putText(draw, '%s' % (i + 1), tuple(raw_j[i]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(183, 211, 0), thickness=int(tosize / 250))
                    cv2.circle(draw, center=tuple(raw_j[i]), radius=int(tosize / 256), thickness=int(tosize / 128),
                               color=(255, 0, 0))
                else:
                    cv2.putText(draw, '%s' % (i + 1), tuple(raw_j[i]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(183, 211, 0), thickness=int(tosize / 250))
                    cv2.circle(draw, center=tuple(raw_j[i]), radius=int(tosize / 256), thickness=int(tosize / 128),
                               color=(0, 255, 0))
            cv2.imshow('draw',draw)
            cv2.waitKey()

    def save_draw_results(self,save_path,tosize=256):
        '''
        save pic results
        :return:
        '''
        for name in self.pic_list:
            img = cv2.imread(os.path.join(self.img_dir, name))
            attr = self.data.attr_dict[name]
            bbox = self.data.bbox_dict[name]
            bbox =data.str2num_list(bbox)
            # img = data.crop_new(img,bbox)
            img = data.padcrop_new(img,bbox)
            draw = img.copy()
            draw = cv2.resize(draw,(tosize,tosize))
            heat_joints = self.joints[name]
            raw_j = np.zeros_like(heat_joints)
            # new_j
            for index in range(heat_joints.shape[0]):
                raw_j[index][0] = int(heat_joints[index][1] * tosize / 64)
                raw_j[index][1] = int(heat_joints[index][0] * tosize / 64)
            for i in range(raw_j.shape[0]):
                if attr[i] == '0':
                    cv2.putText(draw, '%s' % (i + 1), tuple(raw_j[i]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(183, 211, 0), thickness=int(tosize / 250))
                    cv2.circle(draw, center=tuple(raw_j[i]), radius=int(tosize / 256), thickness=int(tosize / 128),
                               color=(0, 0, 255))
                    # ----------------------draw invisible results---------------------
                # elif attr[i] == '1':
                    # cv2.putText(draw, '%s' % (i + 1), tuple(raw_j[i]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    #             color=(183, 211, 0), thickness=int(tosize / 250))
                    # cv2.circle(draw, center=tuple(raw_j[i]), radius=int(tosize / 256), thickness=int(tosize / 128),
                    #            color=(255, 0, 0))
                # else:
                #     cv2.putText(draw, '%s' % (i + 1), tuple(raw_j[i]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                #                 color=(183, 211, 0), thickness=int(tosize / 250))
                #     cv2.circle(draw, center=tuple(raw_j[i]), radius=int(tosize / 256), thickness=int(tosize / 128),
                #                color=(0, 255, 0))
            savename = os.path.join(save_path, name)
            cv2.imwrite(savename,draw)

    def SeeFailture(self,SavePath,tosize=256,index = 1):
        '''
        given an index and save failure case of this index point
        :param SavePath:
        :param tosize:
        :return:
        '''
        for name in self.pic_list:
            img = cv2.imread(os.path.join(self.img_dir, name))
            attr = self.data.attr_dict[name]
            bbox = self.data.bbox_dict[name]
            bbox =data.str2num_list(bbox)
            img = data.crop_new(img,bbox)
            draw = img.copy()
            draw = cv2.resize(draw,(tosize,tosize))
            heat_joints = self.joints[name]
            raw_j = np.zeros_like(heat_joints)
            # ---------check pcki--------------------
            attr = self.data.attr_dict[name]
            if attr[index] == '0':
                pcki = self.pcki(name, index)
                if pcki >= 1:
                    # ------------------draw----------------------------
                    for i in range(heat_joints.shape[0]):
                        raw_j[i][0] = int(heat_joints[i][1] * tosize / 64)
                        raw_j[i][1] = int(heat_joints[i][0] * tosize / 64)
                    for i in range(raw_j.shape[0]):
                        if attr[i] == '0':
                            cv2.putText(draw, '%s' % (i + 1), tuple(raw_j[i]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                        color=(183, 211, 0), thickness=int(tosize / 250))
                            cv2.circle(draw, center=tuple(raw_j[i]), radius=int(tosize / 256), thickness=int(tosize / 128),
                                       color=(0, 0, 255))
                    SaveName = os.path.join(SavePath,name)
                    cv2.imwrite(SaveName,draw)



    def pcki(self,name, index):
        predj = self.joints[name]
        srl = np.zeros_like(predj)
        srl[:, 0] = predj[:, 1]
        srl[:, 1] = predj[:, 0]
        bbox = self.data.bbox_dict[name]
        bbox = self.data.str2num_list(bbox)
        gtj = self.data.Raw2CropResize_joints(name=name,bbox=bbox,tosize=64)
        pcki = np.linalg.norm(gtj[index]-srl[index]) / 6.4
        return pcki

    def padding_packi(self,name,index):
        predj = self.joints[name]
        srl = np.zeros_like(predj)
        # 12 x 2
        srl[:, 0] = predj[:, 1]
        srl[:, 1] = predj[:, 0]
        bbox = self.data.bbox_dict[name]
        bbox = self.data.str2num_list(bbox)
        gtj = self.data.Raw2PadCrop_joints(name=name, inbbox=bbox, tosize=64)
        raw_height = bbox[3] - bbox[1]
        raw_width = bbox[2] - bbox[0]
        agu_box = max(raw_height, raw_width)
        ratio = np.asarray([raw_height / agu_box, raw_width / agu_box])
        length = ratio * 6.4
        # srl_arround: 12 x 4
        srl_arround = np.asarray([srl[:,0]-length[1], srl[:,0]+length[1], srl[:,1]-length[0], srl[:,1]+length[0]])
        srl_arround = np.swapaxes(srl_arround,0,1)
        if (gtj[index,0] < srl_arround[index,1]) and (gtj[index,0] > srl_arround[index,0])\
            and (gtj[index,1] < srl_arround[index,3]) and (gtj[index,1] > srl_arround[index,2]):
            pcki = 0.1
        else:
            pcki = 2
        return pcki




    def pck(self):
        print('Testing %d' % len(self.pic_list))
        joints_count = 0
        tp = 0
        total = len(self.pic_list)
        cur = 0
        for name in self.pic_list:
            block = int(cur/total*20)
            print('\rProcessing:|{0}{1}|'.format('='*block,' '*(20-block)) + '%0.3s%%' % (cur/total*100),end='')
            visi_joints, visi_count = self.data.gen_eval(name)
            joints_count += visi_count
            for index in visi_joints:
                pcki = self.pcki(name,index)
                # pcki = self.padding_packi(name,index)
                if pcki < 1:
                    tp += 1
            cur += 1
        print('joints_count:',joints_count)
        print('tp:',tp)
        return tp/joints_count


    def each_pck(self):
        print('Testing %d' % len(self.pic_list))
        joints_count = np.zeros((12,1),dtype=np.float32)
        tp = np.zeros((12,1),dtype=np.float32)
        total = len(self.pic_list)
        cur = 0
        for name in self.pic_list:
            block = int(cur / total * 20)
            attr = self.data.attr_dict[name]
            for index in range(12):
                if attr[index] == '0':
                    joints_count[index] += 1
                    # pcki = self.pcki(name, index)
                    pcki = self.padding_packi(name, index)
                    if pcki < 1:
                        tp[index] += 1
            tmp_joints = np.sum(joints_count)
            tmp_tp = np.sum(tp)
            tmp_pck = tmp_tp/tmp_joints
            cur += 1
            print('\rProcessing:|{0}{1}|'.format('=' * block, ' ' * (20 - block)) + '%0.3s%%' % (cur / total * 100)
                  + '    pck: %0.3f' % (tmp_pck),
                  end='')
        print('\n------Result------')
        for index in range(12):
            print('%s:%f'%(self.joints_list[index],tp[index]/joints_count[index]))
        total_joints = np.sum(joints_count,axis=0)
        total_tp = np.sum(tp,axis=0)
        # print(joints_count)
        print('Total:%f' % (total_tp/total_joints))


    def visual_ske(self,InPath,ifSave,OutPath):
        img_list = os.listdir(InPath)
        img_list = fnmatch.filter(img_list, '*.jpg')
        for name in img_list:
            skeleton = cv2.imread(os.path.join(InPath,name))
            if skeleton.shape[2] == 1:
                ske3 = np.repeat(skeleton, repeats=3, axis=2)
            else:
                ske3 = skeleton
            img = cv2.imread(os.path.join(self.img_dir, name))
            bbox = self.data.bbox_dict[name]
            bbox = data.str2num_list(bbox)
            img = data.crop_new(img, bbox)
            draw = img.copy()
            draw = cv2.resize(draw, (256, 256))
            visu = 0.3 * draw + 0.7 * ske3
            visu = visu.astype(np.uint8)
            visu = cv2.resize(visu,(700,700))
            if ifSave:
                cv2.imwrite(os.path.join(OutPath,name),visu)
            else:
                cv2.imshow('visu', visu)
                cv2.waitKey()

    def read_ske(self,ifsave=False, SavePath=None,VisuPath=None):
        for name in self.pic_list:
            skeleton = self.ske[name]
            mean = np.mean(skeleton)
            skeleton = skeleton[5]
            skeleton = np.squeeze(skeleton, axis=0)
            ske_thresh = cv2.threshold(skeleton,0,maxval=256,type=cv2.THRESH_TOZERO)
            ske_thresh = ske_thresh[1]
            ske_norm = (ske_thresh - ske_thresh.min())/(ske_thresh.max()-ske_thresh.min())
            ske = ske_norm * 255    #Dim: 256 x 256
            ske = ske.astype(np.uint8)
            ske = np.expand_dims(ske,axis=2)    #Dim: 256 x 256 x 1
            ske3 = np.repeat(ske,repeats=3,axis=2)  #Dim: 256 x 256 x 3
            if ifsave == False:
                cv2.imshow('ske',ske)
                cv2.imshow('ske3',ske3[:,:,1])
            else:
                cv2.imwrite(os.path.join(SavePath,name),ske)
            img = cv2.imread(os.path.join(self.img_dir, name))
            bbox = self.data.bbox_dict[name]
            bbox = data.str2num_list(bbox)
            img = data.crop_new(img, bbox)
            draw = img.copy()
            draw = cv2.resize(draw, (256, 256))
            visu = 0.3 * draw + 0.7 * ske3
            visu = visu.astype(np.uint8)
            if ifsave == False:
                cv2.imshow('visu',visu)
                cv2.waitKey()
            else:
                cv2.imwrite(os.path.join(VisuPath,name),visu)
            # print(ske.mean())
            # print('1')
            # cv2.imshow('ske',ske)

    def Read4Skes(self,SkePath,ifsave,SavePath,VisuPath):
        for name in self.pic_list:
            total = cv2.imread(os.path.join(SkePath,'c0_%s'%name))
            body = cv2.imread(os.path.join(SkePath,'c1_%s'%name))
            left = cv2.imread(os.path.join(SkePath,'c2_%s'%name))
            right = cv2.imread(os.path.join(SkePath,'c3_%s'%name))
            concate = np.concatenate((total, body, left, right), axis=1)

            img = cv2.imread(os.path.join(self.img_dir, name))
            bbox = self.data.bbox_dict[name]
            bbox = data.str2num_list(bbox)
            img = data.padcrop_new(img,bbox)
            draw = img.copy()
            draw = cv2.resize(draw, (256, 256))
            draw = np.concatenate((draw,draw,draw,draw),axis=1)

            visu = 0.3 * draw + 0.7 * concate
            visu = visu.astype(np.uint8)
            if ifsave == False:
                cv2.imshow('visu', visu)
                cv2.imshow('con',concate)
                cv2.waitKey()


        # self.pic_list = list(self.ske.keys())
        # for name in self.pic_list:
        #     ske4 = self.ske[name]
        #     ske4 = np.squeeze(ske4)
        #     total = ske4[:,:,0]
        #     total = self.Normalize255(total)
        #     body = ske4[:,:,1]
        #     body = self.Normalize255(body)
        #     left = ske4[:,:,2]
        #     left = self.Normalize255(left)
        #     right = ske4[:,:,3]
        #     right = self.Normalize255(right)
        #     concate = np.concatenate((total,body,left,right), axis=1)
        #     concate = np.repeat(np.expand_dims(concate,axis=2),repeats=3,axis=2)
        #
        #     img = cv2.imread(os.path.join(self.img_dir, name))
        #     bbox = self.data.bbox_dict[name]
        #     bbox = data.str2num_list(bbox)
        #     img = data.crop_new(img, bbox)
        #     draw = img.copy()
        #     draw = cv2.resize(draw, (256, 256))
        #     draw = np.concatenate((draw,draw,draw,draw),axis=1)
        #     visu = 0.3 * draw + 0.7 * concate
        #     visu = visu.astype(np.uint8)
        #     if ifsave == False:
        #         cv2.imshow('visu', visu)
        #         cv2.imshow('con',concate)
        #         cv2.waitKey()
        #     # print(ske4)


    def Normalize255(self,input):
        output = cv2.threshold(input, 0, maxval=256, type=cv2.THRESH_TOZERO)
        output = output[1]
        max = output.max()
        min = output.min()
        output = (output - min)/(max - min)
        output = output * 255
        return output.astype(np.uint8)

    def Normalize1(self,input):
        max = input.max()
        min = input.min()
        output = (input - min)/(max - min)
        return output


    def ResultCompare(self,LeftPath,RightPath,OutPath):
        imglist = os.listdir(LeftPath)
        imglist = fnmatch.filter(imglist, '*.jpg')
        for name in imglist:
            left = cv2.imread(os.path.join(LeftPath,name))
            right = cv2.imread(os.path.join(RightPath,name))
            concate = np.concatenate((left,right),axis=1)
            cv2.imwrite(os.path.join(OutPath,name),concate)
            # print(concate.shape)


    def ThreeCompare(self,LeftPath,MiddlePath,RightPath,OutPath):
        imglist = os.listdir(LeftPath)
        imglist = fnmatch.filter(imglist, '*.jpg')
        for name in imglist:
            left = cv2.imread(os.path.join(LeftPath, name))
            middle = cv2.imread(os.path.join(MiddlePath, name))
            right = cv2.imread(os.path.join(RightPath, name))
            concate = np.concatenate((left,middle, right), axis=1)
            cv2.imwrite(os.path.join(OutPath, name), concate)

    # def ImgPlusSke(self,ske_path,save_path):
    #     imglist = os.listdir(ske_path)
    #     imglist = fnmatch.filter(imglist, '*.jpg')
    #     for name in imglist:





if __name__ == '__main__':
    key_path = r'D:\Airplane Keypart\skey_data\SKEY\output_0102'
    out_dir = os.path.join(key_path,'heatmaps.npy')
    joint_dir = os.path.join(key_path,'joints.npy')
    ske_dir = r'D:\Airplane Keypart\skey_data\SKEY\output_0102\ske'
    # ske_dir = r'D:\Airplane Keypart\skey_data\HED\output_1230'
    img_dir = r'D:\Airplane Keypart\Dataset\FRVC\data\images'
    # csv_path = r'D:\Airplane Keypart\hourglasstensorlfow\hourglass-branch/via_export_csv.csv'
    # bbox_path = r'D:\Airplane Keypart\hourglasstensorlfow\data\Dataset\FRVC\data\images_box.txt'
    # npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_train.npy'
    npy_path = r'D:\Airplane Keypart\Dataset\FRVC\data/FGVC_Keypoints_test_new.npy'
    params = config.parser_config('config.cfg')
    data = data_process.Data(img_dir=img_dir, npy_path=npy_path)
    data.readnpy()
    post = PostProcessor(out_dir=out_dir,joints_dir=joint_dir, dataset=data)
    # post.draw_points(tosize=700)

    # --------------compute pck--------------------
    post.each_pck()

    # -------------save keypoints pic----------------
    # savepath = r'D:\Airplane Keypart\skey_data\SKEY\output_0102\key'
    # post.save_draw_results(savepath,tosize=700)

    # -------------save failure case-----------------
    # save_path = r'D:\Airplane Keypart\skey_data\HG\failure_0102'
    # post.SeeFailture(save_path,tosize=700,index=1)


    # inpath = r'D:\Airplane Keypart\skey_data\HED\output_1218\side5'
    # outpath = r'D:\Airplane Keypart\skey_data\HED\output_1218\visu'
    # post.visual_ske(inpath,False,outpath)

    # -----read ske-------
    # post.read_ske()

    # ----save ske pic----
    # save_path = r'D:\Airplane Keypart\skey_data\HED\output_1225\ske'
    # visu_path = r'D:\Airplane Keypart\skey_data\HED\output_1225\visu'
    # post.read_ske(ifsave=True, SavePath=save_path, VisuPath=visu_path)

    # ----read, concate and save 4 channel skeleton----------------
    # save_path = r'D:\Airplane Keypart\skey_data\HED\output_1226\ske'
    # visu_path = r'D:\Airplane Keypart\skey_data\HED\output_1226\visu'
    # post.Read4Skes(SkePath=ske_dir, ifsave=False, SavePath=save_path, VisuPath=visu_path)


    # -----concate two result----------
    # left_path = r'D:\Airplane Keypart\skey_data\SKEY\failure_1219'
    # right_path = r'D:\Airplane Keypart\skey_data\SKEY\output_1220\ske\visu'
    # out_path = r'D:\Airplane Keypart\skey_data\SKEY\failure_1219\plus_ske'
    # post.ResultCompare(left_path,right_path,out_path)

    # -----concate three results --------
    # left_path = r'D:\Airplane Keypart\skey_data\HG\output_1213\key'
    # middle_path = r'D:\Airplane Keypart\skey_data\HG\discuss_1214\pics_1214_visi_only'
    # right_path = r'D:\Airplane Keypart\skey_data\SKEY\output_1220\key'
    # out_path = r'F:\discuss_1220\key_compare'
    # post.ThreeCompare(left_path,middle_path,right_path,out_path)
