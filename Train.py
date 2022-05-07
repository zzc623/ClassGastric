#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 6/1/2021 1:56 PM 
# @Author : Zhicheng Zhang 
# @E-mail : zhicheng0623@gmail.com
# @Site :  
# @File : Train.py 
# @Software: PyCharm

import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils'))

import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import datetime, time
import utils.ckpt as ckpt
import random
import copy
from ops import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
from imgaug import augmenters as iaa

aug90 = iaa.Sometimes(1, iaa.Affine(rotate=(-90, -90),mode="constant",cval=(-1000,-1000)))
aug180 = iaa.Sometimes(1, iaa.Affine(rotate=(-180, -180),mode="constant",cval=(-1000,-1000)))
aug270 = iaa.Sometimes(1, iaa.Affine(rotate=(-270, -270),mode="constant",cval=(-1000,-1000)))
augFlip1 = iaa.Sometimes(1, iaa.Fliplr(1))
augFlip2 = iaa.Sometimes(1, iaa.Flipud(1))
augCon= iaa.Sometimes(1, iaa.ContrastNormalization(0.5, 0.9))
augBlur = iaa.Sometimes(1, iaa.GaussianBlur(sigma=(0.0, 0.8)))
augSharpen = iaa.Sometimes(1, iaa.Sharpen(alpha=0.1, lightness=0.7))

Ix = Iy = 160
def augmentHK(image,k):
    w1 = image.reshape(image.shape[0], image.shape[1])
    # k = random.randrange(1, 7)
    global w2
    if k == 1:
        w2 = iaa.Sometimes(1, iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-180, 180)
        ,mode="constant",cval=(0,0)
    )).augment_image(w1)
    elif k == 2:
        w2 = augFlip1.augment_image(w1)
    elif k == 3:
        w2 = augFlip2.augment_image(w1)
    elif k == 4:
        w2 = augCon.augment_image(w1)
    elif k == 5:
        w2 = augBlur.augment_image(w1)
    elif k == 6:
        w2 = augSharpen.augment_image(w1)
    else:
        w2 = w1
    w3 = w2.reshape( w2.shape[0], w2.shape[1])
    w4 = np.array(w3)
    return w4
aug = 6



class model(object):
    def __init__(self):
        self.w_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
        self.reg =  tf.contrib.layers.l2_regularizer(scale=0.1)
        self.param = {}
        self.param['retrain'] = True

        self.param['model_save_path'] = os.path.join('./Results', os.path.basename(__file__).split('.')[0], 'model')
        self.param['tensorboard_save_logs'] = os.path.join('./logs', os.path.basename(__file__).split('.')[0])

    def _l2normalize(self, v, eps=1e-12):
        with tf.name_scope('l2normalize'):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def global_avg_pooling(self, x):
        gap = tf.reduce_mean(x, axis=[1, 2])
        return gap



    def swish(self, x):
        return x * tf.nn.sigmoid(x)

    def cox_loss(self, score, time_value, event):
        '''
        Args
            score: 		predicted survival time_value, tf tensor of shape (None, 1)
            time_value:		true survival time_value, tf tensor of shape (None, )
            event:		event, tf tensor of shape (None, )
        Return
            loss:		partial likelihood of cox regression
        '''

        ## cox regression computes the risk score, we want the opposite
        score = -score

        ## find index i satisfying event[i]==1
        ix = tf.where(tf.cast(event, tf.bool))  # shape of ix is [None, 1]

        ## sel_mat is a matrix where sel_mat[i,j]==1 where time_value[i]<=time_value[j]
        sel_mat = tf.cast(tf.gather(time_value, ix) <= time_value, tf.float32)

        ## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time_value[i]<=time_value[j] and event[i]==1
        p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
        # p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(tf.transpose(tf.exp(score)), axis=-1))
        loss = -tf.reduce_mean(p_lik)

        return loss

    def hinge_loss(self, score, time_value, event):
        '''
        Args
        score:	 	predicted score, tf tensor of shape (None, 1)
        time_value:		true survival time_value, tf tensor of shape (None, )
        event:		event, tf tensor of shape (None, )
        '''
        ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
        ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1) < time_value,
                                     tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')
        ## if score[i]>score[j], incur hinge loss
        s1 = tf.gather(score, ix[:, 0])
        s2 = tf.gather(score, ix[:, 1])
        loss = tf.reduce_mean(tf.maximum(1 + s1 - s2, 0.0), name='loss')

        return loss

    def log_loss(self, score, time_value, event):
        '''
        Args
        score: 	predicted survival time_value, tf tensor of shape (None, 1)
        time_value:		true survival time_value, tf tensor of shape (None, )
        event:		event, tf tensor of shape (None, )
        '''
        ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
        ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1) < time_value,
                                     tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')
        ## if score[i]>score[j], incur log loss
        s1 = tf.gather(score, ix[:, 0])
        s2 = tf.gather(score, ix[:, 1])
        loss = tf.reduce_mean(tf.log(1 + tf.exp(s1 - s2)), name='loss')
        return loss

    def __concordance_index(self, score, time_value, event):
        '''
        Args
            score: 		predicted score, tf tensor of shape (None, )
            time_value:		true survival time_value, tf tensor of shape (None, )
            event:		event, tf tensor of shape (None, )
        '''

        ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
        ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1) < time_value,
                                     tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

        ## count how many score[i]<score[j]
        s1 = tf.gather(score, ix[:, 0])
        s2 = tf.gather(score, ix[:, 1])
        ci = tf.reduce_mean(tf.cast(s1 < s2, tf.float32), name='c_index')

        return ci
    def ResNet18(self,x,w_init=None, drop= 0,norm=None,is_training=True,name='res50'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            ch = 64
            x = tf.layers.conv2d(x, filters=ch, kernel_size=[7, 7], padding='SAME', strides=[2, 2],kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            if norm != None:
                x = norm(x)
            x = tf.nn.leaky_relu(x)

            conv = tf.layers.max_pooling2d(x, pool_size=[3, 3], strides=[2, 2], padding='SAME')

            for i in range(2):
                x = tf.layers.conv2d(conv, filters=ch, kernel_size=[3, 3], padding='SAME',kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
                x = tf.layers.dropout(x, rate=drop)
                if norm != None:
                    x = norm(x)
                x = tf.nn.leaky_relu(x)

                x = tf.layers.conv2d(x, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
                x = tf.layers.dropout(x, rate=drop)
                conv = tf.nn.leaky_relu(x + conv)

            ch *= 2
            x = tf.layers.conv2d(conv, filters=ch, kernel_size=[3, 3], strides=[2,2],padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            if norm != None:
                x = norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            conv = tf.nn.leaky_relu(x + tf.layers.conv2d(conv, filters=ch, kernel_size=[1, 1], strides=[2,2],padding='SAME',kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training))

            x = tf.layers.conv2d(conv, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            if norm != None:
                x = norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            conv = tf.nn.leaky_relu(x + conv)

            ch *= 2
            x = tf.layers.conv2d(conv, filters=ch, kernel_size=[3, 3], padding='SAME',strides=[2,2], kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            if norm != None:
                x = norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            conv = tf.nn.leaky_relu(x + tf.layers.conv2d(conv, filters=ch, kernel_size=[1, 1], strides=[2,2],padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training))

            x = tf.layers.conv2d(conv, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            if norm != None:
                x = norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            conv = tf.nn.leaky_relu(x + conv)

            ch *= 2
            x = tf.layers.conv2d(conv, filters=ch, kernel_size=[3, 3], strides=[2,2],padding='SAME', kernel_initializer=w_init,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            if norm != None:
                x = norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            conv = tf.nn.leaky_relu(x + tf.layers.conv2d(conv, filters=ch, kernel_size=[1, 1], strides=[2,2],padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training))

            x = tf.layers.conv2d(conv, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            if norm != None:
                x = norm(x)
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, filters=ch, kernel_size=[3, 3], padding='SAME', kernel_initializer=w_init,kernel_regularizer=self.reg,trainable=is_training)
            x = tf.layers.dropout(x, rate=drop)
            conv = tf.nn.leaky_relu(x + conv)

            feature = self.global_avg_pooling(conv)
            return feature

    def model(self, is_training = True,name='model'):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = tf.concat(tf.split(self.x, num_or_size_splits=5, axis=-1), axis=0)
            x = self.ResNet18(x,w_init=self.w_init,norm=None,drop=self.drop,is_training=is_training)
            x = tf.concat(tf.split(tf.expand_dims(x,axis=1), num_or_size_splits=5, axis=0), axis=1)

            x = tf.layers.dense(x, 256, kernel_initializer=self.w_init,activation=tf.nn.leaky_relu,kernel_regularizer=self.reg,trainable=is_training)
            # x = self.swish(x)

            feature = tf.layers.dense(x, 256, kernel_initializer=self.w_init,activation=tf.nn.leaky_relu,kernel_regularizer=self.reg,trainable=is_training)  #80*256
            # feature = self.swish(feature)

            a = tf.layers.dense(feature, 256, kernel_initializer=self.w_init,activation=tf.nn.tanh,kernel_regularizer=self.reg,trainable=is_training)
            # a = self.swish(a)
            b = tf.layers.dense(feature, 256, kernel_initializer=self.w_init,activation=tf.nn.sigmoid,kernel_regularizer=self.reg,trainable=is_training)

            A = tf.multiply(a,b)
            A = tf.layers.dense(A, 2, kernel_initializer=self.w_init,kernel_regularizer=self.reg,trainable=is_training)
            A = tf.nn.softmax(tf.transpose(A,perm=[0,2,1]))
            M = tf.matmul(A,feature)  #2*256


        with tf.variable_scope('ISP', reuse=tf.AUTO_REUSE):
            # logit = tf.layers.dense(M[:,0], 128, kernel_initializer=self.w_init,trainable=is_training,activation=tf.nn.leaky_relu)
            logit = tf.layers.dense(M[:,0], 4, kernel_initializer=self.w_init,kernel_regularizer=self.reg,trainable=is_training)

        with tf.variable_scope('survival', reuse=tf.AUTO_REUSE):
            s = tf.layers.dense(tf.concat([M[:,1], tf.nn.softmax(logit)], axis=-1), 1, kernel_initializer=self.w_init,kernel_regularizer=self.reg,activation=tf.nn.leaky_relu)
            s = tf.layers.dense(s, 1, kernel_initializer=self.w_init,kernel_regularizer=self.reg)
            # s = tf.layers.dense(M[:,0], 1, kernel_initializer=self.w_init)

        return logit, s

    def train(self):
        n_epochs = 20000
        checkpointdir = self.param['model_save_path']
        ####################################################################################
        if not os.path.exists(checkpointdir):
            checkpoints_dir = checkpointdir
            os.makedirs(checkpoints_dir)
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            checkpoints_dir = checkpointdir.format(current_time)
        ####################################################################################
        # Read the filename of Training dataset
        self.param['batchsize'] = 16

        ##################################################################
        train_data = np.load('./Dataset/Resize_data/Training.npy',allow_pickle=True)

        train_img = np.empty([np.uint16(len(train_data)), 160, 160, 5], dtype=np.float32)
        train_ISP = np.empty([np.uint16(len(train_data)), 4], dtype=np.float32)
        train_DFS = np.empty([np.uint16(len(train_data))], dtype=np.float32)
        train_st  = np.empty([np.uint16(len(train_data))], dtype=np.float32)

        for i in range(np.uint16(len(train_data))):
            if i == 93 or i == 262 or i == 180:
                continue
            else:
                train_img[i, :, :, :] = train_data[i]['img']
                if train_data[i]['ISP'][0] == 1:
                    train_ISP[i, :] = [1, 0, 0, 0]
                elif train_data[i]['ISP'][0] == 2:
                    train_ISP[i, :] = [0, 1, 0, 0]
                elif train_data[i]['ISP'][0] == 3:
                    train_ISP[i, :] = [0, 0, 1, 0]
                elif train_data[i]['ISP'][0] == 4:
                    train_ISP[i, :] = [0, 0, 0, 1]

                # plt.imshow(train_img[i, :, :, 2])
                # plt.title(str(i))
                # plt.pause(.1)

                train_DFS[i] = train_data[i]['DFS'][0]
                train_st[i] = train_data[i]['st'][0]

        aug_bool = 1
        if aug_bool:
            train_img_aug = np.empty([np.shape(train_img)[0] * (1 + aug), Ix, Iy, 5], dtype=np.float32)
            train_ISP_aug = np.empty([np.shape(train_img)[0] * (1 + aug), 4], dtype=np.float32)
            train_DFS_aug = np.empty([np.shape(train_img)[0] * (1 + aug)], dtype=np.float32)
            train_st_aug  = np.empty([np.shape(train_img)[0] * (1 + aug)], dtype=np.float32)

            iter = 0
            for i in range(np.shape(train_img)[0]):
                train_img_aug[iter, :, :, :] = train_img[i, :, :, :]
                train_ISP_aug[iter, :] = train_ISP[i, :]
                train_DFS_aug[iter] = train_DFS[i]
                train_st_aug[iter] = train_st[i]
                iter += 1
                for ii in range(aug):
                    k = random.randrange(1, 7)
                    for jj in range(5):
                        random_augmented_image = np.array(augmentHK(train_img[i, :, :, jj], k), dtype=np.float)
                        train_img_aug[iter, :, :, jj] = random_augmented_image
                    train_ISP_aug[iter, :] = train_ISP[i, :]
                    train_DFS_aug[iter] = train_DFS[i]
                    train_st_aug[iter] = train_st[i]
                    iter += 1
        else:
            train_img_aug = train_img
            train_ISP_aug = train_ISP
            train_DFS_aug = train_DFS
            train_st_aug = train_st

        ##################################################################
        valid_data = np.load('./Dataset/Resize_data/Validation.npy',allow_pickle=True)
        valid_img = np.empty([np.uint16(len(valid_data)), 160, 160, 5], dtype=np.float32)
        valid_ISP = np.empty([np.uint16(len(valid_data)), 4], dtype=np.float32)
        valid_DFS = np.empty([np.uint16(len(valid_data))], dtype=np.float32)
        valid_st = np.empty([np.uint16(len(valid_data))], dtype=np.float32)

        for i in range(np.uint16(len(valid_data))):
            if i == 124:
                continue
            else:
                valid_img[i, :, :, :] = valid_data[i]['img']

                if valid_data[i]['ISP'][0] == 1:
                    valid_ISP[i, :] = [1, 0, 0, 0]
                elif valid_data[i]['ISP'][0] == 2:
                    valid_ISP[i, :] = [0, 1, 0, 0]
                elif valid_data[i]['ISP'][0] == 3:
                    valid_ISP[i, :] = [0, 0, 1, 0]
                elif valid_data[i]['ISP'][0] == 4:
                    valid_ISP[i, :] = [0, 0, 0, 1]

                valid_DFS[i] = valid_data[i]['DFS'][0]
                valid_st[i] = valid_data[i]['st'][0]

        #####################################################################################

        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            ####################################################################################
            self.x = tf.placeholder(tf.float32, [self.param['batchsize'], 160, 160, 5])
            self.ISP = tf.placeholder(tf.float32, [self.param['batchsize'], 4])
            self.DFS = tf.placeholder(tf.float32, [self.param['batchsize']])
            self.st = tf.placeholder(tf.float32, [self.param['batchsize']])

            self.epoch = tf.placeholder(tf.uint16)
            self.is_training = tf.placeholder(tf.bool)
            self.drop = tf.placeholder(tf.float32)

            ####################################################################################

            logit, s = self.model(name='model',is_training=True)

            ####################################################################################

            gene_4_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ISP, logits=logit))

            score = s
            time_value = self.DFS
            event = self.st
            cox_value = self.cox_loss(score, time_value, event)

            loss = gene_4_cls_loss + 1*cox_value

            ci = self.__concordance_index(s, self.DFS, self.st)
            ####################################################################################

            global_step = tf.Variable(0)
            self.g_lr = tf.train.polynomial_decay(learning_rate=0.00005, global_step=global_step, decay_steps=500,end_learning_rate=1e-5, power=0.5, cycle=False)

            self.ADAM_opt = tf.train.GradientDescentOptimizer(1e-5).minimize(loss)

            ####################################################################################
            if os.path.exists('Validation.txt') and self.param['retrain']:
                os.remove('Validation.txt')
            if os.path.exists('Training.txt') and self.param['retrain']:
                os.remove('Training.txt')
            ####################################################################################
            with tf.Session(config=config, graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                ####################################################################################
                # Read the existed model
                if not self.param['retrain']:
                    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                    meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                    ckpt.load_ckpt(sess=sess, save_dir=checkpoints_dir, is_latest=True)
                    epoch_pre = int(meta_graph_path.split("-")[1].split(".")[0])
                else:
                    sess.run(tf.global_variables_initializer())
                    epoch_pre = 0
                ####################################################################################
                valid_taget_all = np.zeros(
                    (self.param['batchsize'] * (np.shape(valid_data)[0] // self.param['batchsize']), 4),
                    dtype=np.float32)
                valid_pprob_all = np.zeros(
                    (self.param['batchsize'] * (np.shape(valid_data)[0] // self.param['batchsize']), 4),
                    dtype=np.float32)

                train_taget_all = np.zeros(
                    (self.param['batchsize'] * (np.shape(train_data)[0] // self.param['batchsize']), 4),
                    dtype=np.float32)
                train_pprob_all = np.zeros(
                    (self.param['batchsize'] * (np.shape(train_data)[0] // self.param['batchsize']), 4),
                    dtype=np.float32)
                ####################################################################################
                for epoch in range(epoch_pre, n_epochs):
                    ####################################################################################
                    #
                    #  Training stage
                    #
                    ####################################################################################
                    # self.param['batchsize'] = 8
                    ploss_all, pgene_4_cls_loss_all, pcox_value_all, pci_all = 0, 0, 0, 0
                    Train_Num = [i for i in range(np.shape(train_img_aug)[0])]
                    random.shuffle(Train_Num)
                    count = 0
                    for iq in range(np.shape(train_img_aug)[0] // self.param['batchsize']):
                        train_final_input = train_img_aug[Train_Num[iq * self.param['batchsize']:(iq + 1) * self.param['batchsize']],:, :, :]
                        train_final_ISP = train_ISP_aug[Train_Num[iq * self.param['batchsize']:(iq + 1) * self.param['batchsize']], :]
                        train_final_DFS = train_DFS_aug[Train_Num[iq * self.param['batchsize']:(iq + 1) * self.param['batchsize']]]
                        train_final_st = train_st_aug[Train_Num[iq * self.param['batchsize']:(iq + 1) * self.param['batchsize']]]

                        feed_dict = {self.x: train_final_input, self.ISP: train_final_ISP, self.DFS: train_final_DFS,self.st: train_final_st, self.drop: 0.5, self.epoch: epoch}

                        _, ploss, pgene_4_cls_loss, pcox_value, pci = sess.run(
                            [self.ADAM_opt, loss, gene_4_cls_loss, cox_value, ci], feed_dict)

                        ploss_all += ploss
                        pgene_4_cls_loss_all += pgene_4_cls_loss
                        pci_all += pci
                        pcox_value_all += pcox_value
                        count += 1
                        # print(plogit)

                    print('epoch:-' + str(epoch + 1) + '\t' + 'ploss_all:' + str(
                        ploss_all / count) + '\t' + 'pgene_4_cls_loss_all:' + str(
                        pgene_4_cls_loss_all / count) + '\t' + 'pcox_value_all:' + str(
                        pcox_value_all / count) + '\t' + 'pci_all:' + str(
                        pci_all / count) + '\n')
                    
                    if (epoch + 1) % 1 == 0:
                        var_list = [var for var in tf.global_variables()]
                        print(time.asctime(time.localtime(time.time())),
                              'the %d-th iterations. Saving Models...' % epoch)
                        ckpt.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir=checkpoints_dir, global_step=epoch,
                                       var_list=var_list)
                        print("[*] Saving checkpoints SUCCESS! ")


if __name__ == '__main__':
    modelX = model()
    modelX.train()
