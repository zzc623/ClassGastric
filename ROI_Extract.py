#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/12/2021 1:34 AM 
# @Author : Zhicheng Zhang 
# @E-mail : zhicheng0623@gmail.com
# @Site :  
# @File : ROI_Extraction.py 
# @Software: PyCharm

import numpy as np
import os
import sys
import glob
import pydicom,copy
import Dataloader as Dataloader
import xlrd
import matplotlib.pyplot as plt
import cv2


def roi_extract(X, Label):

    if np.array(Label).ndim == 2:
        m = np.squeeze(Label[:, :])
    else:
        for ii in range(np.shape(Label)[0]):
            if np.sum(Label[ii,:,:]) != 0:
                m = np.squeeze(Label[ii,:,:])
                break

    X0 = X.copy()
    m0 = m.copy()

    img_max = np.max(X[X < 125])
    img_min = np.min(X[X > -125])
    X[X > 125] = img_max
    X[X < -125] = img_min

    X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    X = clahe.apply(X)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))


    x, y = np.where(m > 0)

    w0, h0 = m.shape
    x_min = max(0, int(np.min(x) - 5))
    x_max = min(w0, int(np.max(x) + 5))
    y_min = max(0, int(np.min(y) - 5))
    y_max = min(h0, int(np.max(y) + 5))

    m = m[x_min:x_max, y_min:y_max]
    X = X[x_min:x_max, y_min:y_max]

    X_m_1 = X.copy()
    X_m_1 = (X_m_1 - np.min(X_m_1[m > 0])) / (np.max(X_m_1[m > 0]) - np.min(X_m_1[m > 0]))
    X_m_1[m == 0] = 0
    X_m_2 = X.copy()
    X_m_2[m > 0] = 0
    h, w = X_m_1.shape
    if h < w:
        pad_1 = (w - h) // 2
        pad_2 = w - pad_1 - h
        X_m_1 = np.lib.pad(X_m_1, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0, 0))
        m = np.lib.pad(m, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0, 0))
    elif h >= w:
        pad_1 = (h - w) // 2
        pad_2 = h - pad_1 - w
        X_m_1 = np.lib.pad(X_m_1, ((0, 0), (pad_1, pad_2)), 'constant', constant_values=(0, 0))
        m = np.lib.pad(m, ((0, 0), (pad_1, pad_2)), 'constant', constant_values=(0, 0))

    if X_m_1.shape[0] != 160 or X_m_1.shape[1] != 160:

        X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(m, (160, 160), interpolation=cv2.INTER_CUBIC)

    if m0.shape[0] != 160 or m0.shape[1] != 160:
        m0 = cv2.resize(m0, (160, 160), interpolation=cv2.INTER_CUBIC)

    X_m_1 = (X_m_1 - np.min(X_m_1[m > 0])) / (np.max(X_m_1[m > 0]) - np.min(X_m_1[m > 0]))
    X_m_1[m <= 0] = 0

    X_m_1 = np.expand_dims(X_m_1, axis=2)
    XX = np.concatenate((X_m_1, X_m_1, X_m_1), axis=-1)

    return XX





Single_image_folder = './Data/Nanfang/ROI'
Multi_image_folder = './Data/Nanfang/ORG'

wb = xlrd.open_workbook(filename='Training.xlsx')
sheet1 = wb.sheet_by_index(0)
rows = sheet1.row_values(0)

data_org = np.empty((512, 512, 5), dtype=np.float32)
data_out = np.empty((160, 160, 5), dtype=np.float32)
datasets = []
for i in range(1,np.shape(sheet1.col_values(2))[0]):
    rows = sheet1.row_values(i)
    patient_number = np.uint16(rows[0])
    ISP = np.uint16(rows[1])
    DFS = np.uint16(rows[2])
    st = np.uint16(rows[3])
    patient_id = glob.glob(os.path.join(Single_image_folder, str( np.uint16(rows[0]))))[0]

    Data_Filename = glob.glob(os.path.join(patient_id,'*.dcm'))[0]
    Label_Filename = glob.glob(os.path.join(patient_id,'*.nii'))[0]

    CT_file_single = Dataloader.Dataloader(Data_Filename).load_dcm()
    Label = Dataloader.Dataloader(Label_Filename).load_nii()

    Instanse_number = int(pydicom.dcmread(Data_Filename).InstanceNumber)
    Series_des = (pydicom.dcmread(Data_Filename)).get('SeriesDescription')

    case_id = glob.glob(os.path.join(Multi_image_folder,patient_id.split('\\')[-1],'*.dcm'))

    for j in range(np.shape(case_id)[0]):
        mm = pydicom.dcmread(case_id[j],force=True)
        if mm.get('SeriesDescription') == Series_des:
            if mm.InstanceNumber == Instanse_number-2:
                data_org[:,:,0] = Dataloader.Dataloader(case_id[j]).load_dcm()#
            if mm.InstanceNumber == Instanse_number-1:
                data_org[:,:,1] = Dataloader.Dataloader(case_id[j]).load_dcm()#
            if mm.InstanceNumber == Instanse_number:
                data_org[:,:,2] = Dataloader.Dataloader(case_id[j]).load_dcm()#
            if mm.InstanceNumber == Instanse_number+1:
                data_org[:,:,3] = Dataloader.Dataloader(case_id[j]).load_dcm()#
            if mm.InstanceNumber == Instanse_number+2:
                data_org[:,:,4] = Dataloader.Dataloader(case_id[j]).load_dcm()#

   

    if np.array(Label).ndim == 2:
        m = np.squeeze(Label[:, :])
    else:
        for ii in range(np.shape(Label)[0]):
            if np.sum(Label[ii,:,:]) != 0:
                m = np.squeeze(Label[ii,:,:])
                break


    for jj in range(5):
        X = copy.deepcopy(data_org[:,:,jj])

        img_max = np.max(X[X < 125])
        img_min = np.min(X[X > -125])
        X[X > 125] = img_max
        X[X < -125] = img_min

        X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
        clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
        X = clahe.apply(X)
        X = (X - np.min(X)) / (np.max(X) - np.min(X))

        
        m0 = m.copy()
        w0, h0 = m0.shape
        x, y = np.where(m0 > 0)

        x_min = max(0, int(np.min(x) - 5))
        x_max = min(w0, int(np.max(x) + 5))
        y_min = max(0, int(np.min(y) - 5))
        y_max = min(h0, int(np.max(y) + 5))
        m0 = m0[x_min:x_max, y_min:y_max]




        X_m_1 = X[x_min:x_max, y_min:y_max]
        X_m_1 = (X_m_1 - np.min(X_m_1[m0 > 0])) / (np.max(X_m_1[m0 > 0]) - np.min(X_m_1[m0 > 0]))
        X_m_1[m0 == 0] = 0
        h, w = X_m_1.shape
        if h < w:
            pad_1 = (w - h) // 2
            pad_2 = w - pad_1 - h
            X_m_1 = np.lib.pad(X_m_1, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0, 0))
            m0 = np.lib.pad(m0, ((pad_1, pad_2), (0, 0)), 'constant', constant_values=(0, 0))
        elif h >= w:
            pad_1 = (h - w) // 2
            pad_2 = h - pad_1 - w
            X_m_1 = np.lib.pad(X_m_1, ((0, 0), (pad_1, pad_2)), 'constant', constant_values=(0, 0))
            m0 = np.lib.pad(m0, ((0, 0), (pad_1, pad_2)), 'constant', constant_values=(0, 0))

        if X_m_1.shape[0] != 160 or X_m_1.shape[1] != 160:

            X_m_1 = cv2.resize(X_m_1, (160, 160), interpolation=cv2.INTER_CUBIC)
            m0 = cv2.resize(m0, (160, 160), interpolation=cv2.INTER_CUBIC)

        if m0.shape[0] != 160 or m0.shape[1] != 160:
            m0 = cv2.resize(m0, (160, 160), interpolation=cv2.INTER_CUBIC)

        X_m_1 = (X_m_1 - np.min(X_m_1[m0 > 0])) / (np.max(X_m_1[m0 > 0]) - np.min(X_m_1[m0 > 0]))
        X_m_1[m0 <= 0] = 0

        data_out[:,:,jj] = copy.deepcopy(X_m_1)


    datasets.append({'patient_number': patient_number, 'img': copy.deepcopy(data_out), 'ISP': np.array([copy.deepcopy(ISP)]), 'DFS': np.array([int(copy.deepcopy(DFS))]),'st': np.array([int(copy.deepcopy(st))])})
np.save(os.path.join('./Resize_data','data.npy'), datasets)

