#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 11/16/2019 12:27 PM 
# @Author : Zhicheng Zhang 
# @E-mail : zhicheng0623@gmail.com
# @Site :  
# @File : Dataloader.py 
# @Software: PyCharm

import SimpleITK as sitk
import numpy as np

class Dataloader:
    def __init__(self,name):
        self.filename = name
    def load_dcm(self):

        # Read the .nii image containing the volume with SimpleITK:
        sitk_t1 = sitk.ReadImage(self.filename)
        # and access the numpy array:
        data = np.squeeze(sitk.GetArrayFromImage(sitk_t1))
        return data


    def load_nii(self):

        # Read the .nii image containing the volume with SimpleITK:
        sitk_t1 = sitk.ReadImage(self.filename)

        # and access the numpy array:
        data = np.squeeze(sitk.GetArrayFromImage(sitk_t1))

        return data

    def load_mhd(self):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(self.filename)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        origin = np.array(list(reversed(itkimage.GetOrigin())))

        # Read the spacing along each dimension
        spacing = np.array(list(reversed(itkimage.GetSpacing())))

        return ct_scan, origin, spacing

    def load_series_dicom(self):
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(self.filename)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        data = sitk.GetArrayFromImage(image)
        origin = np.array(list(reversed(image.GetOrigin())))
        size = image.GetSize()
        # print("Image size:", size[0], size[1], size[2])

        spacing = image.GetSpacing()
        # print("Image spacing:", spacing[0], spacing[1], spacing[2])

        return image, data, origin, size, spacing
