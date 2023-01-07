# ClassGastric
This is the implementation of the submitted paper “Biology-guided Deep Learning Predicts Prognosis and Cancer Immunotherapy Response”


## Data
A set of sample was stored in the folder "Data"

## Requirements
This code has been tested On Ubuntu 18.04  
The required 3rd libraries are listed as follow：  
Python        =3.6  
TensorFlow    =1.10  
cudatoolkit   =9.0  
cudnn         =7.6.5  
imgaug        =0.4.0  
numpy         =1.19.2  
scikit-learn  =0.24.1  
simpleitk     =2.0.2  
opencv-python =4.5.1.48  
xlrd  
pydicom  


## How to run  
1、Run the “ROI_Extract.py” to pre-process the patient data and convert the dicom into npy.    
2、Run the "Train.py" to re-train the well-designed model.  
