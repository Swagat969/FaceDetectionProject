# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:20:39 2020

@author: SwagatSourav
"""
import os 
import cv2
import pickle
import numpy as np

data_dir = os.path.join(os.getcwd(),"pickledata")
img_dir = os.path.join(os.getcwd(),"image")


def preprocess(image):
    image = cv2.resize(image,(100,100))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    return image


images = []
labels = []

for i in os.listdir(img_dir):
    image=cv2.imread(os.path.join(img_dir,i))
    image = preprocess(image)
    images.append(image)
    labels.append(i.split('_')[0])
    
    
images = np.array(images)
labels = np.array(labels)

with open(os.path.join(data_dir,"images.p"),"wb") as f:
    pickle.dump(images,f)
    
with open(os.path.join(data_dir,"labels.p"),"wb") as f:
    pickle.dump(labels,f)
    

    


    


