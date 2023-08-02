import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os
import shutil
from deepface import DeepFace

img1_path = r'C:\Users\ANFIELD\Desktop\tu\cn240\ann\Data_fer+old\deepfake\no_same_data\contempt\contempt_old\1.png'
test_path =r'C:\Users\ANFIELD\Desktop\tu\cn240\ann\Data_fer+old\deepfake\no_same_data\contempt\contempt_old'
input_img = cv2.imread(img1_path)


df = DeepFace.find(img_path=input_img,db_path=test_path, enforce_detection=False, model_name="Facenet512", detector_backend="mtcnn", prog_bar=False)
