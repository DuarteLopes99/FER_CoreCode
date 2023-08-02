# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:21:32 2023

@author: DuarteLopes
"""

# RAFD
# =============================================================================
#     1: Surprise
#     2: Fear
#     3: Disgust
#     4: Happiness
#     5: Sadness
#     6: Anger
#     7: Neutral
# =============================================================================
import tensorflow as tf 
import os
import sys 
import cv2
import pickle
import numpy as np
import csv

#import pdb


            
def RAFD_csv(pathAlign, pathSh, pathLabel,csv_path):
    #Assumindo numAlign > numSh
    j = 1
    csvHead = ['ImgName','Path Aligned','Path Sh','Label']
    
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csvHead)
        
        for img in os.listdir(pathSh):
            pathImgSh = os.path.join(pathSh,img)
            pathImgAlign = os.path.join(pathAlign,img)
            with open(pathLabel, "r") as file:
                for line in file:
                    if img in line:
                        last_char = line.strip()[-1]
                        last_char = int(last_char) - 1
                        
            writer.writerow([img, pathImgAlign ,pathImgSh, last_char])

            if j % 100 == 0:
                sys.stdout.write("Img Processed =>")
                sys.stdout.write(str(j))
                sys.stdout.write('\n')
            j = j + 1  

def RAFD_ALL_csv(pathAlign, pathLabel,csv_path):
    #Assumindo numAlign > numSh
    j = 1
    csvHead = ['ImgName','Path Aligned','Label']
    
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csvHead)
        num_files = len(os.listdir(pathAlign))
        sys.stdout.write(str(num_files))
        sys.stdout.write('\n')
        for img in os.listdir(pathAlign):
            pathImgAlign = os.path.join(pathAlign,img)
            #sys.stdout.write("Img:")
            #sys.stdout.write(pathImgAlign)
            #sys.stdout.write('\n')
            with open(pathLabel, "r") as file:
                for line in file:
                    if img in line:
                        last_char = line.strip()[-1]
                        last_char = int(last_char) - 1
                        
            writer.writerow([img, pathImgAlign,last_char])

            if j % 100 == 0:
                sys.stdout.write("Img Processed =>")
                sys.stdout.write(str(j))
                sys.stdout.write('\n')
            j = j + 1  

def JAFFE_csv(pathAlign, pathSh, pathLabel,csv_path):
    #Assumindo numAlign > numSh
    j = 1
    csvHead = ['ImgName','Path Aligned','Path Sh','Label']
    
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csvHead)
        
        for img in os.listdir(pathSh):
            pathImgSh = os.path.join(pathSh,img)
            pathImgAlign = os.path.join(pathAlign,img)
            with open(pathLabel, "r") as file:
                for line in file:
                    if img in line:
                        last_char = line.strip()[-1]
                        last_char = int(last_char)
                        
            writer.writerow([img, pathImgAlign ,pathImgSh, last_char])

            if j % 100 == 0:
                sys.stdout.write("Img Processed =>")
                sys.stdout.write(str(j))
                sys.stdout.write('\n')
            j = j + 1  
            
def AffectNet_csv(pathAlign, pathSh, pathLabel,csv_path):
    #Assumindo numAlign > numSh
    j = 1
    csvHead = ['ImgName','Path Aligned','Path Sh','Label']
    
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csvHead)
        
        for img in os.listdir(pathSh):
            pathImgSh = os.path.join(pathSh,img)
            pathImgAlign = os.path.join(pathAlign,img)
            with open(pathLabel, "r") as file:
                for line in file:
                    if img in line:
                        last_char = line.strip()[-1]
                        last_char = int(last_char) - 1
                        
            writer.writerow([img, pathImgAlign ,pathImgSh, last_char])

            if j % 100 == 0:
                sys.stdout.write("Img Processed =>")
                sys.stdout.write(str(j))
                sys.stdout.write('\n')
            j = j + 1  

######## CSV File
xr=1
if xr>=0:
    pathAlign = '/nas-ctm01/datasets/public/DB/FER_/JAFFE_Align/Test'
    pathSh = '/nas-ctm01/datasets/public/DB/FER_/JAFFE_Sh/Test'
    pathLabel = '/nas-ctm01/datasets/public/DB/FER_/JAFFE/LabelsJAFFE.txt'
    csv_pathTest = '/nas-ctm01/datasets/public/DB/FER_/JAFFE_Test.csv'
    #JAFFE_csv(pathAlign, pathSh, pathLabel,csv_pathTest)

    pathAlign = '/nas-ctm01/datasets/public/DB/FER_/JAFFE_Align/Train'
    pathSh = '/nas-ctm01/datasets/public/DB/FER_/JAFFE_Sh/Train'
    pathLabel = '/nas-ctm01/datasets/public/DB/FER_/JAFFE/LabelsJAFFE.txt'
    csv_pathTrain = '/nas-ctm01/datasets/public/DB/FER_/JAFFE_Train.csv'
    #JAFFE_csv(pathAlign, pathSh, pathLabel,csv_pathTrain)

    ### RAFD
    # Train
    pathAlign = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Align/Train'
    pathLabel = '/nas-ctm01/datasets/public/DB/FER_/Real-world Affective Faces (RAF) Database/TrainLabel.txt'
    csv_pathTrain = '/nas-ctm01/datasets/public/DB/FER_/RAFD_ALL_Train.csv'
    #RAFD_ALL_csv(pathAlign, pathLabel ,csv_pathTrain)

    # Test
    pathAlign = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Align/Test'
    pathLabel = '/nas-ctm01/datasets/public/DB/FER_/Real-world Affective Faces (RAF) Database/TestLabel.txt'
    csv_pathTest = '/nas-ctm01/datasets/public/DB/FER_/RAFD_ALL_Test.csv'
    #RAFD_ALL_csv(pathAlign, pathLabel,csv_pathTest)

    # Train wOut Align
    path = '/nas-ctm01/datasets/public/DB/FER_/Real-world Affective Faces (RAF) Database/Train'
    pathLabel = '/nas-ctm01/datasets/public/DB/FER_/Real-world Affective Faces (RAF) Database/TrainLabel.txt'
    csv_pathTrain = '/nas-ctm01/datasets/public/DB/FER_/RAFD_ALL_NoAlign_Train.csv'
    #RAFD_ALL_csv(path, pathLabel ,csv_pathTrain)

    # Test wOut Align
    path = '/nas-ctm01/datasets/public/DB/FER_/Real-world Affective Faces (RAF) Database/Test'
    pathLabel = '/nas-ctm01/datasets/public/DB/FER_/Real-world Affective Faces (RAF) Database/TestLabel.txt'
    csv_pathTest = '/nas-ctm01/datasets/public/DB/FER_/RAFD_ALL_NoAlign_Test.csv'
    #RAFD_ALL_csv(path, pathLabel,csv_pathTest)


# Train AffectNet wOut Align
path = '/nas-ctm01/datasets/public/DB/FER_/AffectNet/train_set'
pathLabel = ''
csv_pathTrain = ''
RAFD_ALL_csv(path, pathLabel ,csv_pathTrain)

# Test AffectNet wOut Align
path = ''
pathLabel = ''
csv_pathTest = ''
RAFD_ALL_csv(path, pathLabel,csv_pathTest)