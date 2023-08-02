# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:10:05 2023

@author: DuarteLopes
"""

############################# DATA TREATMENT #################################
## Novas abordagens para: Detetar, Recortar e Alinhar as amostras 

# Uso RetinaFace ---> Detection & Aligment 

from landmarksFunctions import point2MaskSilhouette,drawingAllLandmarks,indexes4Mask, point2Mask , getLandmarksMediapipe, indexes4Face, point2Silhouette
from _dataPreProcessing_ import  roundPrediction, dataSpliter, faceDetectionSet, oneChannel_threeChannel, dataDivider, datDivider_mutipleList,dataAugmentation
import tensorflow as tf 
import os

import cv2
import mediapipe as mp

import matplotlib.pyplot as plt 
import random
import numpy as np

from retinaface import RetinaFace
from deepface import DeepFace

#from imutils import face_utils
import dlib

import pickle

import gc
with tf.device('/gpu:0'):
    
    def dataSetOriginalCreation(Datadirectory,Classes, name):
        images = []
        labels = []
        
        
        for category in Classes:
            path = os.path.join(Datadirectory,category) #Cria o path diretamente para a pasta de cada uma das emoções
            classes_num = Classes.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img))
                
                images.append(img_array)
                labels.append(classes_num)
        
        dataset = [images, labels]
        with open(name, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return images, labels
    
    def indexesAlignmentMediapipe():
        rightEye = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]
        leftEye = leftEye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
        return rightEye, leftEye
        
    
    def shape_to_np(shape,numPoints,dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((numPoints, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, numPoints):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords
    
    def faceAlignerMediapipe (img,landmark, desired_faceRatioX = 0.46, desired_faceRatioY = 0.4, imgSize = 224):
        leftEye_coord = []
        rightEye_coord = []
        rightEyeINDEX, leftEyeINDEX = indexesAlignmentMediapipe()
        for index in leftEyeINDEX:
            leftEye_coord.append(landmark [index])
        
        for index in rightEyeINDEX:
            rightEye_coord.append(landmark [index])
        
        #Calcular o valor CENTRAL de cada olho
        avgLeftX = sum(coord[0] for coord in leftEye_coord) / len(leftEye_coord)
        avgLeftY = sum(coord[1] for coord in leftEye_coord) / len(leftEye_coord)
        center_leftEye = (int(avgLeftX), int (avgLeftY))
        
        avgRightX = sum(coord[0] for coord in rightEye_coord) / len(rightEye_coord)
        avgRightY = sum(coord[1] for coord in rightEye_coord) / len(rightEye_coord)
        center_rightEye = (int(avgRightX), int(avgRightY))
        
        #Calcular o angulo entre os CENTROS olhos
        dY = center_rightEye[1] - center_leftEye[1]
        dX = center_rightEye[0] - center_leftEye[0]
        angle = np.degrees(np.arctan2(dY, dX))
                
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        
        # Valor real da distancia desejada entre os olhos; Mutiplicamos a nossa relaçao pelo valor da imagem
        desiredDist = desired_faceRatioX * imgSize
        
        scale = desiredDist / dist
        ##### Passo Final ######
        # Rodar a imagem de acordo com esta informação toda 
        
        eyesMidPoint = ((center_leftEye[0] + center_rightEye[0]) // 2,
            (center_leftEye[1] + center_rightEye[1]) // 2)
        
        xMid = int(eyesMidPoint[0])
        yMid = int(eyesMidPoint[1])
        M = cv2.getRotationMatrix2D((xMid,yMid), angle, scale)
        
        desired_Y = imgSize * desired_faceRatioY
        tY = desired_Y
        
        # Forçar a imagem a transladar para que os olhos estejam sempre na mesma
        # posição relativa em relação a tY
        
        #M[0, 2] += (tX - eyesMidPoint[0]
        M[1, 2] += (tY - yMid)
        
        
        alignIMG = cv2.warpAffine(img, 
                                M, 
                                (imgSize, imgSize), 
                                flags = cv2.INTER_CUBIC)
        
        # return the aligned face
        return alignIMG
        
    def faceCropper_RetinaFace (Datadirectory,Classes):  
        DataAlign = []
        DataAlignResize = []
        Labels = []
        landmarkByIMG = []
        
        i = 1
        for category in Classes:
                path = os.path.join(Datadirectory,category) #Cria o path diretamente para a pasta de cada uma das emoções
                classes_num = Classes.index(category)
                for img in os.listdir(path):
                    img_array_cropped = RetinaFace.extract_faces(img_path = os.path.join(path,img), align = False)
                    if len(img_array_cropped)>0:
                        img_array_cropped = img_array_cropped[0]
                        img_array_cropped = cv2.resize(img_array_cropped,(224,224))
                        landmarks = getLandmarksMediapipe(img_array_cropped, 224,3)
                        
                    DataAlign.append(img_array_cropped)
                    DataAlignResize.append(img_array_cropped)
                    Labels.append(classes_num)
                    landmarkByIMG.append(landmarks)
        
                    if i % 10 == 0:
                        print("Img Cropped =>", i)
                    i = i + 1
        #EXTRA
        img_and_landmarks = [DataAlignResize, landmarkByIMG]
        
        return DataAlign, Labels, img_and_landmarks
    
    def faceCropper_Mediapipe (Data, Labels): 
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        DataAlign = []
        DataAlignResize = []
        landmarkByIMG = []
        
        i = 1
        for img in Data:
            results = face_detection.process(img)
            if results.detections == None:
                imgCropped = ['No Keypoints']
                landmarks = ['No Landmarks Detected']
            else:
                face_coords = results.detections[0].location_data.relative_bounding_box
                h, w, _ = img.shape
                xmin = int(face_coords.xmin * w)
                ymin = int(face_coords.ymin * h)
                xmax = int((face_coords.xmin + face_coords.width) * w)
                ymax = int((face_coords.ymin + face_coords.height) * h)

                # Crop the face from the input image
                imgCropped = img[ymin:ymax, xmin:xmax]
                imgCropped = cv2.resize(imgCropped,(224,224))
                landmarks = getLandmarksMediapipe(imgCropped, 224,3)
                
            DataAlign.append(imgCropped)
            DataAlignResize.append(imgCropped)
            landmarkByIMG.append(landmarks)

            if i % 10 == 0:
                print("Img Cropped =>", i)
            i = i + 1
        #EXTRA
        img_and_landmarks = [DataAlignResize, landmarkByIMG]
        
        return DataAlign, Labels, img_and_landmarks
    
    def reduceDataSet(raw, dataset, labels, _type_ = 0):
        index = len(dataset)-1
        i = 0
        for sample in reversed(dataset):
            if sample == 'No Keypoints' or sample == ['No Keypoints']:                
                if _type_ == 1:
                        del raw[index]
                        del dataset[index]
                        del labels[index]

                else:
                    del dataset[index]
                    del labels[index]
                print("Sample ", index, " Eliminated")
                i = i + 1
            index = index - 1
        print ("Total Smaples Eliminated =",i)
        
        return raw, dataset,labels
    
    def delLandmarks (landmarks):
        index = len(landmarks)-1
        for sample in reversed(landmarks):
            if sample == ['No Landmarks Detected']:
                del landmarks[index]
            index = index - 1
                
        return landmarks
    def create_dataSet_align(Datadirectory,Classes,save,name ="XXXXXX", _type_= 0):
            dataCK_Aligned = []
            i = 0
            # STEP #1 => Detect&Crop Faces (RetinaFace)
            # STEP #2 => Resize to 224x224
            dataCK, Labels,img_AND_Land = faceCropper_RetinaFace (Datadirectory,Classes)
            #dataCK, Labels,img_AND_Land = faceCropper_Mediapipe (data, labels)
            landmarks = []
            # STEP #3 => Extract Keypoints& Face Align
            for i in range(len(img_AND_Land[0])):
                img = img_AND_Land[0][i]
                landmark = img_AND_Land[1][i]
                if landmark == ['No Landmarks Detected']:
                    dataCK_Aligned.append('No Keypoints')
                    landmarks.append(landmark)
                else:
                    alignedSample = faceAlignerMediapipe (img, landmark)
                    dataCK_Aligned.append(alignedSample)
                    landmarks.append(landmark)
                print("Sample Aligned,",i)
                i = i+1
                
            # STEP #4 => Eliminate faces whitout keypoints 
            dataCK, dataCK_Aligned, Labels = reduceDataSet(dataCK, dataCK_Aligned, Labels, _type_)
            
            
            # STEP #5 => Saving in a pickle format 
            dataSetAligned = [dataCK_Aligned, Labels] 
            if save == True:
                with open(name, 'wb') as handle:
                    pickle.dump(dataSetAligned, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return dataCK, dataCK_Aligned, Labels
    
    def create_dataSet_align_Mediapipe(data, labels,save,name ="XXXXXX", _type_= 0):
        dataCK_Aligned = []
        i = 0
        # STEP #1 => Detect&Crop Faces (RetinaFace)
        # STEP #2 => Resize to 224x224
        dataCK, Labels,img_AND_Land = faceCropper_Mediapipe (data, labels)
        landmarks = []
        # STEP #3 => Extract Keypoints& Face Align
        for i in range(len(img_AND_Land[0])):
            img = img_AND_Land[0][i]
            landmark = img_AND_Land[1][i]
            if landmark == ['No Landmarks Detected']:
                dataCK_Aligned.append('No Keypoints')
                landmarks.append(landmark)
            else:
                alignedSample = faceAlignerMediapipe (img, landmark)
                dataCK_Aligned.append(alignedSample)
                landmarks.append(landmark)
            print("Sample Aligned,",i)
            i = i+1
            
        # STEP #4 => Eliminate faces whitout keypoints 
        dataCK, dataCK_Aligned, Labels = reduceDataSet(dataCK, dataCK_Aligned, Labels, _type_)
        
        
        # STEP #5 => Saving in a pickle format 
        dataSetAligned = [dataCK_Aligned, Labels] 
        if save == True:
            with open(name, 'wb') as handle:
                pickle.dump(dataSetAligned, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return dataCK, dataCK_Aligned, Labels
        
    #DataFolder_CK = r"C:\Users\Duarte Lopes\Desktop\GitHub - Repo\Database\CK+_Complete"
    #Classes = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    #nameOriginal = r"C:\Users\Duarte Lopes\Desktop\GitHub - Repo\Database\CK+_Aligned\CK_Original"
    #data, labels = dataSetOriginalCreation(DataFolder_CK,Classes, nameOriginal)
    
    ########### Retina Face Cropping + Mediapipe Align ####################
    
    #dataCK_raw, labelsCK, img_AND_Land = faceCropper_RetinaFace (DataFolder_CK,Classes)
    #nameRaw = r"C:\Users\Duarte Lopes\Desktop\GitHub - Repo\Database\CK+_Aligned\CK_CropRetina_AlignMediapipe"
    #dataCK, dataCK_align_resize,labelsCK = create_dataSet_align(DataFolder_CK,Classes, True, nameRaw, 1)
    
    ############### Mediapipe Croping + Align ##############################
    
    #dataCropped, labels, img_AND_Land = faceCropper_Mediapipe (data, labels)
    #nameCroppedMediapipe = r"C:\Users\Duarte Lopes\Desktop\GitHub - Repo\Database\CK+_Aligned\CK_CropMediapipe"
    #dataCK, dataCK_align_resize,labelsCK = create_dataSet_align_Mediapipe(data, labels, True, nameCroppedMediapipe, 1)
    
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
############################# Shilouette ######################################

    def creationSilhouetteSet_v2_0 (Dataset, Labels, imgSize, save, name="XXXXXX", _name_ = "YYYYY", channels = 1):
        indexes = indexes4Face()
        nSilhouette = 0
        
        totalSamples = len(Dataset)
        landmarks = []
        newFaces = []
        for i in range(totalSamples):
            if i != 'No Keypoints':  
                kpts = getLandmarksMediapipe(Dataset[i], imgSize,3)
                landmarks.append(kpts)
        print("[All Landmarks Computes]")
        for m in range(len(landmarks)):
            if landmarks[m] == ['No Landmarks Detected']:
                newFaces.append('No Keypoints')
            else:
                if channels == 3:
                    img = point2Silhouette(224, landmarks[m], indexes, (255,255,255),3)
                else:
                    img = point2Silhouette(224, landmarks[m], indexes, (255,255,255))
                newFaces.append(img) 
                
                if nSilhouette % 10 == 0:
                    print("Silhouettes Computed:", nSilhouette)
                nSilhouette = nSilhouette + 1

        
        
        Dataset, newFaces, Labels = reduceDataSet(Dataset, newFaces , Labels, 1)
        
        dataSetAligned = [Dataset, Labels] 
        if save == True:
            with open(_name_, 'wb') as handle:
                pickle.dump(dataSetAligned, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        sillhoutteDataSet = [newFaces, Labels] 
        if save == True:
            with open(name, 'wb') as handle:
                pickle.dump(sillhoutteDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return newFaces, Labels, dataSetAligned 
    
    
    with open("C://Users//Duarte Lopes//Desktop//GitHub - Repo//Database//CK+_Aligned//CK_CropRetina_AlignMediapipe", 'rb') as handle:
        DataSetAligned = pickle.load(handle)  
    ImgAligned,Labels = DataSetAligned
    
    # Add a _ in the new face, theoretically NOT same set. Cuz Silhouette is applied in ALIGN Images
    # Mediapipe is now computed is NEW ALIGNED IMAGES => [=/= Results]
    nameAligned = r"C:\Users\Duarte Lopes\Desktop\GitHub - Repo\Database\CK+_Aligned\_CK_CropRetina_AlignMediapipe_"
    nameSilhouette = r"C:\Users\Duarte Lopes\Desktop\GitHub - Repo\Database\CK+_Aligned\CK_SilhouetteAligned_Mediapipe_CH3"
    imgSize = 224
    channels = 3 
    silhouetteFaces,labelsCK, dataSetAlignedCUT = creationSilhouetteSet_v2_0 (ImgAligned, Labels, imgSize, True, nameSilhouette, nameAligned, channels)
    
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
############################# Masked Shilouette ######################################    
    def creationMaskedSilhouetteSet (Dataset, Labels, imgSize, save, name="XXXXXX", _name_ = "YYYYY", channels = 1):
        indexes = indexes4Face()
        nSilhouette = 0
        
        totalSamples = len(Dataset)
        landmarks = []
        newFaces = []
        for i in range(totalSamples):
            if i != 'No Keypoints':  
                kpts = getLandmarksMediapipe(Dataset[i], imgSize,3)
                landmarks.append(kpts)
        print("[All Landmarks Computes]")
        for m in range(len(landmarks)):
            if landmarks[m] == ['No Landmarks Detected']:
                newFaces.append('No Keypoints')
            else:
                if channels == 3:
                    img = point2MaskSilhouette(Dataset[m],224, landmarks[m], indexes, (255,255,255),3)
                else:
                    img = point2MaskSilhouette(Dataset[m],224, landmarks[m], indexes, (255,255,255))
                newFaces.append(img) 
                
                if nSilhouette % 10 == 0:
                    print("Silhouettes Computed:", nSilhouette)
                nSilhouette = nSilhouette + 1

        Dataset, newFaces, Labels = reduceDataSet(Dataset, newFaces , Labels, 1)
        
        dataSetAligned = [Dataset, Labels] 
        if save == True:
            with open(_name_, 'wb') as handle:
                pickle.dump(dataSetAligned, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        sillhoutteDataSet = [newFaces, Labels] 
        if save == True:
            with open(name, 'wb') as handle:
                pickle.dump(sillhoutteDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return newFaces, Labels, dataSetAligned 
    
# =============================================================================
#     ######### Printing examples ######################
#     for i in range(10):
#         j = random.randint(0, len(dataCK_raw))
#         plt.figure(i)
#         plt.title("Image number"+str(i))
#         plt.subplot(1,2,1)
#         plt.imshow(dataCK_raw[j])
#         plt.subplot(1,2,2)
#         plt.imshow(dataCK_align[j])
# =============================================================================


    
    