# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:56:50 2023

@author: DuarteLopes
"""
import pickle
import numpy as np 
import cv2 

import mediapipe as mp
############################ LANDMARKS - New #################################
def getFaceCropped():
    return 

def oneChannel_threeChannel (img,h,w):
    if img.shape == (h,w): # if img is grayscale, expand
        #print ("convert 1-channel image to ", 3, " image.")
        new_img = np.zeros((h,w,3))
        for ch in range(3):
            for xx in range(h):
                for yy in range(w):
                    new_img[xx,yy,ch] = img[xx,yy]
        img = new_img
    img = img.astype(np.uint8)
    return img

def getLandmarksMediapipe (img, imgSize, nChannels):
    if nChannels != 3:
        img = cv2.resize(img,(imgSize,imgSize))
        imgRGB = oneChannel_threeChannel (img,imgSize,imgSize)
        landmarks = []
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                          refine_landmarks=False,
                                          min_detection_confidence=0.5)
        
        results = face_mesh.process(imgRGB)
        if results.multi_face_landmarks == None:
            landmarks.append("No Landmarks Detected")
        else: 
            for face_landmarks in results.multi_face_landmarks:
                #mp_drawing.draw_landmarks(imgEx, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
                #print('face_landmarks:', face_landmarks)
                for landmark in face_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
            
                    shape = img.shape 
                    relativeX = int(x * shape[1])
                    relativeY = int(y * shape[0])
                    
                    landmarks.append([relativeX,relativeY])
        return landmarks
    if nChannels == 3 :
        landmarks = []
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                          refine_landmarks=False,
                                          min_detection_confidence=0.5)
        
        results = face_mesh.process(img)
        if results.multi_face_landmarks == None:
            landmarks.append("No Landmarks Detected")
        else: 
            for face_landmarks in results.multi_face_landmarks:
                #mp_drawing.draw_landmarks(imgEx, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
                #print('face_landmarks:', face_landmarks)
                for landmark in face_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
            
                    shape = img.shape 
                    relativeX = int(x * shape[1])
                    relativeY = int(y * shape[0])
                    
                    landmarks.append([relativeX,relativeY])
        return landmarks
    
def points2Lines(img, faceLandmarks,index , imgSize,color, isClosed = False):
        points = []
        for i in index:
            point = faceLandmarks[i]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points], isClosed, color, thickness=2, lineType=cv2.LINE_8)
        #img = img.reshape((imgSize,imgSize,3))
        #img = img.reshape(imgSize)
        return img
    
def points2FillShape(img, faceLandmarks,index , imgSize,color):
        points = []
        for i in index:
            point = faceLandmarks[i]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [points], color)
        return img
    
def point2Rectangle(img, faceLandmarks,index, imgSize,color, H = False):
        if H == True:
            start_point = [faceLandmarks[index[0]][0], faceLandmarks[index[1]][1]] #represents the top left corner of rectangle
            end_point = index[2] #represents the bottom right corner of rectangle
            cv2.rectangle(img,  start_point, faceLandmarks[end_point], color, thickness = -1)
        else:
            start_point = index[3]
            end_point = index[4]
            cv2.rectangle(img,  faceLandmarks[start_point], faceLandmarks[end_point], color, thickness = -1)
            
        img = img.reshape((imgSize,imgSize,3))
        return img
def point2MaskSilhouette(imgOrig,imgSize, landmarks, indexes, color, channels = 1):
    #Create black image
    # In this case the image does not need 3 channels 
    if channels == 3:
        img = np.zeros((imgSize, imgSize, 3))
    else:
        img = np.zeros((imgSize, imgSize))
        
                # Nose bridge [0 - 5] - 5
    img = points2Lines(img, landmarks, indexes[0 : 5],imgSize, color)
    
                # Right eyebrow [5 - 12] - 7
    img = points2Lines(img, landmarks, indexes [5:12] ,imgSize, color)
    
                # Left eyebrow [12 - 19] - 7
    img = points2Lines(img, landmarks, indexes [12:19], imgSize, color) 
    
                # Nose [19 - 23] - 4
    img = points2FillShape(img, landmarks, indexes [19:23] , imgSize, color) #Fill
    
                # Left eye [23 - 39] - 16
    img = points2FillShape(img, landmarks, indexes [23 : 39] , imgSize, color) #Fill 
    
                # Right Eye [39 - 55] - 16
    img = points2FillShape(img, landmarks, indexes [39:55] , imgSize, color) #Fill
    
                # Jaw line [55 - 80]
    img = points2Lines(img, landmarks, indexes [55:80] ,imgSize, color)  
    
                #Mouth  #1 [80 - 101]
    img = points2FillShape(img, landmarks, indexes [80:101] , imgSize, color) #Fill
                
                #Mouth #2 [101 - 122]
    img =points2FillShape(img, landmarks, indexes [101:122] , imgSize, color)  #Fill  
    
    imgFinal = img * imgOrig
    return imgFinal    

def point2Silhouette(imgSize, landmarks, indexes, color, channels = 1):
        # Intervalo aberto no fim (?)
    
        #Create black image
        # In this case the image does not need 3 channels 
        if channels == 3:
            img = np.zeros((imgSize, imgSize, 3))
        else:
            img = np.zeros((imgSize, imgSize))
            
                    # Nose bridge [0 - 5] - 5
        img = points2Lines(img, landmarks, indexes[0 : 5],imgSize, color)
        
                    # Right eyebrow [5 - 12] - 7
        img = points2Lines(img, landmarks, indexes [5:12] ,imgSize, color)
        
                    # Left eyebrow [12 - 19] - 7
        img = points2Lines(img, landmarks, indexes [12:19], imgSize, color) 
        
                    # Nose [19 - 23] - 4
        img = points2Lines(img, landmarks, indexes [19:23] , imgSize, color, True) 
        
                    # Left eye [23 - 39] - 16
        img = points2Lines(img, landmarks, indexes [23 : 39] , imgSize, color, True)  
        
                    # Right Eye [39 - 55] - 16
        img = points2Lines(img, landmarks, indexes [39:55] , imgSize, color, True) 
        
                    # Jaw line [55 - 80]
        img = points2Lines(img, landmarks, indexes [55:80] ,imgSize, color)  
        
                    #Mouth  #1 [80 - 101]
        img = points2Lines(img, landmarks, indexes [80:101] , imgSize, color, True)
                    
                    #Mouth #2 [101 - 122]
        img = points2Lines(img, landmarks, indexes [101:122] , imgSize, color, True)     
        
        return img
    
def point2Mask(imgSize, landmarks, indexes, color):
                #Create black image
        img = np.zeros((imgSize, imgSize,3))
        
        #Horizontal Rectangle
        img = point2Rectangle(img, landmarks,indexes, imgSize,color, True)
        
        #Vertical Rectangle
        img = point2Rectangle(img, landmarks,indexes, imgSize,color)
        
        img = img/255
        #img = np.stack([img,img,img],axis=-1)
        return img
    
def drawingAllLandmarks(imgSize,kpts, color,radius, thickness):
        img = np.zeros((imgSize, imgSize,3))
        total = len(kpts)
        for i in range(total):
            coord = kpts[i]
            img = cv2.circle(img, coord, radius, color, thickness)
        return img
    
def indexes4Face():
        noseBridge = [5,195,197,6,168] #New 4 -> 5 pts
        rightEyebrow = [417,441,442,443,444,445,342] #New 5 -> 7 pts
        leftEyebrow = [113,225,224,223,222,221,193] #New 5 -> 7 pts
        #nose = [19,125,241,238,79,166,60,99,97,2,326,328,290,392,309,458,461,354,19] #New 5 -> 18 pts
        nose = [5,98,327,5] #New 5 -> 4pts 
        leftEye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7] #New 10 -> 16 pts
        rightEye = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382] #New 10 -> 16 pts
        jawLine = [21,34,227,137,177,215,138,135,169,170,140,171,175,396,369,395,394,364,367,435,401,366,447,264,251] #New 11 - > 25 pts
        outterLip = [0,37,39,40,185,61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0] #New 14 -> 21 + 21
        #innerLip = [11,72,73,74,184,76,77,90,180,85,16,315,404,320,307,306,408,304,303,302,11] #New 14 -> 21 + 21
        innerLip_2 = [12,38,41,42,183,62,96,89,179,86,15,316,403,319,325,292,407,272,271,268,12] # Extra Lip
        
        indexes = (noseBridge + 
                   rightEyebrow + 
                   leftEyebrow + 
                   nose + 
                   leftEye + 
                   rightEye + 
                   jawLine + 
                   outterLip +
                   #innerLip +
                   innerLip_2)
        return indexes
    
def indexes4Mask():
       
        #71y + 68x Canto Direito 
        horizontalRectangle = [71,68,346]
        verticalRectangle = [66,431]
        
        indexes = horizontalRectangle + verticalRectangle
        
        return indexes