import os
import cv2
from retinaface import RetinaFace
import pickle
from landmarksFunctions import getLandmarksMediapipe, point2Silhouette, indexes4Face
import numpy as np
import sys 
import pdb

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
########################## ALIGN ##########################################
# =============================================================================
#     ############################## Labels #################################
#     with open(pathLabel, "r") as file:
#         for line in file:
#             if img in line:
#                 last_char = line.strip()[-1]
#                 Labels.append(int(last_char))
#     if j % 10 == 0:
#         print("Img Cropped =>", j)
#     j = j + 1    
# =============================================================================
def indexesAlignmentMediapipe():
    rightEye = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]
    leftEye = leftEye = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
    return rightEye, leftEye

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


def faceAlign(path):
    pathDestino = '/nas-ctm01/datasets/public/DB/FER/RAFD_Align/Train'
    j = 1
    l = 1
    k = 1
    for img in os.listdir(path): 
        pathImg = os.path.join(pathDestino,img)

        image = cv2.imread(os.path.join(path,img))

        #sys.stdout.write("PathIMG:")
        #sys.stdout.write(str(os.path.join(path,img)))
        #sys.stdout.write('\n\n')
        landmark = getLandmarksMediapipe(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 224,3)
        #pdb.set_trace()
        if landmark != ['No Landmarks Detected']:
            image = faceAlignerMediapipe (image, landmark)
            # Convert from BGR to RGB
            cv2.imwrite(pathImg, image) 
            j = j + 1
        else:
            l = l + 1
        
        if k % 50 == 0:
            sys.stdout.write("Sample Aligned:")
            sys.stdout.write(str(j))
            sys.stdout.write('\n')
            sys.stdout.write("Samples NOT ALIGNED:")
            sys.stdout.write(str(l))
            sys.stdout.write('\n')

        k = k + 1
    return

def faceAlignRetinaFace(path,_type_):
    if _type_ == "Test":
        pathDestino = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Aligned/Test'
    if _type_ == "Train":
        pathDestino = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Aligned/Train'

    j = 1
    l = 1
    k = 1
    for img in os.listdir(path): 
        pathImg = os.path.join(pathDestino,img)
        image = RetinaFace.extract_faces(img_path = os.path.join(path,img), align = True)
        
        if len(image)>0:
                image = cv2.resize(cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB), (224, 224), cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(pathImg, image)
        else:
            l = l + 1

        if k % 50 == 0:
            sys.stdout.write("Sample Aligned:")
            sys.stdout.write(str(j))
            sys.stdout.write('\n')
            sys.stdout.write("Samples NOT ALIGNED:")
            sys.stdout.write(str(l))
            sys.stdout.write('\n')
        k = k + 1

def faceSilhouette(path,_type_,begin,end):
    if _type_ == "Test":
        pathDestino = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Sh/Test'
    if _type_ == "Train":
        pathDestino = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Sh/Train'

    indexes = indexes4Face()
    j = 1
    l = 1
    k = 1
    for img in os.listdir(path)[begin:end]: 
        pathImg = os.path.join(pathDestino,img)
        image = cv2.imread(os.path.join(path,img))
        #sys.stdout.write(os.path.join(path,img))
        #sys.stdout.write('\n')
        landmark = getLandmarksMediapipe(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 224,3)
        if landmark != ['No Landmarks Detected']:
            image = point2Silhouette(224, landmark, indexes, (255,255,255),3)
            cv2.imwrite(pathImg, image) 
            j = j+1
        else:
            l = l+1
        if k % 200 == 0:
            sys.stdout.write("Sample Aligned:")
            sys.stdout.write(str(j))
            sys.stdout.write('\n')
            sys.stdout.write("Samples NOT ALIGNED:")
            sys.stdout.write(str(l))
            sys.stdout.write('\n')
            k = k + 1


pathTrain = '/nas-ctm01/datasets/public/DB/FER_/Real-world Affective Faces (RAF) Database/Train'
#faceAlignRetinaFace(pathTrain,"Train")
pathTest = '/nas-ctm01/datasets/public/DB/FER_/Real-world Affective Faces (RAF) Database/Test'
##faceAlignRetinaFace(pathTest,"Test")

## Sh
pathTrain = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Align/Train'
faceSilhouette(pathTrain,"Train",0,3000)
pathTest = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Align/Test'
#faceSilhouette(pathTest,"Test")
