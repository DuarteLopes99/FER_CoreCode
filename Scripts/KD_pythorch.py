# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:38:42 2023

@author: DuarteLopes
"""

#################### Knowledge Destillation #################################

#Train CNN with Original Samples  (Teacher)
#Train SimplerCNN with Silhouette Samples (Student)

###################### PyTorch EDITION #######################################
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary
import pandas as pd
import sys 
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import wandb
from losses import TripletLoss, LabelSmoothingCrossEntropy

import albumentations as A
from albumentations.pytorch import ToTensorV2

import time

import cv2

start_time = time.time()

# FUNCTIONS FOR TRAIN & VALIDATION
#############################################################################################################################
class CustomDataSet (Dataset):
    def __init__(self, csvFile, dirImg, typeImg, transform=None):
        self.data_frame = pd.read_csv(csvFile)
        self.dirImg = dirImg
        self.transform = transform
        self.typeImg = typeImg

    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        if self.typeImg == 'Orign':
            img_path = os.path.join(self.dirImg, self.data_frame.iloc[idx, 1]) 
        elif self.typeImg == 'Sh':
            img_path = os.path.join(self.dirImg, self.data_frame.iloc[idx, 2]) 
        else:
            raise ValueError("ImgType must be either 'Orign' or 'Sh'")

        img_name = self.data_frame.iloc[idx, 0]
        image = Image.open(img_path)
        label = self.data_frame.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return img_name, image, label

class CustomDataSet_KD (Dataset):
    def __init__(self, csvFile, dirImg, dirSh, NoTransform, transform=None):
        self.data_frame = pd.read_csv(csvFile)
        self.dirImg = dirImg
        self.dirSh = dirSh
        self.transform = transform
        self.NoTransform = NoTransform

    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        img_path_Or = os.path.join(self.dirImg, self.data_frame.iloc[idx, 1]) 
        img_path_Sh = os.path.join(self.dirSh, self.data_frame.iloc[idx, 2])  
        image_Or = Image.open(img_path_Or)
        image_Sh = Image.open(img_path_Sh)
       
        img_name = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 3]
    
        if self.transform is not None:
            
            # TorchVision Transform
            #image_Sh = self.transform(image_Sh)
            #image_Or = self.NoTransform(image_Or)

            # PIL to Array to Use Albumations 
            Sh_Arr = np.array(image_Sh)
            Or_Arr = np.array(image_Or)

            # Albumations trasnform
            image_Sh = self.transform(image = Sh_Arr)["image"]
            #image_Or = self.transform(image = Or_Arr)["image"]
            image_Or = self.NoTransform(image = Or_Arr)["image"]

        return img_name, image_Or, image_Sh, label


transform_112 = transforms.Compose([
    #torchvision.transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])

transform_W_Augm_112 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((112, 112)),
    #transforms.RandomApply([
        #transforms.RandomRotation(20),
        #transforms.RandomCrop(112, padding=32)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])


transform_Albumations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(112,112),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()])

NoTransform_Albumations = A.Compose([
    A.Resize(112,112),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()])

# Loss_Function
def distillation_loss(outputs, labels, teacher_logits, T, alpha, softmax = False):

    LSCE = LabelSmoothingCrossEntropy(smoothing=0.2)
    # Compute SOFTMAX 
    SMax_teacher_logits = nn.functional.softmax(teacher_logits / T, dim=1)
    #SMax_outputs = nn.functional.softmax(outputs / T, dim=1)
    LogSMax_outputs = nn.functional.log_softmax(outputs / T, dim=1)    #log_softmax cuz of KL_Div
    
    # Compute the hard targets loss
    # Crossentropy vs LSCE

    # CE
    Smax_outputs =  nn.functional.softmax(outputs / T, dim=1)
    hard_lossT = torch.nn.functional.cross_entropy(Smax_outputs,labels)
    hard_loss = torch.nn.functional.cross_entropy(outputs, labels)

    # LSCE
    CE_loss = nn.CrossEntropyLoss()(outputs, labels)
    LSCE_loss = LSCE(outputs, labels)
    #hard_loss =  2 * LSCE_loss + CE_loss

    soft_loss = nn.MSELoss()(outputs, teacher_logits)
    soft_lossT = nn.MSELoss()(outputs/T, teacher_logits/T)
    
    soft_loss_SMaxT_KL  = torch.nn.KLDivLoss()(LogSMax_outputs, SMax_teacher_logits) 
    

    soft_loss_SMaxT_KL = torch.nn.KLDivLoss(reduction='batchmean')(LogSMax_outputs,SMax_teacher_logits) # KL_Div
    #soft_loss_SMaxT = nn.MSELoss()(LogSMax_outputs,SMax_teacher_logits) # MSE
    
    #teacher_logits = teacher_logits / 10000
    #soft_loss = nn.functional.kl_div(outputs, teacher_logits, reduction='batchmean',log_target = True) * T * T

    if softmax == True:
        # Loses with Alpha
        softLossCombine = soft_loss_SMaxT_KL * (T*T * 2.0 * alpha)
        hardLossCombine = hard_loss * (1. - alpha)

        distillation_lossT = softLossCombine + hardLossCombine
        
        return distillation_lossT, hardLossCombine, softLossCombine
    else :
        # Combining losses
        distillation_loss = (1 - alpha) * hard_loss + alpha * soft_loss
        distillation_lossT = (1 - alpha) * hard_loss + alpha * soft_lossT
        return distillation_loss, distillation_lossT, (1 - alpha) * hard_loss, alpha * soft_loss, alpha * soft_lossT

def train_val_KD(model, teacherModel,train_loader,val_loader ,epochs, KD_loss,T,alpha,optimizer):
    train_accValues = []
    train_lossValues = []
    val_accValues = []
    val_lossValues = []
    
    softLossValue = []

    for epoch in range(epochs):
        sys.stdout.write("[Epoch:]")
        sys.stdout.write(str(epoch + 1))
        sys.stdout.write('\n')

        lossEpoch = 0.0
        lossEpoch_T = 0.0
        
        termo_1 = 0.0
        termo_2 = 0.0
        
        correct = 0
        total = 0
        
        val_correct = 0
        val_total = 0

        correctNum = 0
        correctNumStu = 0

        # ERRO
        #nameInpust2, inputs_teacher2,inputs2, labels2
        for i, (nameInputs, inputs_teacher,inputs,labels) in enumerate(train_loader):
        #for i, ((nameInputs, inputs, labels), (nameInputs,inputs_teacher, labels_teacher)) in enumerate(zip(StudentTrain_loader, TeacherTrainLoader)):
            # Cimpute TeacherLogits of Teacher Model 
            with torch.no_grad():
                teacher_logits = teacherModel(inputs_teacher)
                #teacher_logits = teacherPredictions_Train.numpy()
                #teacher_logits = torch.Tensor(teacher_logits)

            # calling zero_grad() at the start of each iteration ensures that the gradients are not accumulated from previous iterations
            optimizer.zero_grad()
            # Compute Train -> DestillationLoss
            predsStudents = model(inputs)
            
            # Knowledge Distllation Loss
            #loss, lossT ,hard_loss, soft_loss, soft_loss_T = KD_loss(predsStudents,labels,teacher_logits,T=T, alpha=alpha)
            lossT, hard_loss, soft_loss = KD_loss(predsStudents, labels, teacher_logits, T=T, alpha=alpha, softmax=True)

            lossT.backward()
            optimizer.step()
            
            #Correct Number of teacher Logits
            _, pred = torch.max(teacher_logits.data, 1)  # get the index of the max log-probability  
            correctNum += (pred == labels).sum().item()


            # Update TotalLoss
            ###lossEpoch += loss.item()
            lossEpoch_T += lossT.item()
            wandb.log({"epoch": epoch, "batchNum": i+1, "BatchLoss": lossT.item()})
            
            termo_1 += hard_loss.item()
            termo_2 += soft_loss.item()
            
            # Calculate accuracy - Calculate the number os correct predictions
            _, predicted = torch.max(predsStudents.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # Print accuracy for each epoch
        accuracy = 100 * correct / total
        train_accValues.append(accuracy)
        stdTrainAcc = np.std(train_accValues)

        # Print loss for each epoch
        softLossValue.append(termo_2)
        train_lossValues.append(lossEpoch_T)
        avgLoss = lossEpoch_T/len(train_loader)

        # Log training metrics
        wandb.log({"epoch": epoch, "Train_KDloss": lossEpoch_T, "train_acc": accuracy, "SoftLoss (Destil):": termo_2,"HardLoss (Student)":termo_1,"AvgLoss":avgLoss,
                   "CorrectTeacher_vs_label":correctNum,"CorrectStudent_vs_label":correct})

        #OutFile
        sys.stdout.write("Total Correct =>")
        sys.stdout.write(str(correct))
        sys.stdout.write('\s')
        sys.stdout.write("Total=>")
        sys.stdout.write(str(total))
        sys.stdout.write('\n')
        
        ################################################
        ############## VALIDATION  #####################
        ################################################
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            #nameInpust2, inputs_teacher2,inputs2, labels2
            for i, (nameInputs, inputs_teacher,inputs,labels) in enumerate(val_loader):
                
                # Preds
                valOutputs = model(inputs)
                loss = nn.CrossEntropyLoss()(valOutputs, labels)
                #loss = nn.functional.cross_entropy(valOutputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(valOutputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        # Print accuracy for each epoch
        val_accuracy = 100 * val_correct / val_total
        val_accValues.append(val_accuracy)
        #Standart Desviation
        avgValLoss = val_loss/len(val_loader)

        #OutFile
        sys.stdout.write("Total Correct =>")
        sys.stdout.write(str(val_correct))
        sys.stdout.write('\s')
        sys.stdout.write("Total=>")
        sys.stdout.write(str(val_total))
        sys.stdout.write('\n')

        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_acc": val_accuracy, "AvgLossVal":avgValLoss})
        val_lossValues.append(val_loss)
        
    return train_accValues,train_lossValues, val_accValues, val_lossValues, softLossValue, model
    
def send2Excel (array1, array2, array3, array4, array5, NameFileExcel):
    panda1 = pd.DataFrame (array1)
    panda2 = pd.DataFrame (array2)
    panda3 = pd.DataFrame (array3)
    panda4 = pd.DataFrame (array4)
    panda5 = pd.DataFrame (array5)
    
    with pd.ExcelWriter(NameFileExcel) as writer:
        panda1.to_excel(writer, sheet_name= "TrainAcc")
        panda2.to_excel(writer, sheet_name= "Val_Acc")
        panda3.to_excel(writer, sheet_name= "TrainLoss")
        panda4.to_excel(writer, sheet_name= "Val_Loss")
        panda5.to_excel(writer, sheet_name= "Soft_Loss")
    
    return 'Done To Excel'
################################################################################################################################################
################################################################### DATA #######################################################################
data = "RAFD"

if data == "RAFD":
    pathCSV = '/nas-ctm01/homes/pdcunha/KD_FER/Excel_DataSets/RAFD_Train.csv'
    pathFolderOrign = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Align/Train'
    pathFolderSh = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Sh/Train'
if data == "CK+":
    pathCSV = '/nas-ctm01/homes/pdcunha/KD_FER/Excel_DataSets/CK+_Train.csv'
    pathFolderOrign = '/nas-ctm01/datasets/public/DB/FER_/CK+_Align/Train'
    pathFolderSh = '/nas-ctm01/datasets/public/DB/FER_/CK+_Sh/Train'
if data == "JAFFE":
    pathCSV = '/nas-ctm01/homes/pdcunha/KD_FER/Excel_DataSets/JAFFE_Train.csv'
    pathFolderOrign = '/nas-ctm01/datasets/public/DB/FER_/JAFFE_Align/Train'
    pathFolderSh = '/nas-ctm01/datasets/public/DB/FER_/JAFFE_Sh/Train'


#DataSetAll = CustomDataSet_KD(csvFile = pathCSV, dirImg = pathFolderOrign, dirSh = pathFolderSh, NoTransform = transform_112, transform = transform_W_Augm_112)
DataSetAll = CustomDataSet_KD(csvFile = pathCSV, dirImg = pathFolderOrign, dirSh = pathFolderSh, NoTransform = NoTransform_Albumations, transform=transform_Albumations)
OrDataSet = CustomDataSet(csvFile = pathCSV, dirImg = pathFolderOrign, typeImg='Orign',transform=transform_112)


#### Split the dataset into training and validation sets
train_idx, val_idx = train_test_split(range(len(DataSetAll)), test_size=0.2, random_state=42)

Dataset_Train = Subset(DataSetAll, train_idx)
Dataset_Val = Subset(DataSetAll, val_idx)

# DataLoader
trainLoaderOr = DataLoader(OrDataSet, batch_size=32)
train_loader = DataLoader(Dataset_Train, batch_size=32,shuffle=True)
val_loader = DataLoader(Dataset_Val, batch_size=32)

############################################################ MODEL #############################################################################
# Teacher (Trained Model)
modelName_ = 'Model_10-Orign_RAF_Frozen_60_IrisNet100 - LSCE'
PATH =  '/nas-ctm01/homes/pdcunha/KD_FER/TeacherModelStats_Back2Start/Model_10-Orign_RAF_Frozen_60_IrisNet100 - LSCE'
teacherModel = torch.load(PATH)
teacherModel.eval()

# Studend Model - ResNet18
resNet18 = models.resnet18(weights = 'IMAGENET1K_V1')
num_ftrs = resNet18.fc.in_features 
resNet18.fc = nn.Linear(num_ftrs, 7)

# Student Model  - MobileNetV2
mobileNetModel = models.mobilenet_v2(weights = "MobileNet_V2_Weights.IMAGENET1K_V2")
num_ftrs = mobileNetModel.classifier[-1].in_features
mobileNetModel.classifier[-1]  = nn.Linear(num_ftrs, 7)


# Student Model - MobileNetV3
# Torch dinamically adaptable (?)
mobileV3 = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
num_ftrs = mobileV3.classifier[-1].in_features
mobileV3.classifier[-1]  = nn.Linear(num_ftrs, 7)

#Student Model - Iris50 s\ Pesos 
from elasticFace.backbones.iresnet import iresnet100, iresnet50
from elasticFace.config.config import config as cfg
backbone = iresnet50(num_features=cfg.embedding_size)
num_ftrs = backbone.fc.in_features
backbone.fc = nn.Linear(num_ftrs, 7)
backbone.features = nn.BatchNorm1d(7)
iris50 = backbone

# Choose Student Model
studentModel = mobileV3
#studentModel = iris50
#studentModel = resNet18
#studentModel = mobileNetModel

# Hyperparameters
lr = 1e-5
epochs = 150
batch_size = 32
T = 4
alpha = 0.5

# Optimizer & Loss
weight_decay = 1e-5
beta1 = 0.9
beta2 = 0.999

#optimizer = optim.SGD(studentModel.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(studentModel.parameters(), lr=lr)
#optimizer = optim.Adam(studentModel.parameters(), lr = lr, weight_decay=weight_decay)

criterion = LabelSmoothingCrossEntropy(smoothing=0.2)


# for i, sample in enumerate(zip(trainLoaderOr,train_loader)):
# #for i, (nameInputs, inputs, inputs_teacher,labels) in enumerate(DataSetAll):
#     sample_from_loader1, sample_from_loader2 = sample
#     nameInputs1, inputs_teacher1,labels1 = sample_from_loader1
#     nameInpust2, inputs_teacher2,inputs2, labels2 = sample_from_loader2


#     #sys.stdout.write(str(sample_from_loader1))
#     sys.stdout.write(str("NameInputs"))
#     sys.stdout.write(str(nameInputs1))
#     sys.stdout.write('\n')
#     #sys.stdout.write(str("Teacher"))
#     #sys.stdout.write(str(inputs_teacher1))
#     #sys.stdout.write('\n')
#     sys.stdout.write(str("Labels"))
#     sys.stdout.write(str(labels1))
#     sys.stdout.write('\n')
#     sys.stdout.write('... ANOTHER BATCH ...')
#     sys.stdout.write('\n')
#     sys.stdout.write(str("NameInputs"))
#     sys.stdout.write(str(nameInpust2))
#     sys.stdout.write('\n')
#     #sys.stdout.write(str("Teacher"))
#     #sys.stdout.write(str(inputs_teacher2))
#     #sys.stdout.write('\n')
#     #sys.stdout.write(str("Student"))
#     #sys.stdout.write(str(inputs2))
#     #sys.stdout.write('\n')
#     sys.stdout.write(str("Labels"))
#     sys.stdout.write(str(labels2))


#     sys.stdout.write('\n')
#     sys.stdout.write('... Predicstions ...')
#     sys.stdout.write('\n')

#     pred1 = teacherModel(inputs_teacher1)
#     pred2 = teacherModel(inputs_teacher2)

#     _, pred_1 = torch.max(pred1.data, 1)  # get the index of the max log-probability
#     _, pred_2 = torch.max(pred2.data, 1)  # get the index of the max log-probability
#     correctNum1 = (pred_1 == labels1).sum().item()
#     correctNum2 = (pred_2 == labels2).sum().item()


#     sys.stdout.write(str("Pred1:"))
#     sys.stdout.write(str(pred_1))
#     sys.stdout.write(str(correctNum1))
#     sys.stdout.write('\n')
#     sys.stdout.write(str("Pred2:"))
#     sys.stdout.write(str(pred_2))
#     sys.stdout.write(str(correctNum2))
#     sys.stdout.write('\n')


############################################################ Trai&Val ################################################################
# Weight&Bias
wandb.login(key="71b7d0b39c786fe2a5cde2536f439a670d5b1a6e")
wandb.init(
    # set the wandb project where this run will be logged
    entity="duarte-lopes99",
    project='EmotionRecognition-Back2Start_Student&KDModels',
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size":batch_size,
    "optimizer":optimizer,
    "num_classes":7
    }
)
accValue, lossValue, val_accValues, val_lossValues, softLossValue, trainedModel =  train_val_KD (studentModel, teacherModel,train_loader, val_loader ,epochs, distillation_loss, T, alpha, optimizer)
## Saving Model 
PATH =  '/nas-ctm01/homes/pdcunha/KD_FER/KD_Models/KD_10_TM10_MobileV3_RAFD'
torch.save(trainedModel, PATH)
NameFileExcel = '/nas-ctm01/homes/pdcunha/KD_FER/KD_Models/KD_10_TM10_MobileV3_RAFD.xlsx'
send2Excel (accValue, val_accValues, lossValue, val_lossValues, softLossValue, NameFileExcel)

sys.stdout.write("[--- Model training: %s minutes ---]")
sys.stdout.write(((time.time() - start_time)/60))

sys.stdout.write("HyperParameters:")
sys.stdout.write('\n')
sys.stdout.write(T)
sys.stdout.write('\n')
sys.stdout.write(alpha)
sys.stdout.write('\n')
sys.stdout.write('modelName_')
    