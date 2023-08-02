###################### TEACHER MODEL VSCode #########################################
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader
import pandas as pd
import sys 
import os
import csv
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import wandb
import time

from losses import TripletLoss, LabelSmoothingCrossEntropy

from torch.nn.functional import cross_entropy

start_time = time.time()
############################### DATA #############################################
# Class Necessary to transform my data Array b-----> Tensor 
class CustomDataSet (Dataset):
    def __init__(self, csvFile, dirImg, typeImg, transform=None):
        self.data_frame = pd.read_csv(csvFile)
        self.dirImg = dirImg
        self.transform = transform
        self.typeImg = typeImg

        self.class_count = self.data_frame.iloc[:, 3].value_counts().sort_index().tolist()

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
    def get_class_count(self):
        return self.class_count
    def get_class_weights(self):
        labels = self.data_frame.iloc[:, 3].tolist()
        class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
        return class_weights
    
# class CustomDataSet (Dataset):
#     def __init__(self, csvFile, dirImg, typeImg, transform=None):
#         self.data_frame = pd.read_csv(csvFile)
#         self.dirImg = dirImg
#         self.transform = transform
#         self.typeImg = typeImg

#         self.class_count = self.data_frame.iloc[:, 2].value_counts().sort_index().tolist()

#     def __len__(self):
#         return len(self.data_frame)
#     def __getitem__(self, idx):
#         if self.typeImg == 'Orign':
#             img_path = os.path.join(self.dirImg, self.data_frame.iloc[idx, 1]) 
#         elif self.typeImg == 'Sh':
#             img_path = os.path.join(self.dirImg, self.data_frame.iloc[idx, 2]) 
#         else:
#             raise ValueError("ImgType must be either 'Orign' or 'Sh'")

#         img_name = self.data_frame.iloc[idx, 0]
#         image = Image.open(img_path)
#         label = self.data_frame.iloc[idx, 2]
#         if self.transform:
#             image = self.transform(image)
#         return img_name, image, label
#     def get_class_count(self):
#         return self.class_count
#     def get_class_weights(self):
#         labels = self.data_frame.iloc[:, 2].tolist()
#         class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y = labels)
#         return class_weights

transform = transforms.Compose([
    #torchvision.transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])

transform_W_Augm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((112, 112)),
    transforms.RandomApply([
        transforms.RandomRotation(20),
        transforms.RandomCrop(112, padding=32)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])

data = "CK+"
# RAFD
if data == "RAFD":
    pathCSV = '/nas-ctm01/datasets/public/DB/FER_/RAFD_ALL_Train.csv'
    pathFolderOrign = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Align/Train'
    #pathFolderSh = '/nas-ctm01/datasets/public/DB/FER_/RAFD_Sh/Train'
# CK+
if data == "CK+":
    pathCSV = '/nas-ctm01/datasets/public/DB/FER_/CK+_Train.csv'
    pathFolderOrign = '/nas-ctm01/datasets/public/DB/FER_/CK+_Align/Train'
    #pathFolderSh = '/nas-ctm01/datasets/public/DB/FER_/CK+_Sh/Train'


OrignDataset = CustomDataSet(csvFile = pathCSV, dirImg = pathFolderOrign, typeImg='Orign',transform=transform_W_Augm)
#OrignDataset = CustomDataSet(csvFile = pathCSV, dirImg = pathFolderOrign, typeImg='Orign',transform=transform)
#OrignDataset = CustomDataSet(csvFile = pathCSV, dirImg = pathFolderSh, typeImg='Sh',transform=transform)

# Split the dataset into training and validation sets
train_idx, val_idx = train_test_split(range(len(OrignDataset)), test_size=0.2, random_state=42)

OrignDataset_Train = Subset(OrignDataset, train_idx)
OrignDataset_Val = Subset(OrignDataset, val_idx)
# DataLoader
train_loader = DataLoader(OrignDataset_Train, batch_size=32, shuffle=True)
val_loader = DataLoader(OrignDataset_Val, batch_size=32)

modelName_ = 'Model_11-Orign_CK+_Frozen_60_IrisNet100 - LSCE'
############################ Train & Val. ################################
# Train the model
def train_and_validate (model, trainData, valData, num_epochs, lossFunction, optimizer,criterion):

    train_accValues = []
    train_lossValues = []
    val_accValues = []
    val_lossValues = []

    model.train()

    for epoch in range(num_epochs):
        sys.stdout.write("[Epoch:]")
        sys.stdout.write(str(epoch+1))
        sys.stdout.write('\n')

        lossEpoch = 0.0

        preds = []
        samples = []

        correct = 0
        total = 0

        val_correct = 0
        val_total = 0

        ########### TRAINING DATA ######################
        for i, (nameInputs, inputs, labels) in enumerate(trainData):
            # calling zero_grad() at the start of each iteration ensures that the gradients are not accumulated from previous iterations
            optimizer.zero_grad()

            # The output will be a tensor of size (batch_size, num_classes), where num_classes is the number of classes in your classification problem
            outputs = model(inputs)

            # This line calculates the loss between the predicted outputs and the ground truth labels.
            #loss = lossFunction(outputs, labels)

            # Loss Label-Smoothed Cross-Entropy Loss
            CE_loss = lossFunction(outputs, labels)
            lsce_loss = criterion(outputs, labels)
            loss = 2 * lsce_loss + CE_loss

            loss.backward()
            optimizer.step()

            wandb.log({"epoch": epoch, "batchNum": i+1, "BatchLoss": loss.item()})
            lossEpoch += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            preds.append(predicted)
            samples.append(labels)

        # Calculate training accuracy and loss for each epoch
        accuracy = 100 * correct / total
        train_accValues.append(accuracy)
        train_lossValues.append(lossEpoch)
        avgLoss = lossEpoch/len(trainData)
        sys.stdout.write("Train Total Correct =>")
        sys.stdout.write(str(correct))
        sys.stdout.write(' ')
        sys.stdout.write("Train Total=>")
        sys.stdout.write(str(total))
        sys.stdout.write('\n')
        # Log training metrics
        wandb.log({"epoch": epoch, "train_loss": lossEpoch, "train_acc": accuracy, "AvgLoss":avgLoss})
        ########### VALIDATION DATA ####################
        val_loss = 0.0
        model.eval()    

        if epoch%9 == 0:
            path = '/nas-ctm01/homes/pdcunha/KD_FER/ExcelFIles/'+ modelName_ +'/FileEpoch'+str(epoch+1)+'.csv'
            with open(path, "w", newline="") as csv_file:
                csvHead =["batch","logits","pred","label","ValLoss"]
                writer = csv.writer(csv_file)
                writer.writerow(csvHead)
    
        with torch.no_grad():
            for i, (nameInputs, inputs, labels) in enumerate(valData):

                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, labels)
                #loss = lossFunction(outputs,labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                if epoch%9 == 0:
                    path = '/nas-ctm01/homes/pdcunha/KD_FER/ExcelFIles/'+ modelName_ +'/FileEpoch'+str(epoch+1)+'.csv'
                    with open(path, "a", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([str(i+1), str(outputs) ,str(predicted), str(labels),str(val_loss)])
                # LOG W&B TABLE 
                #data = [(i+1), outputs,labels]
                #table = wandb.Table(columns=["Batch", "LogitOut", "True Label"])
                #table.add_data(str((i+1)), str(outputs),str(labels))
                #wandb.log({"Validation Pred Table": table})

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation accuracy and loss for each epoch
        sys.stdout.write("Val Total Correct =>")
        sys.stdout.write(str(val_correct))
        sys.stdout.write(' ')
        sys.stdout.write("Val Total=>")
        sys.stdout.write(str(val_total))
        sys.stdout.write('\n')

        val_accuracy = 100 * val_correct / val_total
        val_accValues.append(val_accuracy)
        val_lossValues.append(val_loss)
        avgValLoss = val_loss/len(valData)

        # Log validation metrics
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_acc": val_accuracy, "AvgLossVal":avgValLoss})

    return train_accValues, train_lossValues, val_accValues, val_lossValues, model

def send2Excel (array1, array2, array3, array4, NameFileExcel):
    panda1 = pd.DataFrame (array1)
    panda2 = pd.DataFrame (array2)
    panda3 = pd.DataFrame (array3)
    panda4 = pd.DataFrame (array4)
    
    with pd.ExcelWriter(NameFileExcel) as writer:
        panda1.to_excel(writer, sheet_name= "TrainAcc")
        panda2.to_excel(writer, sheet_name= "Val_Acc")
        panda3.to_excel(writer, sheet_name= "TrainLoss")
        panda4.to_excel(writer, sheet_name= "Val_Loss")
    return 'Done To Excel'

################### Train with ResNet 34 #################################
resnetModel = models.resnet34(weights = "ResNet34_Weights.IMAGENET1K_V1")
num_ftrs = resnetModel.fc.in_features
resnetModel.fc = nn.Linear(num_ftrs, 7)
## (?) resnetModel.add_module('softmax', nn.Softmax(dim=1)) ## (?)
#teacherModel = resnetModel

################### ResNet 50 #################################
resNetModel50 = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
num_ftrs = resNetModel50.fc.in_features
resNetModel50.fc = nn.Linear(num_ftrs, 7)
teacherModel = resNetModel50

### Wide ResNet 50 V2
wideResNetModel50 = models.wide_resnet50_2(weights="Wide_ResNet50_2_Weights.IMAGENET1K_V2")
num_ftrs = wideResNetModel50.fc.in_features
wideResNetModel50.fc = nn.Linear(num_ftrs, 7)
#teacherModel = wideResNetModel50

# Student Model 
mobileNetModel = models.mobilenet_v2(weights = "MobileNet_V2_Weights.IMAGENET1K_V2")
num_ftrs = mobileNetModel.classifier[-1].in_features
mobileNetModel.classifier[-1]  = nn.Linear(num_ftrs, 7)
#studentModel = mobileNetModel

# ResNet18
resNet18 = models.resnet18(weights = 'IMAGENET1K_V1')
num_ftrs = resNet18.fc.in_features 
resNet18.fc = nn.Linear(num_ftrs, 7)
#studentModel = resNet18

# IrisNet - FaceRecogniton Trained Model
from elasticFace.backbones.iresnet import iresnet100, iresnet50
from elasticFace.config.config import config as cfg
backbone = iresnet100(num_features=cfg.embedding_size)
backbone.load_state_dict(torch.load('/nas-ctm01/homes/pdcunha/KD_FER/elasticFace/modelWeights/ElasticFaceArc.pth'))
num_ftrs = backbone.fc.in_features
backbone.fc = nn.Linear(num_ftrs, 7)
backbone.features = nn.BatchNorm1d(7)
#model = backbone

# FREEZE LAYERS 
def freeze_layers(model, percentage_to_freeze):
    newModel = model
    total_layers = len(list(newModel.parameters()))
    layers_to_freeze = int(total_layers * percentage_to_freeze)

    count = 0
    for name, param in newModel.named_parameters():
        if count < layers_to_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
        count += 1
    return  newModel


freeze_percentage = 0.6  # Percentage of layers to freeze (90%)
freezeModel = freeze_layers(backbone, freeze_percentage)

def print_model_layers(model):
     # Verify the parameters of the frozen layers
    for name, param in model.named_parameters():
        sys.stdout.write(f"Layer: {name}, Trainable: {param.requires_grad}")
        sys.stdout.write('\n')

#print_model_layers(freezeModel)

#sys.stdout.write(str(model))
#sys.stdout.write('\n')
#sys.stdout.write(str(model.state_dict()['fc.weight'].shape))
#sys.stdout.write('\n')
#sys.stdout.write(str(model.state_dict()['fc.bias'].shape))
#################### Student or Teacher ? ################################
#model = teacherModel
model = freezeModel
#model = studentModel    f

# Hyperparameters
lr = 1e-5
#lr = 1e-4

#lossFunction = torch.nn.CrossEntropyLoss()
#classWeights = OrignDataset_Train.dataset.get_class_weights()
#weightsTensor = torch.from_numpy(classWeights).float()
#lossFunction = nn.CrossEntropyLoss(weight=weightsTensor)

lossFunction = nn.CrossEntropyLoss()

# Loss labels Smoothing - AntiOverfitting
criterion = LabelSmoothingCrossEntropy(smoothing=0.2)


num_epochs = 150
batch_size = 32
weight_decay = 1e-5
beta1 = 0.9
beta2 = 0.999

#optimizer = optim.Adam(model.parameters(), lr = lr)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

########################### Weight&Bias #############################################################
wandb.login(key="71b7d0b39c786fe2a5cde2536f439a670d5b1a6e")
wandb.init(
    # set the wandb project where this run will be logged
    entity="duarte-lopes99",
    project='EmotionRecognition-Back2Start_TeacherModels',
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "epochs": num_epochs,
    "batch_size":batch_size,
    "optimizer":optimizer,
    "num_classes":7
    }
)

##### TRAIN____&___VAL
accValue, lossValue, val_accValues, val_lossValues, trainedModel =  train_and_validate (model, train_loader, val_loader,num_epochs, lossFunction, optimizer,criterion)

# Saving Model 
PATH =  '/nas-ctm01/homes/pdcunha/KD_FER/TeacherModelStats_Back2Start/Model_11-Orign_CK+_Frozen_60_IrisNet100 - LSCE'
torch.save(trainedModel, PATH)
NameFileExcel = '/nas-ctm01/homes/pdcunha/KD_FER/TeacherModelStats_Back2Start/Model_11-Orign_CK+_Frozen_60_IrisNet100 - LSCE.xlsx'
send2Excel (accValue, val_accValues, lossValue, val_lossValues, NameFileExcel)

sys.stdout.write("[--- Model training:]")
sys.stdout.write('\n')
sys.stdout.write('TeacherModel - Orign_CK+_Frozen_60_IrisNet10')
sys.stdout.write('\n')
sys.stdout.write('Lr')
sys.stdout.write(str(lr))
sys.stdout.write('\n')
sys.stdout.write('Loss')
sys.stdout.write('Weighted_CE')
sys.stdout.write('\n')
sys.stdout.write('Optim')
sys.stdout.write('Adam')
sys.stdout.write('\n')