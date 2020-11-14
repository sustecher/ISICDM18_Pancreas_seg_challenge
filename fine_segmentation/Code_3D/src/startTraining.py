import sys
import time
import numpy as np
import random
import math
import os
                  
from Modules.General.Utils import getImagesSet
from Modules.IO.sampling import getSamplesSubepoch
from Modules.Parsers.parsersUtils import parserConfigIni

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from startTesting import segmentVolume, dirMake
import pdb
from MFCN_single import FCN, MyDataset
from MFCN_single import train, valid


def startTraining(model, configIniName, inIter):
    print (" ************************************************  STARTING TRAINING **************************************************")
    print (" **********************  Starting training model (Reading parameters) **********************")

    myParserConfigIni = parserConfigIni()
    myParserConfigIni.readConfigIniFile(configIniName,1)

    # Image type (0: Nifti, 1: Matlab)
    imageType = myParserConfigIni.imageTypesTrain

    print (" --- Do training in {} epochs with {} subEpochs each...".format(myParserConfigIni.numberOfEpochs, myParserConfigIni.numberOfSubEpochs))
    print ("-------- Reading Images names used in training/validation -------------")

    # -- Get list of images used for training -- #
    (imageNames_Train, names_Train)                = getImagesSet(myParserConfigIni.imagesFolder,myParserConfigIni.indexesForTraining)  # Images
    (groundTruthNames_Train, gt_names_Train)       = getImagesSet(myParserConfigIni.GroundTruthFolder,myParserConfigIni.indexesForTraining) # Ground truth
    
    # Print names
    print (" ================== Images for training ================")
    for i in range(0,len(names_Train)):
        print(" Image({}): {}  |  GT: {}  ".format(i,names_Train[i], gt_names_Train[i] ))

    numberOfEpochs = myParserConfigIni.numberOfEpochs
    numberOfSubEpochs = myParserConfigIni.numberOfSubEpochs
    numberOfSamplesSupEpoch  = myParserConfigIni.numberOfSamplesSupEpoch
    numberOfClass = myParserConfigIni.n_classes
    sampleSize_Train = myParserConfigIni.Patch_Size_Train
    batch_size = myParserConfigIni.batch_size
    folderName = myParserConfigIni.folderName
    timeForValidation = myParserConfigIni.timeForValidation

    # -- Get list of images used for validation -- #
    myParserConfigIni.readConfigIniFile(configIniName,2)
    sampleSize_Test  = myParserConfigIni.Patch_Size_Train
    strideValues = myParserConfigIni.strideValues

    (imageNames_Val, names_Val)          = getImagesSet(myParserConfigIni.imagesFolder,myParserConfigIni.indexesToSegment)  # Images
    (groundTruthNames_Val, gt_names_Val) = getImagesSet(myParserConfigIni.GroundTruthFolder,myParserConfigIni.indexesToSegment) # Ground truth

    # Print names
    print (" ================== Images for validation ================")
    for i in range(0,len(names_Val)):
        print(" Image({}): {}  |  GT: {}  ".format(i,names_Val[i], gt_names_Val[i]))
    
    print (" ===============================================================")
    
    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=1e-5)
    if torch.cuda.is_available():
        print('============================')
        print("Let's use", torch.cuda.device_count(),"GPUs!")
        net = torch.nn.DataParallel(model)
        net.cuda()
    
    AvDice_Val = 0
    
    for e_i in range(numberOfEpochs):
        # Recover last trained epoch                                 
        print(" ============== EPOCH: {}/{} =================".format(e_i, numberOfEpochs))
        costsOfEpoch = []
        for subE_i in range(numberOfSubEpochs): 
            #epoch_nr = subE_i+1
            print (" --- SubEPOCH: {}/{}".format(subE_i, numberOfSubEpochs))
            # Get all the samples that will be used in this sub-epoch

            [imagesSamplesAll,
            gt_samplesAll] = getSamplesSubepoch(numberOfSamplesSupEpoch,
                                                imageNames_Train,
                                                groundTruthNames_Train,
                                                imageType,
                                                sampleSize_Train
                                                )
            imagesSamplesAll = np.array(imagesSamplesAll)
            gt_samplesAll = np.array(gt_samplesAll)
            train_data = MyDataset(imagesSamplesAll, gt_samplesAll, transform = transforms.ToTensor())
            train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=8)
            

            train(net, train_loader, optimizer, numberOfClass, subE_i)
        
        Dice = []
        if e_i+inIter > timeForValidation:
            numberImagesToSegment = len(imageNames_Val)
            print(" ********************** Starting validation **********************")
            # Run over the images to segment  

            for i_d in range(numberImagesToSegment) :
                print("**********************  Segmenting subject: {} ....total: {}/{}...**********************".format(names_Val[i_d],str(i_d+1),str(numberImagesToSegment)))
        
                [everyROIDice, tmpDice] = segmentVolume(net,
                                                          folderName,
                                                          i_d,
                                                          imageNames_Val,
                                                          names_Val,
                                                          groundTruthNames_Val,
                                                          imageType,
                                                          sampleSize_Test,
                                                          strideValues,
                                                          numberOfClass,
                                                          batch_size,
                                                          0 # Validation (0) or testing (1)
                                                          )      
               
                Dice.append(tmpDice)
            print(" ********************** Validation DONE ********************** ")

            if sum(Dice)/len(Dice) >= AvDice_Val:
                AvDice_Val = sum(Dice)/len(Dice)
                continue
            else:
                break;
        #  --------------- Save the model --------------- 
        BASE_DIR = os.getcwd()
        path_Temp = os.path.join(BASE_DIR,'outputFiles')
        netFolderName = os.path.join(path_Temp,folderName)
        netFolderName  = os.path.join(netFolderName,'Networks')
        dirMake(netFolderName)

        modelFileName = netFolderName + "/FCN_Epoch" + str(e_i+inIter+1)
        torch.save(net,modelFileName)
 
        strFinal =  " Network model saved in " + netFolderName + " as FCN_Epoch" + str (e_i+inIter+1)
        print(strFinal)

    print("................ The whole Training is done..............")
    print(" ************************************************************************************ ")


            


            


