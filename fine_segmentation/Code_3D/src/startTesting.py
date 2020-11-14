
import numpy as np
import time
import os
import pdb

from Modules.General.Evaluation import computeDice
from Modules.General.Utils import getImagesSet
from Modules.IO.ImgOperations.imgOp import applyUnpadding
from Modules.IO.loadData import load_imagesSinglePatient
from Modules.IO.saveData import saveImageAsNifti
from Modules.IO.saveData import saveImageAsMatlab
from Modules.IO.sampling import *
from Modules.Parsers.parsersUtils import parserConfigIni

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from MFCN_single import FCN, MyDatasetTest
from MFCN_single import test

def segmentVolume(myNetworkModel,
                  folderName,
                  i_d,
                  imageNames_Test,
                  names_Test,
                  groundTruthNames_Test,
                  imageType,
                  sampleSize_Test,
                  strideVal,
                  n_classes,
                  batch_Size,
                  task, # Validation (0) or testing (1)
                  ):
        
        # Load the images to segment
        [imgSubject,  
        gtLabelsImage] = load_imagesSinglePatient(i_d,
                                                  imageNames_Test,
                                                  groundTruthNames_Test,
                                                  sampleSize_Test,
                                                  imageType
                                                  )
                                                  
                                  
        # Get image dimensions                                                    
        imgDims = list(imgSubject.shape)
                   
        [ sampleCoords ] = sampleWholeImage(imgSubject,
                                            sampleSize_Test,
                                            strideVal,
                                            batch_Size
                                            )        
        numberOfSamples = len(sampleCoords)
        #print('numberOfSamples:',numberOfSamples)
        sampleID = 0
        numberOfBatches = int(numberOfSamples/batch_Size)
        #print(numberOfSamples,batch_Size)
        #The probability-map that will be constructed by the predictions.
        probMaps = np.zeros([n_classes]+imgDims, dtype = "float32")
        countMaps = np.zeros([n_classes]+imgDims, dtype = "float32")
        # Run over all the batches 
        
        for b_i in range(numberOfBatches) :
                 
            # Get samples for batch b_i
            
            sampleCoords_b = sampleCoords[ b_i*batch_Size : (b_i+1)*batch_Size ]
            

            [imgSamples] = extractSamples(imgSubject,
                                          sampleCoords_b,
                                          sampleSize_Test)

            # Load the data of the batch on the GPU
            imgSamples = np.array(imgSamples)
            test_data = MyDatasetTest(imgSamples,transform=transforms.ToTensor())
            test_loader = DataLoader(test_data, batch_Size, shuffle=False, num_workers=8)
            predictions = test(myNetworkModel, test_loader)

            # --- Now we can generate the probability maps from the predictions ----
            # Run over all the regions
            
            tmpPredMap = np.zeros([n_classes]+imgDims, dtype = "float32")
            tmpCountMap = np.zeros([n_classes]+imgDims, dtype = "float32")
            for r_i in range(batch_Size) :
                sampleCoords_i = sampleCoords[sampleID]
                coords = [ sampleCoords_i[0][0], sampleCoords_i[1][0], sampleCoords_i[2][0] ]

                # Get the min and max coords
                xMin = coords[0]
                xMax = coords[0] + sampleSize_Test[0]

                yMin = coords[1]
                yMax = coords[1] + sampleSize_Test[1]

                zMin = coords[2] 
                zMax = coords[2] + sampleSize_Test[2]
                
                tmpPredMap[:,xMin:xMax, yMin:yMax, zMin:zMax] = predictions[r_i]
                tmpCountMap[:,xMin:xMax, yMin:yMax, zMin:zMax] = 1
                probMaps += tmpPredMap 
                countMaps += tmpCountMap
                #tmpPredMap[:,xMin:xMax, yMin:yMax, zMin:zMax] = predictions[r_i]
                #probMaps += tmpPredMap 
        
                sampleID += 1
            
        # Now: Save the data
        # Get the segmentation from the probability maps ---
        #probMaps /= sampleID
        #segmentationImage = np.argmax(probMaps, axis=0) 
        addition = np.double(np.logical_not(countMaps!=0))
        countMaps = countMaps + addition
        probMaps /= countMaps
        segmentationImage = np.argmax(probMaps, axis=0) 
        #Save Result:
        npDtypeForPredictedImage = np.dtype(np.int16)
        suffixToAdd = "_Segm"
 
        segmentationRes = segmentationImage

        # Generate folders to store the model
        BASE_DIR = os.getcwd()
        path_Temp = os.path.join(BASE_DIR,'outputFiles')

        # For the predictions
        predlFolderName = os.path.join(path_Temp,folderName)
        predlFolderName = os.path.join(predlFolderName,'Pred')
        if task == 0:
            predTestFolderName = os.path.join(predlFolderName,'Validation')
        else:
            predTestFolderName = os.path.join(predlFolderName,'Testing')
        
        dirMake(predTestFolderName)
        nameToSave = predTestFolderName + '/Segmentation_'+ names_Test[i_d]
        
        # Save Segmentation image
        
        print(" ... Saving segmentation result..."),
        if imageType == 0: # nifti
            imageTypeToSave = np.dtype(np.int16)
            saveImageAsNifti(segmentationRes,
                             nameToSave,
                             imageNames_Test[i_d],
                             imageTypeToSave)
        else: # Matlab
            # Cast to int8 for saving purposes
            saveImageAsMatlab(segmentationRes.astype('int8'),
                              nameToSave)


        # Save the prob maps for each class (except background)
        for c_i in range(1, n_classes) :
            
            
            nameToSave = predTestFolderName + '/ProbMap_class_'+ str(c_i) + '_' + names_Test[i_d] 

            probMapClass = probMaps[c_i,:,:,:]

            probMapClassRes = probMapClass

            print(" ... Saving prob map for class {}...".format(str(c_i))),
            if imageType == 0: # nifti
                imageTypeToSave = np.dtype(np.float32)
                saveImageAsNifti(probMapClassRes,
                                 nameToSave,
                                 imageNames_Test[i_d],
                                 imageTypeToSave)
            else:
                # Cast to float32 for saving purposes
                saveImageAsMatlab(probMapClassRes.astype('float32'),
                                  nameToSave)

        # If segmentation done during evaluation, get dice
        if task == 0:
            print(" ... Computing Dice scores: ")
            [DiceArray, DiceAll] = computeDice(segmentationImage,gtLabelsImage)
            for d_i in range(len(DiceArray)):
                print(" ------DSC (Class {}) : {}".format(str(d_i+1),DiceArray[d_i]))
            print("------DSC (All ROIs) : {}".format(DiceAll))
        
        if task == 1:
            print(" ... Computing Dice scores: ")
            [DiceArray, DiceAll] = computeDice(segmentationImage,gtLabelsImage)
            for d_i in range(len(DiceArray)):
                print(" -------------- DSC (Class {}) : {}".format(str(d_i+1),DiceArray[d_i]))
            print("------DSC (All ROIs) : {}".format(DiceAll))

        return DiceArray, DiceAll

""" Main segmentation function """
def startTesting(FCNname,configIniName) :

    #padInputImagesBool = True # from config ini
    print (" ******************************************  STARTING SEGMENTATION ******************************************")

    print (" **********************  Starting segmentation **********************")
    myParserConfigIni = parserConfigIni()
    myParserConfigIni.readConfigIniFile(configIniName,2)
    

    print (" -------- Images to segment -------------")

    print (" -------- Reading Images names for segmentation -------------")
    
    # -- Get list of images used for testing -- #
    (imageNames_Test, names_Test) = getImagesSet(myParserConfigIni.imagesFolder,myParserConfigIni.indexesToSegment)  # Images
    (groundTruthNames_Test, gt_names_Test) = getImagesSet(myParserConfigIni.GroundTruthFolder,myParserConfigIni.indexesToSegment) # Ground truth

    print (" ================== Images for training ================")
    for i in range(0,len(names_Test)):
        print(" Image({}): {}  |  GT: {}  ".format(i,names_Test[i], gt_names_Test[i] ))

    folderName            = myParserConfigIni.folderName
    batch_size            = myParserConfigIni.batch_size
    sampleSize_Test       = myParserConfigIni.Patch_Size_Test
    imageType             = myParserConfigIni.imageTypes
    strideValues          = myParserConfigIni.strideValues
    numberOfClass         = myParserConfigIni.n_classes
    numberImagesToSegment = len(imageNames_Test)

    # --------------- Load my FCN object  --------------- 
    print (" ... Loading model from {}".format(FCNname))
    model = torch.load(FCNname)
 
    print (" ... Network architecture successfully loaded....")

    for i_d in range(numberImagesToSegment) :
        print("**********************  Segmenting subject: {} ....total: {}/{}...**********************".format(names_Test[i_d],str(i_d+1),str(numberImagesToSegment)))
        
        segmentVolume(model,
                      folderName,
                      i_d,
                      imageNames_Test,
                      names_Test,
                      groundTruthNames_Test,
                      imageType,
                      sampleSize_Test,
                      strideValues,
                      numberOfClass,
                      batch_size,
                      1 # Validation (0) or testing (1)
                      )      
       
    print(" **************************************************************************************************** ")


def dirMake(dirPath):
    if dirPath != "":
        if not os.path.exists(dirPath): os.makedirs(dirPath)
        return os.path.normpath(dirPath) + "/"
    else:
        return dirPath
