from .loadData import load_imagesSinglePatient
from .loadData import getRandIndexes
import numpy as np
import math
import random


# **********************************  For Training *********************************
""" This function gets all the samples needed for training a sub-epoch """
def getSamplesSubepoch(numSamples,
                       imageNames,                                                                
                       groundTruthNames,
                       imageType,
                       sampleSizes
                       ):
    print (" ... Get samples for subEpoch...")
    
    numSubjects_Epoch = len(imageNames)
    randIdx = getRandIndexes(numSubjects_Epoch, numSubjects_Epoch)

    samplesPerSubject = numSamples/len(randIdx)
    print (" ... getting {} samples per subject...".format(samplesPerSubject)) 
    
    imagesSamplesAll = [] 
    gt_samplesAll = [] 
    
    numSubjectsSubEpoch = len(randIdx) 
    
    samplingDistribution = getSamplesDistribution(numSamples, numSubjectsSubEpoch)

    for i_d in range(0, numSubjectsSubEpoch) :
        # For displaying purposes
        perc = 100 * float(i_d+1)/numSubjectsSubEpoch
        print("...Processing subject: {}. {} % of the whole training set...".format(str(i_d + 1),perc))

        # -- Load images for a given patient --
        [imgSubject, 
         gtLabelsImage]= load_imagesSinglePatient(randIdx[i_d],
                                                   imageNames,
                                                   groundTruthNames,
                                                   sampleSizes,
                                                   imageType
                                                   )
                                                  

        # -- Get samples for that patient
        [imagesSamplesSinglePatient,
         gtSamplesSinglePatient] = getSamplesSubject(i_d,
                                                     imgSubject,
                                                     gtLabelsImage,
                                                     samplingDistribution,
                                                     sampleSizes,
                                                     )

        imagesSamplesAll = imagesSamplesAll + imagesSamplesSinglePatient
        gt_samplesAll = gt_samplesAll + gtSamplesSinglePatient
 
    return imagesSamplesAll, gt_samplesAll


def getSamplesSubject(imageIdx,
                      imgSubject,
                      gtLabelsImage,
                      samplingDistribution,
                      sampleSizes,
                      ):
    sampleSizes = sampleSizes
    imageSamplesSingleImage = []
    gt_samplesSingleImage = []
            
    imgDim = imgSubject.shape

    # Get weight maps for sampling
    weightMaps = getSamplingWeights(gtLabelsImage)


    # We are extracting segments for foreground and background
    for c_i in range(2) :
        numOfSamplesToGet = samplingDistribution[c_i][imageIdx]
        weightMap = weightMaps[c_i]
        # Define variables to be used
        roiToApply = np.zeros(weightMap.shape, dtype="int32")
        halfSampleDim = np.zeros( (len(sampleSizes), 2) , dtype='int32')

        # Get the size of half patch (i.e sample)
        for i in range( len(sampleSizes) ) :
            if sampleSizes[i]%2 == 0: #even
                dimensionDividedByTwo = sampleSizes[i]/2
                halfSampleDim[i] = [dimensionDividedByTwo - 1, dimensionDividedByTwo] 
            else: #odd
                dimensionDividedByTwoFloor = math.floor(sampleSizes[i]/2) 
                halfSampleDim[i] = [dimensionDividedByTwoFloor, dimensionDividedByTwoFloor] 
    
        # --- Set to 1 those voxels in which we are interested in
        # - Define the limits
        
        roiMinx = halfSampleDim[0][0]
        roiMaxx = imgDim[0] - halfSampleDim[0][1]
        roiMiny = halfSampleDim[1][0]
        roiMaxy = imgDim[1] - halfSampleDim[1][1]
        roiMinz = halfSampleDim[2][0]
        roiMaxz = imgDim[2] - halfSampleDim[2][1]

        #print(roiMinx,roiMaxx,roiMiny,roiMaxy,roiMinz,roiMaxz)
        # Set
        roiToApply[roiMinx:roiMaxx,roiMiny:roiMaxy,roiMinz:roiMaxz] = 1
        
        maskCoords = weightMap * roiToApply
        
        # We do the following because np.random.choice 4th parameter needs the probabilities to sum 1
        maskCoords = maskCoords / (1.0* np.sum(maskCoords))
    
        maskCoordsFlattened = maskCoords.flatten()
  
        centralVoxelsIndexes = np.random.choice(maskCoords.size,
                                                size = numOfSamplesToGet,
                                                replace=True,
                                                p=maskCoordsFlattened)

        centralVoxelsCoord = np.asarray(np.unravel_index(centralVoxelsIndexes, maskCoords.shape))
        
        coordsToSampleArray = np.zeros(list(centralVoxelsCoord.shape) + [2], dtype="int32")
        coordsToSampleArray[:,:,0] = centralVoxelsCoord - halfSampleDim[ :, np.newaxis, 0 ] #np.newaxis broadcasts. To broadcast the -+.
        coordsToSampleArray[:,:,1] = centralVoxelsCoord + halfSampleDim[ :, np.newaxis, 1 ]

        
        # ----- Compute the coordinates that will be used to extract the samples ---- #
        numSamples = len(coordsToSampleArray[0])

        # Extract samples from computed coordinates
        for s_i in range(numSamples) :

            # Get one sample given a coordinate
            coordsToSample = coordsToSampleArray[:,s_i,:] 

            sampleSizes = sampleSizes
            imageSample = np.zeros((1, sampleSizes[0],sampleSizes[1],sampleSizes[2]), dtype = 'float32')
            sample_gt_Orig = np.zeros((1, sampleSizes[0],sampleSizes[1],sampleSizes[2]), dtype = 'int16')

            xMin = coordsToSample[0][0]
            xMax = coordsToSample[0][1] + 1
            yMin = coordsToSample[1][0]
            yMax = coordsToSample[1][1] + 1
            zMin = coordsToSample[2][0]
            zMax = coordsToSample[2][1] + 1
            xMin, xMax = Avoid_Out_of_range(xMin,xMax)
            yMin, yMax = Avoid_Out_of_range(yMin, yMax)
            zMin, zMax = Avoid_Out_of_range(zMin, zMax)

            #print(imgSubject.shape)
            imageSample[:1] = imgSubject[ xMin:xMax,yMin:yMax,zMin:zMax]
            sample_gt_Orig[:1] = gtLabelsImage[xMin:xMax,yMin:yMax,zMin:zMax]

            roiLabelMin = np.zeros(3, dtype = "int16")
            roiLabelMax = np.zeros(3, dtype = "int16")

            for i_x in range(3) :
                roiLabelMax[i_x] = sampleSizes[i_x] - roiLabelMin[i_x]

            gt_sample = sample_gt_Orig[roiLabelMin[0] : roiLabelMax[0],
                                       roiLabelMin[1] : roiLabelMax[1],
                                       roiLabelMin[2] : roiLabelMax[2]]
                                        
            imageSamplesSingleImage.append(imageSample)
            gt_samplesSingleImage.append(gt_sample)

    return imageSamplesSingleImage,gt_samplesSingleImage
   
def Avoid_Out_of_range(xMin,xMax):
    if xMin<0:
        print(xMax, xMin)
        xMax=xMax-xMin
        xMin=0
        print(xMax,xMin)
    else:
        XMin=xMin

    return xMin,xMax

def getSamplingWeights(gtLabelsImage) :

    foreMask = (gtLabelsImage>0).astype(int)
    roiMask = np.ones(np.shape(gtLabelsImage))
    backMask = (roiMask>0) * (foreMask==0)
    weightMaps = [ foreMask, backMask ] 
 
    return weightMaps
     


def getSamplesDistribution( numSamples,
                            numImagesToSample ) :
    # We have to sample foreground and background
    # Assuming that we extract the same number of samples per category: 50% each

    samplesPercentage = np.ones( 2, dtype="float32" ) * 0.5
    samplesPerClass = np.zeros( 2, dtype="int32" )
    samplesDistribution = np.zeros( [ 2, numImagesToSample ] , dtype="int32" )
    
    samplesAssigned = 0
    
    for c_i in range(2) :
        samplesAssignedClass = int(numSamples*samplesPercentage[c_i])
        samplesPerClass[c_i] += samplesAssignedClass
        samplesAssigned += samplesAssignedClass
 
    # Assign the samples that were not assigned due to the rounding error of integer division. 
    nonAssignedSamples = numSamples - samplesAssigned
    classesIDx= np.random.choice(2,
                                 nonAssignedSamples,
                                 True,
                                 p=samplesPercentage)

    for c_i in classesIDx : 
        samplesPerClass[c_i] += 1
        
    for c_i in range(2) :
        samplesAssignedClass = int(samplesPerClass[c_i] / numImagesToSample)  
        samplesDistribution[c_i] += samplesAssignedClass
        samplesNonAssignedClass = samplesPerClass[c_i] % numImagesToSample
        for cU_i in range(samplesNonAssignedClass):
            samplesDistribution[c_i, random.randint(0, numImagesToSample-1)] += 1

    return samplesDistribution

# **********************************  For testing *********************************

def sampleWholeImage(imgSubject,
                     sampleSize,
                     strideVal,
                     batch_size
                     ):

    samplesCoords = []
 
    imgDims = list(imgSubject.shape)
    
    zMinNext=0
    zCentPredicted = False
    
    xmin = 0
    xmax = imgDims[0]
    ymin = 0
    ymax = imgDims[1]
    zmin = 0
    zmax = imgDims[2]
    num_x = int((xmax-xmin-sampleSize[0])/strideVal[0]+1)
    num_y = int((ymax-ymin-sampleSize[1])/strideVal[1]+1)
    num_z = int((zmax-zmin-sampleSize[2])/strideVal[2]+1)

    zMinNext = zmin
    for i_z in range(0,num_z) :
        zMax = min(zMinNext+sampleSize[2], imgDims[2]) 
        zMin = zMax - sampleSize[2]
        zMinNext = zMinNext + strideVal[2]
        yMinNext=ymin
        yCentPredicted = False
        
        for i_y in range(0,num_y) :
            yMax = min(yMinNext+sampleSize[1], imgDims[1]) 
            yMin = yMax - sampleSize[1]
            yMinNext = yMinNext + strideVal[1]
            xMinNext=xmin
            xCentPredicted = False
            
            for i_x in range(0,num_x) :
                xMax = min(xMinNext+sampleSize[0], imgDims[0])
                xMin = xMax - sampleSize[0]
                xMinNext = xMinNext + strideVal[0]
          
                samplesCoords.append([ [xMin, xMax-1], [yMin, yMax-1], [zMin, zMax-1] ])

    sampledRegions = len(samplesCoords)

    if sampledRegions%batch_size != 0:
        numberOfSamplesToAdd =  batch_size - sampledRegions%batch_size
    else:
      numberOfSamplesToAdd = 0
      
    for i in range(numberOfSamplesToAdd) :
        samplesCoords.append(samplesCoords[sampledRegions-1])

    return [samplesCoords]


def extractSamples(imgData,
                   sliceCoords,
                   imagePartDimensions
                   ) :
    numberOfSamples = len(sliceCoords)
    # Create the array that will contain the samples
    samplesArrayShape = [numberOfSamples, 1, imagePartDimensions[0], imagePartDimensions[1], imagePartDimensions[2]]
    samples = np.zeros(samplesArrayShape, dtype= "float32")
    
    for s_i in range(numberOfSamples) :
        cMin = []
        cMax = []
        for c_i in range(3):
            cMin.append(sliceCoords[s_i][c_i][0])
            cMax.append(sliceCoords[s_i][c_i][1] + 1)
            
        samples[s_i] = imgData[cMin[0]:cMax[0],
                               cMin[1]:cMax[1],
                               cMin[2]:cMax[2]]
                                                                    
    return [samples]
