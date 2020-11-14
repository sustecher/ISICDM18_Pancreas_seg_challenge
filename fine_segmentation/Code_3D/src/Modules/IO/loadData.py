import numpy as np
import pdb
# If you are not using nifti files you can comment this line
import nibabel as nib
import scipy.io as sio

from .ImgOperations.imgOp import applyPadding

# ----- Loader for nifti files ------ #
def load_nii (imageFileName, printFileNames) :
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()
    #imageData = (imageData-np.mean(imageData))/float(np.std(imageData))    
    return (imageData,img_proxy)
    
def release_nii_proxy(img_proxy) :
    img_proxy.uncache()


# ----- Loader for matlab format ------- #
# Very important: All the volumes should have been saved as 'vol'.
# Otherwise, change its name here
def load_matlab (imageFileName, printFileNames) :
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))
    
    mat_contents = sio.loadmat(imageFileName)
    imageData = mat_contents['vol']
    
    return (imageData)
    
""" It loads the images (CT/MRI + Ground Truth + ROI) for the patient image Idx"""
def load_imagesSinglePatient(imageIdx, 
                             imageNames, 
                             groundTruthNames, 
                             sampleSizes,
                             imageType
                             ):
    
    if imageIdx >= len(imageNames) :
        print (" ERROR!!!!! : The image index specified is greater than images array size....)")
        exit(1)
    
    # --- Load image data (CT/MRI/...) ---
    printFileNames = False # Get this from config.ini

    imageFileName = imageNames[imageIdx]

    if imageType == 0:
        [imageData,img_proxy] = load_nii(imageFileName, printFileNames)
    else:
        imageData = load_matlab(imageFileName, printFileNames)
        
    if len(imageData.shape) > 3 :
         imageData = imageData[:,:,:,0]
    
    if imageType == 0:
        release_nii_proxy(img_proxy)
    
    # --- Load ground truth (i.e. labels) ---
    if len(groundTruthNames) > 0 : 
        GTFileName = groundTruthNames[imageIdx]
        
        if imageType == 0:
            [gtLabelsData, gt_proxy] = load_nii (GTFileName, printFileNames)
        else:
            gtLabelsData = load_matlab(GTFileName, printFileNames)
        
        # Convert ground truth to int type
        if np.issubdtype( gtLabelsData.dtype, np.int ) : 
            gtLabelsData = gtLabelsData 
        else: 
            np.rint(gtLabelsData).astype("int32")
        
        imageGtLabels = gtLabelsData
        
        if imageType == 0:
            # Release data
            release_nii_proxy(gt_proxy)
        
    else : 
        imageGtLabels = np.empty(0)
        
    # --- Load roi ---

    return [imageData, imageGtLabels]


def loadSingleTransformedTemplateImage(imageIdx,
                                       imageNames,
                                       groundTruthNames,
                                       sampleSizes,
                                       receptiveField,
                                       applyPaddingBool,
                                       imageType
                                       ):
    if imageIdx >= len(imageNames) :
        print (" ERROR!!!!! : The image index specified is greater than images array size....)")
        exit(1)
    
    # --- Load image data (CT/MRI/...) ---
    printFileNames = False # Get this from config.ini

    imageFileName = imageNames[imageIdx]
    if imageType == 0:
        [imageData,img_proxy] = load_nii(imageFileName, printFileNames)
    else:
        imageData = load_matlab(imageFileName, printFileNames)
        
    if applyPaddingBool == True : 
        [imageData, paddingValues] = applyPadding(imageData, sampleSizes, receptiveField)
    else:
        paddingValues = ((0,0),(0,0),(0,0))


    if len(imageData.shape) > 3 :
         imageData = imageData[:,:,:,0]
    
    if imageType == 0:
        release_nii_proxy(img_proxy)
    
    # --- Load ground truth (i.e. labels) ---
    if len(groundTruthNames) > 0 : 
        GTFileName = groundTruthNames[imageIdx]
        
        if imageType == 0:
            [gtLabelsData, gt_proxy] = load_nii (GTFileName, printFileNames)
        else:
            gtLabelsData = load_matlab(GTFileName, printFileNames)
        
        # Convert ground truth to int type
        if np.issubdtype( gtLabelsData.dtype, np.int ) : 
            gtLabelsData = gtLabelsData 
        else: 
            np.rint(gtLabelsData).astype("int32")
        
        imageGtLabels = gtLabelsData
        
        if imageType == 0:
            # Release data
            release_nii_proxy(gt_proxy)
        
        if applyPaddingBool == True : 
            [imageGtLabels, paddingValues] = applyPadding(imageGtLabels,  sampleSizes, receptiveField)
        
    else : 
        imageGtLabels = np.empty(0)

    return [imageData, imageGtLabels]


def loadAllTransformedTemplateImage(imageNames, 
                                    groundTruthNames, 
                                    imageIndxes,
                                    sampleSizes,
                                    receptiveField
                                    ) :
    if len(imageIdx) >= len(imageNames) :
        print (" ERROR!!!!! : The image index specified is greater than images array size....)")
        exit(1)
    
    # --- Load image data (CT/MRI/...) ---
    printFileNames = False # Get this from config.ini
    for tem_i in xrange(len(imageIndxes)):
        if tem_i == 0:
            imageFileName = imageNames[imageIndxes[tem_i]]
            [imageData,img_proxy] = load_nii(imageFileName, printFileNames)

            if applyPaddingBool == True : 
                [imageData, paddingValues] = applyPadding(imageData, sampleSizes, receptiveField)
            else:
                paddingValues = ((0,0),(0,0),(0,0))
            release_nii_proxy(img_proxy)
            templateData = imageData
        else:
            imageFileName = imageNames[imageIndxes[tem_i]]
            [imageData,img_proxy] = load_nii(imageFileName, printFileNames)

            if applyPaddingBool == True : 
                [imageData, paddingValues] = applyPadding(imageData, sampleSizes, receptiveField)
            else:
                paddingValues = ((0,0),(0,0),(0,0))
            release_nii_proxy(img_proxy)
            templateData = np.concatenate((templateData, imageData[np.newaxis,:,:,:]),axis=0)


    for tem_i in xrange(len(imageIndxes)):
        if tem_i == 0:
            gtFileName = groundTruthNames[imageIndxes[tem_i]]
            [gtLabelsData,gt_proxy] = load_nii(gtFileName, printFileNames)

            if applyPaddingBool == True : 
                [gtLabelsData, paddingValues] = applyPadding(gtLabelsData, sampleSizes, receptiveField)
            else:
                paddingValues = ((0,0),(0,0),(0,0))
            release_nii_proxy(gt_proxy)
            templateLabelData = gtLabelsData
        else:
            gtFileName = groundTruthNames[imageIndxes[tem_i]]
            [gtLabelsData,gt_proxy] = load_nii(gtFileName, printFileNames)

            if applyPaddingBool == True : 
                [gtLabelsData, paddingValues] = applyPadding(gtLabelsData, sampleSizes, receptiveField)
            else:
                paddingValues = ((0,0),(0,0),(0,0))
            release_nii_proxy(gt_proxy)
            templateLabelData = np.concatenate((templateLabelData, gtLabelsData[np.newaxis,:,:,:]),axis=0)

    return [templateData, templateLabelData]


        
   
# -------------------------------------------------------- #
def getRandIndexes(total, maxNumberIdx) :
    # Generate a shuffle array of a vector containing "total" elements
    #idxs = range(total)
    idxs = np.arange(0,total)
    np.random.shuffle(idxs)
    rand_idxs = idxs[0:maxNumberIdx]
    return rand_idxs

