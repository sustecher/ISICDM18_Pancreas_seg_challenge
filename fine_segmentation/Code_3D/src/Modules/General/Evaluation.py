import pdb
import numpy as np

# ----- Dice Score -----
def computeDice(autoSeg, groundTruth):
    """ Returns
    -------
    DiceArray : floats array
          
          Dice coefficient as a float on range [0,1].
          Maximum similarity = 1
          No similarity = 0 """
          
    n_classes = int( np.max(groundTruth) + 1)
   
    DiceArray = []
    
    
    for c_i in range(1,n_classes):
        idx_Auto = np.where(autoSeg.flatten() == c_i)[0]
        idx_GT   = np.where(groundTruth.flatten() == c_i)[0]
        
        autoArray = np.zeros(autoSeg.size,dtype=np.bool)
        autoArray[idx_Auto] = 1
        
        gtArray = np.zeros(autoSeg.size,dtype=np.bool)
        gtArray[idx_GT] = 1
        
        dsc = dice(autoArray, gtArray)

        #dice = np.sum(autoSeg[groundTruth==c_i])*2.0 / (np.sum(autoSeg) + np.sum(groundTruth))
        DiceArray.append(dsc)
    
    [All_P1, All_T1, All_and] = whole_dice(groundTruth, autoSeg, n_classes)
    if All_P1+All_T1==0:
        All_Dice =1
    else:
        All_Dice = 2*All_and/(All_P1+All_T1)

    return DiceArray, All_Dice


def dice(im1, im2):
    """
    Computes the Dice coefficient
    ----------
    im1 : boolean array
    im2 : boolean array
    
    If they are not boolean, they will be converted.
    
    -------
    It returns the Dice coefficient as a float on the range [0,1].
        1: Perfect overlapping 
        0: Not overlapping 
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.size != im2.size:
        raise ValueError("Size mismatch between input arrays!!!")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 1.0

    # Compute Dice 
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def whole_dice(truth, predict, n_class):
    All_and = 0
    All_P1 = 0
    All_T1 = 0
    for label in range(1, n_class):
        predict_l = predict==label
        truth_l = truth==label
        P1 = np.count_nonzero(predict_l)
        T1 = np.count_nonzero(truth_l)
        TP = np.logical_and(truth_l, predict_l)
        TP_count = np.count_nonzero(TP)
        All_and = All_and + TP_count
        All_P1 = All_P1 + P1
        All_T1 = All_T1 + T1
    return (All_P1, All_T1, All_and)
