import pdb
import os
import numpy as np
from os.path import isfile, join


def makeFolder(folderName, display_Str) :
    if not os.path.exists(folderName) :
        os.makedirs(folderName)

    strToPrint = "..Folder " + display_Str + " created..."
    print(strToPrint)


""" Get a set of images from a folder given an array of indexes """
def getImagesSet(imagesFolder, imageIndexes) :
   imageNamesToGetWithFullPath = []
   imageNamesToGet = []
   
   if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]
       imageNames.sort()
   
       # Remove corrupted files (if any)
       if '.DS_Store' in imageNames: imageNames.remove('.DS_Store')

       imageNamesToGetWithFullPath = []
       imageNamesToGet = []
  
       if ( len(imageNames) > 0):  
           imageNamesToGetWithFullPath = [join(imagesFolder,imageNames[imageIndexes[i]]) for i in range(0,len(imageIndexes))]
           imageNamesToGet = [imageNames[imageIndexes[i]] for i in range(0,len(imageIndexes))]

   return (imageNamesToGetWithFullPath,imageNamesToGet)
