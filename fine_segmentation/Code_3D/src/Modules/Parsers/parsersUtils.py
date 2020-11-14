import configparser
import json
import os

# -------- Parse parameters to create the network model -------- #
class parserConfigIni(object):
   def __init__(_self):
      _self.networkName = []
      
   #@staticmethod
   def readConfigIniFile(_self,fileName,task):
      # Task: 0-> Generate model
      #       1-> Train model
      #       2-> Segmentation

      def createModel():
          print (" --- Creating model (Reading parameters...)")
          _self.readModelCreation_params(fileName)
      def trainModel():
          print (" --- Training model (Reading parameters...)")
          _self.readModelTraining_params(fileName)
      def testModel():
          print (" --- Testing model (Reading parameters...)")
          _self.readModelTesting_params(fileName)
       
        # TODO. Include more optimizers here
      optionsParser = {0 : createModel,
                       1 : trainModel,
                       2 : testModel}
      optionsParser[task]()

   # Read parameters to Generate model

      # TODO: Do some sanity checks

   # Read parameters to TRAIN model
   def readModelTraining_params(_self,fileName) :
      ConfigIni = configparser.ConfigParser()
      ConfigIni.read(fileName)

      # Get training image names
      # Paths
      _self.imagesFolder             = ConfigIni.get('Training Images','imagesFolder')
      _self.GroundTruthFolder        = ConfigIni.get('Training Images','GroundTruthFolder')
      _self.folderName               = ConfigIni.get('Training Images', 'folderName')
      _self.indexesForTraining       = json.loads(ConfigIni.get('Training Images','indexesForTraining'))
      _self.timeForValidation        = json.loads(ConfigIni.get('Training Images','timeForValidation'))
      _self.imageTypesTrain          = json.loads(ConfigIni.get('Training Images','imageTypes'))
      
      # training params
      _self.n_classes                         = json.loads(ConfigIni.get('Training Parameters','n_classes')) # Number of (segmentation) classes
      _self.Patch_Size_Train                  = json.loads(ConfigIni.get('Training Parameters', 'Patch size for training'))
      _self.batch_size                        = json.loads(ConfigIni.get('Training Parameters','batch_size'))
      _self.numberOfEpochs                    = json.loads(ConfigIni.get('Training Parameters','number of Epochs'))
      _self.numberOfSubEpochs                 = json.loads(ConfigIni.get('Training Parameters','number of SubEpochs'))
      _self.numberOfSamplesSupEpoch           = json.loads(ConfigIni.get('Training Parameters','number of samples at each SubEpoch Train'))

   def readModelTesting_params(_self,fileName) :
      ConfigIni = configparser.ConfigParser()
      ConfigIni.read(fileName)
 
      _self.imagesFolder             = ConfigIni.get('Segmentation Images','imagesFolder')
      _self.folderName               = ConfigIni.get('Segmentation Images','folderName')
      _self.GroundTruthFolder        = ConfigIni.get('Segmentation Images','GroundTruthFolder')
     
      _self.imageTypes               = json.loads(ConfigIni.get('Segmentation Images','imageTypes'))
      _self.indexesToSegment         = json.loads(ConfigIni.get('Segmentation Images','indexesToSegment'))
      
      _self.n_classes                = json.loads(ConfigIni.get('Segmentation Images','n_classes'))
      _self.batch_size               = json.loads(ConfigIni.get('Segmentation Images','batch_size'))
      _self.Patch_Size_Test          = json.loads(ConfigIni.get('Segmentation Images', 'Patch size for segmenting'))
      _self.strideValues             = json.loads(ConfigIni.get('Segmentation Images', 'Stride for patch'))

