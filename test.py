from audioSignal import AudioSignal
from fileManager import FileManager

fileManager = FileManager("./datos_proyecto2/")
fileManager.getFilesNamesFromDirectory()
listOfFiles = fileManager.getFilesNames()
print(len(listOfFiles))

listOfAudios = []
for file in listOfFiles[:3]:
    print("file:  ", file)
    tmp = AudioSignal(file)
    tmp.getAudioSignalFrequency()
    tmp.normalizeEnergy()   
    #tmp.audioSignalTransformFrequency()
    tmp.getMFCCFeatures()
    tmp.getFilterBankFatures()
    #tmp.pltoMFCCFeatures()
    #tmp.pltoFilterBankFeatures()
    #print("Frequency: ",tmp.frequencySampling)
    #print("Signal: ", tmp.audioSignal)
    print("MFCC: ", tmp.featuresMFCC)
    print("Blank: ", tmp.filterbankFeatures)
    listOfAudios.append(tmp)
