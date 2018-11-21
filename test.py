from audioSignal import AudioSignal
from fileManager import FileManager

fileManager = FileManager("./datos_proyecto2/")
fileManager.getFilesNamesFromDirectory()
listOfFiles = fileManager.getFilesNames()
print(len(listOfFiles))

listOfAudios = []
for file in listOfFiles[:]:
    print("file:  ", file)
    tmp = AudioSignal(file)
    tmp.normalizeEnergy() 
    tmp.signalFeatures(0.01, 0.025)
    #tmp.pltoFilterBankFeatures()
    #tmp.pltoMFCCFeatures()
    listOfAudios.append(tmp)

print(len(listOfAudios))
print(len(listOfAudios[0].featuresMFCC))
print(len(listOfAudios[0].featuresMFCC[0]))