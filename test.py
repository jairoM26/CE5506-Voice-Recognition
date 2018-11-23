from audioSignal import AudioSignal
from fileManager import FileManager

fileManager = FileManager("./datos_proyecto2/")
fileManager.getFilesNamesFromDirectory()
listOfFiles = fileManager.getFilesNames()
print(len(listOfFiles))

listOfAudios = []
for file in listOfFiles[:]:
    print("FileName ", )    
    tmp = AudioSignal(file)
    tmp.normalizeEnergy() 
    tmp.signalFeatures(0.01, 0.025)
    tmp.plotSignal('Time (ms)', 'Amplitud', 'Audio Signal ' + tmp.fileName.split("/")[-1] , len(tmp.audioSignal), len(tmp.audioSignal), len(tmp.audioSignal), 0, tmp.audioSignal)
    tmp.plotSignal('F(KHz)', 'Signal Power (db)', 'FFT ' + tmp.fileName.split("/")[-1], len(tmp.pow_frames), len(tmp.pow_frames), len(tmp.pow_frames), 1, tmp.pow_frames)
    tmp.pltoFilterBankFeatures()
    tmp.pltoMFCCFeatures()
    listOfAudios.append(tmp)

print(len(listOfAudios))
print(len(listOfAudios[0].featuresMFCC))
print(len(listOfAudios[0].featuresMFCC[0]))