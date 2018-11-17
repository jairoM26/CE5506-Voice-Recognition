from audioSignal import AudioSignal
from fileManager import FileManager

fileManager = FileManager("./datos_proyecto2/")
fileManager.getFilesNamesFromDirectory()
listOfFiles = fileManager.getFilesNames()
print(len(listOfFiles))

listOfAudios = []
for file in listOfFiles[:10]:
    print("file:  ", file)
    tmp = AudioSignal(file)
    tmp.getAudioSignalFrequency()