'''
@brief class to extract the characteristics of the audio file to analize
'''
class FileManager:
    def __init__(self, pFolder):
        self.folder = pFolder
        self.filesName = []
    
    '''
    @brief to set the directory of the files
    @param pFolder the folder direction
    '''
    def setDirectory(self, pFolder):
        self.folder = pFolder

    '''
    @brief get all the files name from a specific folder
    '''
    def getFilesNamesFromDirectory(self):
        from os import listdir
        from os.path import isfile, join
        try:
            self.filesName = [self.folder + f for f in listdir(self.folder) if isfile(join(self.folder, f))]        
        except:
            print("Unable to open the directory")
            return

    '''
    @brief get all the files name from a specific folder
    @return the files name
    '''
    def getFilesNames(self):
        return self.filesName