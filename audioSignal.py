import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

'''
@brief class to extract the characteristics of the audio file to analize
'''
class AudioSignal:
    def __init__(self,pFileName):
        self.fileName = pFileName
        self.frequencySampling = None 
        self.audioSignal = None 
        self.featuresMFCC = None
        self.filterbankFeatures = None
        self.signalPower = None
        self.plotFlag = False
    
    '''
    @brief set the audio file name and directory (must be a .wav format)
    @param pFileName the directory of the file
    '''
    def setFileName(self, pFileName):
        self.fileName = pFileName

    '''
    @brief set the audio frequency sampling
    @param pFrequencySampling the value
    '''
    def setFrequencySampling(self, pFrequencySampling):
        self.frequencySampling = pFrequencySampling

    '''
    @brief set the audio audio signal characterization
    @param self the directory of the file
    '''
    def setAudioSignal(self, pAudioSignal):
        self.audioSignal = pAudioSignal

    '''
    @brief set the audio mfcc features
    @param pMFCC
    '''
    def setMFCCFeature(self, pMFCC):
        self.featuresMFCC = pMFCC

    '''
    @brief set the audio filter bank Features
    @param pFilterbankFeatures
    '''
    def setFilterBankFeature(self, pFilterbankFeatures):
        self.filterbankFeatures = pFilterbankFeatures
    
    '''
    @brief methods that get the frequency and the audio signal of an audio file
    @returns frequency_sampling, audioSignal
    '''
    def getAudioSignalFrequency(self):
        try:
            frequencySampling, audioSignal = wavfile.read(self.fileName)
            return frequencySampling, audioSignal
        except:
            print("Unable to get frequency_sampling, audioSignal from " + self.fileName)
            return 

    '''
    @brief use the mfcc librarie to to get some mfcc features
    @return featuresMFCC
    '''
    def getMFCCFeatures(self):
        try:
            featuresMFCC = mfcc(self.audioSignal, self.frequencySampling)
            print('\nMFCC:\nNumber of windows =', featuresMFCC.shape[0])
            print('Length of each feature =', featuresMFCC.shape[1])
            return featuresMFCC
        except:
            print("Could not get the mfcc features from file "+ self.fileName)

    '''
    '''
    def getFilterBankFatures(self):
        try:
            filterbankFeatures = logfbank(self.audioSignal, self.frequencySampling)
            print('\nFilter bank:\nNumber of windows =', filterbankFeatures.shape[0])
            print('Length of each feature =', filterbankFeatures.shape[1])
            return filterbankFeatures
        except:
            print("Could not get filter bank features from file")

    '''
    @brief normalize the energy
    '''
    def normalizeEnergy(self):
        print('\nSignal shape:', self.audioSignal.shape)
        print('Signal Datatype:', self.audioSignal.dtype)
        print('Signal duration:', round(self.audioSignal.shape[0] / 
        float(self.frequencySampling), 2), 'seconds')

        self.audioSignal = self.audioSignal / np.power(2, 15)

        #if(self.plotFlag):
        #   self.plotSignal('Frequency (kHz)', 'Signal power (dB)', 'Freq vs Power', 1000.0, lengthSignal, len_fts, 1, self.signalPower)

    '''
    '''
    def audioSignalTransformFrequency(self):
        lengthSignal = len(self.audioSignal)
        half_length = np.ceil((lengthSignal + 1) / 2.0).astype(np.int)
        signalFrequency = np.fft.fft(self.audioSignal)
        signalFrequency = abs(signalFrequency[0:half_length]) / lengthSignal
        signalFrequency **= 2
        len_fts = len(signalFrequency)
        if lengthSignal % 2:
            signalFrequency[1:len_fts] *= 2
        else:
            signalFrequency[1:len_fts-1] *= 2
        self.signalPower = 10 * np.log10(signalFrequency)
        if(self.plotFlag):
            self.plotSignal('Frequency (kHz)', 'Signal power (dB)', 'Freq vs Power', 1000.0, lengthSignal, len_fts, 1, self.signalPower)

    '''
    @brief method to plot the audio signal
    '''
    def plotSignal(self,pXLabel, pYLabel, pTitle, pAmount, pSize, pSize2, pType, pSignal):
        if(pType ==0): 
            time_axis = pAmount * np.arange(0, pSize, 1) / float(self.frequencySampling)
        else: 
            time_axis = np.arange(0, pSize2, 1)*(self.frequencySampling / pSize) / pAmount
        plt.plot(time_axis, pSignal, color='blue')
        plt.xlabel(pXLabel)
        plt.ylabel(pYLabel)
        plt.title(pTitle)
        plt.show()

    
    
    '''
    @brief plot the mfcc features
    @param pFeaturesMFCC the mfcc features
    '''
    def pltoMFCCFeatures(self):
        pFeaturesMFCC = self.featuresMFCC.T
        plt.matshow(pFeaturesMFCC)
        plt.title('MFCC Features')
        plt.show()

    '''
    @brief plot the mfcc features
    @param pFeaturesMFCC the mfcc features
    '''
    def pltoFilterBankFeatures(self):
        filterbankFeatures = self.filterbankFeatures.T
        plt.matshow(filterbankFeatures)
        plt.title('Filter bank')
        plt.show()