import numpy as np
import wave
import pylab
import os
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math as math
from PIL import Image
plt.switch_backend('agg')
   
# def get_wav_info(file_path):
#     #wav = wave.open(file_path, 'r')
#     #frames = wav.readframes(-1)
#     #sound_info = pylab.fromstring(frames, 'Int16')
#     #frame_rate = wav.getframerate()
#     #wav.close()
#     [frame_rate, frames] = wavfile.read(file_path)
#     #scaling to -1 , 1
#     max_nb_bit = float(2**(bitDepth-1))  
#     frames = frames / (max_nb_bit + 1.0) 
    
#     return frames, frame_rate

windowSize = 256
frameStep = 64

def get_wav_info(file_path):
    wav = wave.open(file_path, 'r')
    frames = wav.readframes(-1)
    frames = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    byteDepth = wav.getsampwidth()
    bitDepth = byteDepth * 8
   
    max_nb_bit = float(2**(bitDepth-1))  
    frames = frames / (max_nb_bit + 1.0) 
   
    wav.close()
    return frames, frame_rate

def graph_spectrogram(file_path, out_img_file):
    sound_info, frame_rate = get_wav_info(file_path)
    fig = plt.figure(num=None, figsize=(19, 12), edgecolor = None, frameon=False)
    #plt.subplot(111)
    frames, freqs, time, im = plt.specgram(sound_info, Fs=frame_rate)#, cmap=plt.cm.gist_heat)
    im.axes.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    #im.frameon = False
    plt.axis((0,max(time),0,max(freqs)))
    
    #fig.savefig(out_img_file, bbox_inches='tight', transparent=True, pad_inches=0, edgecolor='w')
    plt.savefig(out_img_file, bbox_inches='tight', transparent=True, pad_inches=0, edgecolor='w')
    
    #fig.savefig(out_img_file, bbox_inches='tight', transparent=True, pad_inches=0, edgecolor='w')
    plt.clf()
    plt.close()
    

def SpectogramToImage(file_path, out_img_file):
    waveform, frame_rate = get_wav_info(file_path)
    spect = CreateSpectogram(waveform)
    magSpec, freqs, times = SpectrogramForDisplay(spect, frame_rate, frameStep)
    A = np.real(magSpec)
    normalized, minEl, maxEl = normalize(A)
    normedRGB = toRGB(normalized)
    img = Image.fromarray(normedRGB)
    img.save(out_img_file)
    return normedRGB, minEl, maxEl

def CreateSpectogram(waveform, windowSize = 256, frameStep = 64):
    fftSize = 2*windowSize
    fftB = int(math.floor(windowSize/2))
    fftE = int(fftB + windowSize)
    fftBuffer = np.zeros(fftSize)
    
    frameCount = int(math.floor((waveform.size - windowSize)/frameStep) + 1)
    spectrum = np.zeros((fftSize, frameCount));
    spectrum = spectrum + 0j
    
    h = 0.54 - 0.46*np.cos(2*math.pi* np.arange(windowSize)/(windowSize-1))
    for frameNumber in range(0,frameCount):
        waveB = (frameNumber)*frameStep
        waveE = waveB + windowSize
        fftBuffer = 0*fftBuffer #Make sure the entire buffer is empty
        fftBuffer[fftB:fftE] = np.multiply(waveform[waveB:waveE], h)
        fftBuffer = np.fft.fftshift(fftBuffer)
        spectrum[:,frameNumber] = np.fft.fft(fftBuffer)
        
    return spectrum

def SpectrogramForDisplay(spec, fs, frameStep, enhanceHighFrequencies=1):
    fftSize, frameCount = spec.shape
    windowSize = int(math.floor(fftSize/2))
    freqs = np.arange(windowSize+1)*fs/fftSize;
    times = np.arange(frameCount+1) * float(frameStep)/fs;
    posFreqs = spec[0:windowSize+1, :];
    negFreqs = np.flipud(spec[windowSize:,:])

    powerSpec = posFreqs;
    freqRange = np.arange(1,windowSize+1);
    powerSpec[freqRange,:] = np.multiply(powerSpec[freqRange,:], negFreqs)

    magSpec = np.sqrt(powerSpec);
    return magSpec, freqs, times

def normalize(arr):
    fliped_real = np.flipud(arr)
    normalized = ( fliped_real - fliped_real.min())/( fliped_real.max() - fliped_real.min())
    minEl = fliped_real.min()
    maxEl = fliped_real.max()
    return normalized, minEl, maxEl

def toRGB(arr):
    gray = (arr*255.).astype('uint8')
    res = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    return res


def fromRgb(arr):
    G = arr[:,:,0]
    return (G/255.).astype(float)

def originalFromNormalized(arr, minEl, maxEl):
    return arr*( maxEl - minEl ) + minEl

def backwardDisplaySpectogram(spect,originalFromImage, windowSize ):
    backwardSpec = spect
    fftSize, frameCount = spect.shape
    backwardSpec[0] = np.multiply(originalFromImage[0], originalFromImage[0])
    freqRange = np.arange(1,windowSize+1);
    backwardSpec[freqRange,:] = originalFromImage[freqRange,:]
    negFreqRange = np.arange(windowSize,fftSize);
    backwardSpec[negFreqRange,:] = np.flipud(originalFromImage[freqRange,:])
    return backwardSpec

def SaveRGB(arr, path):
    img = Image.fromarray(arr)
    img.save(path)
    return img

def ReadRGB(path):
    im = Image.open(path, 'r')
    height, width = im.size
    pixel_values = list(im.getdata())
    pixel_values = np.array(pixel_values).reshape((width, height, 3))
    pixel_values = np.uint8(pixel_values)
    return pixel_values