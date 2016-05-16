import wave
import pylab
import audiotools
import os
import numpy as np

name_counter = 0
frames_number = 100

def SeparateNistOnHeaderAndData(input_file):
    """
    Header begins with a label of the form NISTxx where xx is a version code followed by the number of bytes in the header.

    sample_rate - sample rate in Hz
    sample_n_bytes - number of bytes in each sample
    sample_count - number of samples in file
    sample_byte_format - byte order
    sample_coding - speech coding eg pcm,  tex2html_wrap_inline19966 law, shortpack
    channels_interleaved - for 2 channel data only
    """
    #TODO: parce header to get all values
    with open(input_file, 'rb') as pcmfile:
        Nist_version = pcmfile.readline()
        HeaderBytes = int(pcmfile.readline())
            #closing file. Now we know header size

    with open(input_file, 'rb') as pcmfile:
        header = pcmfile.read(HeaderBytes)
        data = pcmfile.read()
        
    return (header,data)

def PCM2Wav(input_file, output_path):
    output_path = output_path + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    file_name = os.path.basename(input_file)
    wav_name = file_name.replace(".WAV", ".wav")
    wav_file= output_path + wav_name
    
    header, data = SeparateNistOnHeaderAndData(input_file)

    wavfile = wave.open(wav_file, 'wb')
    wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE')) #mono, 16bit, 16000 Hz
    wavfile.writeframes(data)
    wavfile.close()
    #audiotools.open(input_file).convert(wav_file, audiotools.WaveAudio)
    return wav_file

def Flac2Wav(flac_file, wav_path):
    wav_path = wav_path + "/"
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)
        
    file_name = os.path.basename(flac_file)
    wav_name = file_name.replace(".WAV", ".wav")
    wav_file= wav_path + wav_name
    audiotools.open(flac_file).convert(wav_file, audiotools.WaveAudio)
    return wav_file

def cutIntoChunks(file_path, output_folder):
    output_folder = output_folder + "/"
    chunkDuration = 2./100; # 20 ms
    chankStep =  2./100; # 20 ms
    wav = wave.open(file_path, 'r')

    frameRate = wav.getframerate()
    nChannels = wav.getnchannels()
    sampWidth = wav.getsampwidth()
    
    numSamplesPerChunk = int(chunkDuration*frameRate);
    numSamplesPerStep = int(chankStep*frameRate);
    
    totalNumSamples = wav.getnframes();

    chunkCnt = 0;
    startLoc = 0;
    
    while startLoc + numSamplesPerChunk <= totalNumSamples:
        endLoc = min(startLoc + numSamplesPerChunk - 1,totalNumSamples)
        wav.setpos(startLoc)
        chunk_frames = wav.readframes(endLoc-startLoc)
    
        test_chunk = output_folder + str(chunkCnt) + ".wav"
        chunkAudio = wave.open(test_chunk,'w')
        chunkAudio.setnchannels(nChannels)
        chunkAudio.setsampwidth(sampWidth)
        chunkAudio.setframerate(frameRate)
        chunkAudio.writeframes(chunk_frames)
        chunkAudio.close()
    
        startLoc += numSamplesPerStep
        chunkCnt += 1
    
    wav.close()
    #os.remove(file_path)

def cutPhonemeChunk(file_path, output_folder, from_frame, to_frame, phone):
    
    global name_counter
    file_name = os.path.basename(file_path)
    name, extension = os.path.splitext(file_name)
    
    parts = file_path.split("/")
    speaker = parts[-2]
    #file_name = parts[-1].split(".")[0]
    
    speaker_folder = output_folder + "/" + speaker
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)
    
    name_candidate = phone + "_" + str(from_frame) + "_" + str(to_frame) + "_" + name
    output_file = speaker_folder + "/" + name_candidate + extension
    if os.path.isfile(output_file):
        output_file = speaker_folder + "/" + name_candidate + "_" + str(name_counter) + extension
        name_counter = name_counter + 1    
    
    #print output_file
    
    wav = wave.open(file_path, 'r')

    frameRate = wav.getframerate()
    nChannels = wav.getnchannels()
    sampWidth = wav.getsampwidth()
    
    totalNumSamples = wav.getnframes();
    start = from_frame
    #int((float(from_frame)/frames_number)*frameRate)
    #print start
    end = to_frame
    #int((float(to_frame)/frames_number)*frameRate)
    #print end
    
    wav.setpos(start)
    chunk_frames = wav.readframes(end-start)
    
    chunkAudio = wave.open(output_file,'w')
    chunkAudio.setnchannels(nChannels)
    chunkAudio.setsampwidth(sampWidth)
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunk_frames)
    chunkAudio.close()
    
    wav.close()
    return output_file