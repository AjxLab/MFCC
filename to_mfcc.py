# -*- coding:utf-8 -*-
import numpy as np
import librosa

def to_mfcc(file, n_mfcc=12, rate=16000):
    ## -----*----- 音声データをMFCCに変換 -----*----- ##
    x, fs = librosa.load(file, sr=rate)
    mfcc = librosa.feature.mfcc(x, sr=fs, n_mfcc=n_mfcc)
    mfcc = np.reshape(mfcc, (mfcc.shape[0], mfcc.shape[1], 1))
    return np.array(mfcc, dtype=np.float32)


if __name__ == '__main__':
    wav_file = 'your .wav file path'
    to_mfcc(wav_file)
