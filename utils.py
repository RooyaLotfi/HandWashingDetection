import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split


def extract_feature(X, sample_rate, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result


def load_data(test_size=0.2):
    X, y = [], []
    time_window = 10
    for file in glob.glob("data/*washing/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the state label washing or not washing
        state = basename.split("-")[0]
        # extract speech features
        with soundfile.SoundFile(file) as sound_file:
            audio = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            win_size = time_window*sample_rate
            if win_size <= np.size(audio):
                i = 0
                while i <= np.size(audio) - win_size:
                    features = extract_feature(audio[i:i+win_size], sample_rate, mfcc=True, chroma=True, mel=True)
                    i = i+win_size
                    # add to data
                    X.append(features)
                    y.append(state)

    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

