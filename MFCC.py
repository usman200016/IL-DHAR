import librosa
import librosa.display
import matplotlib.pyplot as plt

def extract_mfcc(signal, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=n_mfcc)

    return mfcc