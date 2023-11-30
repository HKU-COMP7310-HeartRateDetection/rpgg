# read mat file
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from scipy import sparse
import matplotlib.pyplot as plt
import time
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
from scipy.signal import find_peaks
import os

def POS_WANG(frames, fs):
    WinSec = 1.6
    RGB = _process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP

def _process_video(frames):
    """Calculates the average value of each frame."""
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)

def _calculate_fft_hr(ppg_signal, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

# Define the directory path
dir_path = './mmpd'
for filename in os.listdir(dir_path):
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(os.path.join(dir_path, filename)):
        if os.path.isfile(os.path.join(dir_path, filename)):
            # Do something with the file
            sample_file = sio.loadmat('./mmpd/' + filename)
            video = sample_file['video']
            frames = np.array(sample_file['video'])
            BVP_POS = POS_WANG(frames, 30)
            print(filename,":",_calculate_fft_hr(BVP_POS))