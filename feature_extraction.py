import numpy as np
import pywt
from scipy.stats import entropy
from scipy.fft import fft

def hjorth_parameters(signal):
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)

    var_zero = np.var(signal)
    var_d1 = np.var(first_derivative)
    var_d2 = np.var(second_derivative)

    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility

    return var_zero, mobility, complexity

def wavelet_energy(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    energy = [np.sum(c ** 2) for c in coeffs]
    return energy

def fft_power(signal):
    fft_vals = np.abs(fft(signal))
    fft_vals = fft_vals[:len(fft_vals)//2]

    delta = np.sum(fft_vals[0:4])
    theta = np.sum(fft_vals[4:8])
    alpha = np.sum(fft_vals[8:12])
    beta  = np.sum(fft_vals[12:30])
    gamma = np.sum(fft_vals[30:50])

    return [delta, theta, alpha, beta, gamma]

def extract_features(file_path):
    signal = np.loadtxt(file_path)

    mean_ = np.mean(signal)
    std_ = np.std(signal)
    max_ = np.max(signal)
    min_ = np.min(signal)
    energy_ = np.sum(signal**2)

    activity, mobility, complexity = hjorth_parameters(signal)
    wavelet_feats = wavelet_energy(signal)
    fft_feats = fft_power(signal)
    hist, _ = np.histogram(signal, bins=50, density=True)
    ent = entropy(hist + 1e-12)

    features = [
        mean_, std_, max_, min_, energy_,
        activity, mobility, complexity,
        ent
    ] + wavelet_feats + fft_feats

    return features
