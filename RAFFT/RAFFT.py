import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

def findmid(spec):
    M = np.ceil(len(spec)/2)
    specM = spec[int(M-np.floor(M/4)-1):int(M+np.floor(M/4)-1)]
    I = np.argmin(specM)
    mid = int(I + M - np.floor(M/4))
    return mid

def move(seg, lag):
    if lag == 0 or lag >= len(seg):
        movedSeg=seg
        return movedSeg
    if lag > 0:
        ins = np.ones(lag)*seg[0]
        movedSeg = np.concatenate((ins,seg[0:len(seg)-lag-1]))
    elif lag < 0:
        lag = np.abs(lag)
        ins = np.ones(lag)*seg[len(seg)-1]
        movedSeg = np.concatenate((seg[lag:len(seg)-1], ins))
    return movedSeg

def FFTcorr(spectrum, target, shift):
    # M = len(target)
    # diff = 1000000
    # for i in range(1, 20):
    #     curdiff = ((2**i) - M)
    #     if curdiff > 0 and curdiff < diff:
    #         diff = curdiff
    #
    # target[M + diff-1] = 0
    # spectrum[M + diff-1] = 0
    # M = M + diff
    # x = np.fft(target)
    # y = np.fft(spectrum)
    # R = x * np.conjugate(y)
    # R = R / (M)
    # rev = np.ifft(R)
    # vals = np.real(rev)
    # maxpos = 1
    # maxi = -1
    # if M < shift:
    #     shift = M
    # for i in range(shift):
    #     if vals[i] > maxi:
    #         maxi = vals[i]
    #         maxpos = i
    #     if vals[len(vals) - i] > maxi:
    #         maxi = vals[len(vals) - i]
    #         maxpos = len(vals) - i
    #
    # # if the max segment correlation is very poor then assume no correlation and
    # # no lag
    # if maxi < 0.1:
    #     lag = 0
    #     return lag
    # if maxpos > len(vals) / 2:
    #     lag = maxpos - len(vals) - 1
    # else:
    #     lag = maxpos - 1
    lag = np.argmax(correlate(spectrum, target, 'same', method='fft'))
    return lag

def recurAlign(spectrum, reference, shift, lookahead):
    if len(spectrum) < 10:
        aligned = spectrum
        return aligned
    lag = FFTcorr(spectrum, reference, shift)
    if lag == 0 and lookahead <= 0:
        aligned = spectrum
        return aligned
    else:
        if lag == 0:
            lookahead = lookahead - 1
        aligned = spectrum
        if np.abs(lag) < len(spectrum):
            aligned = move(spectrum, lag)
        mid = findmid(aligned)
        firstSH = aligned[0:mid-1]
        firstRH = reference[0:mid-1]
        secSH = aligned[mid:len(aligned)-1]
        secRH = reference[mid:len(reference)-1]
        aligned1 = recurAlign(firstSH, firstRH, shift, lookahead)
        aligned2 = recurAlign(secSH, secRH, shift, lookahead)
        aligned = np.concatenate((aligned1, aligned2))
    return aligned

def RAFFT(spectra, reference, shift, lookahead):
    if len(reference) != len(spectra):
        raise ArithmeticError
    elif len(reference) == 1:
        raise ArithmeticError

    if shift is None:
        shift = len(reference)
        lookahead = 1
    elif shift is not None or lookahead == 0:
        lookahead = 1
    alignedSpectrum = recurAlign(spectra, reference, shift, lookahead)
    return alignedSpectrum

print('Small script to explain the use of "RAFFT.py"')
print('Loading example data ("gauss.txt"; two Gaussians plus some noise).')
X = np.loadtxt('gauss.txt')
plt.plot(X[0], 'b-')
plt.plot(X[1], 'r-')
Y = RAFFT(X[1], X[0], shift=None, lookahead=1)
plt.plot(Y, 'k-')
plt.title('Data (blue=target, red=sample, black=corrected)')
plt.show()

