import numpy as np
from sklearn.utils import shuffle
import random
import numpy as np
import idac_tcn as tcn
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from scipy.fft import fft


## GENERATOR FUNCTION

def gen(X, noise, bs):
    # X.shape = noise.shape = (m, n)
    # where m is number of examples and n is the number of samples
    random.seed()

    ## filtering
    sps = 40
    nyq = 1/2 * sps
    f_min = 0.02 / nyq
    f_max = 5.00 / nyq
    b, a = signal.butter(2, [f_min, f_max], btype='bandpass')

    ## write filter wrapper function
    def wrap_filt(x, b, a):
        out = signal.filtfilt(b, a, x)
        return(out)
    #
  
    # apply filter along event and noise array
    X = np.apply_along_axis(wrap_filt, 1, X, b, a)
    noise = np.apply_along_axis(wrap_filt, 1, noise, b, a)
  






    # ## try the frequency domain
    # def moving_average(x, w):
    #     return np.convolve(x, np.ones(w), 'same') / w
    # def wrap_dmean(x):
    #     return(x - np.mean(x))

    # X = np.apply_along_axis(wrap_dmean, 1, X)
    # X = np.abs(fft(X))
    # noise = np.abs(fft(noise))

    # X = np.apply_along_axis(moving_average, 1, X, 10)
    # noise = np.apply_along_axis(moving_average, 1, noise, 10)

    # X = X[:,0:X.shape[1]//2]
    # noise = noise[:,0:noise.shape[1]//2]









    ## normalize events and noise between 0 and 1
    X = (MinMaxScaler().fit_transform(X.transpose())).transpose()
    noise = (MinMaxScaler().fit_transform(noise.transpose())).transpose()

    ## check if there are nan values in the training data
    X = X[~np.isnan((np.sum(X, axis=1)))]
    noise = noise[~np.isnan((np.sum(noise, axis=1)))]

    while True:
        # Normal distribution with mean bs/2 and sigma bs/20 batch
        # Set positives = bs/2 for a perfect 50/50 split
        #positives = int(bs/2)
        positives = int(np.random.normal(bs/2, bs/20))
        negatives = bs - positives
        
        # Select random indexes from the entire dataset for the batch
        # These indexes may repeat.  There are Signals x Noise possible combinations before any other augmentation.
        pos_idx = np.random.choice(range(X.shape[0]), positives)
        neg_idx = np.random.choice(range(noise.shape[0]), negatives)
        
        # pos_x is the propagated signal with random noise added
        # other pre-processing batch functions can be added here
        ##pos_x = add_noise(X[pos_idx], noise[aug_noise_idx], noise_scale=.1, signal_scale=10)
        pos_x = X[pos_idx]
        
        neg_x = noise[neg_idx]
        
        Yb = np.concatenate((np.ones(positives, dtype=np.dtype('int8')), np.zeros(negatives, dtype=np.dtype('int8'))))
        Xb = np.concatenate((pos_x, neg_x), axis=0)
        
        Xb, Yb = shuffle(Xb, Yb)
        ##Xb = Xb*np.random.rand(bs,1,1)

        yield (Xb, Yb)

