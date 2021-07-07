import numpy as np
from sklearn.utils import shuffle


## GENERATOR FUNCTION

def gen(X, noise, bs):
    # X.shape = noise.shape = (m, n)
    # where m is number of examples and n is the number of samples
    random.seed()

    ## normalize X between 0 and 1
    x_min = np.repeat(np.min(X, axis=1), repeats=X.shape[1]).reshape(X.shape)
    x_max = np.repeat(np.max(X, axis=1), repeats=X.shape[1]).reshape(X.shape)
    X = (X - x_min) / (x_max - x_min)

    ## normalize the noise between 0 and 1
    noise_min = np.repeat(np.min(noise, axis=1), repeats=noise.shape[1]).reshape(noise.shape)
    noise_max = np.repeat(np.max(noise, axis=1), repeats=noise.shape[1]).reshape(noise.shape)
    noise = (noise - noise_min) / (noise_max - noise_min)

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
        
        yield (Xb, Yb)

