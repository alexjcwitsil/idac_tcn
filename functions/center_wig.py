import numpy as np



## define a function to center wiggles.
## this is used to window the synthetic events

def center_wig(wig, ind):

    shift_by = (len(wig) - ind) - int(len(wig)/2)
    shift_wig = wig.copy()
    shift_wig = np.roll(shift_wig, shift_by)

    return(shift_wig)

