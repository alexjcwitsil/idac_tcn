
#  calculate the receptive field length of the neural network

def rec_fld_len(pdict):
    r = 0
    for d in pdict['d']:
        r = r + d * (pdict['k']-1)
    return f'{r/pdict["r_smp"]} seconds'




