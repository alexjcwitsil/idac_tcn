import numpy as np
import obspy
import os 
import idac_tcn as tcn
import random

## compile synthetic events into a single numpy file
## could be used/thought of as a generator function

### FUTURE IMPROVEMENTS SHOUDL INCLUDE ###
### SHOULD SEPARATE FUNCTION TO INPUT SHIFT LOCATION RATHER THAN INPUT PROPAGATION DIRECTORY ####
### SHOULD GET RID OF INDS INPUT

def compile_mseeds(mseed_dir, prop_dir, win, inds = [None, None], center=False):

    ## list the mseed files
    mseed_files_all = os.listdir(mseed_dir)

    ## limit the miniseed files if necessary
    if inds == [None, None]:
        mseed_files = mseed_files_all
    else:
        mseed_files = mseed_files_all[inds[0]: inds[1]]
    #
    
    ## read in dummy file to find the lenght of windowed sythetic files 
    dme_mseed = obspy.read(mseed_dir + mseed_files[0])[0]

    ## grab the sampling rate
    dme_sps = dme_mseed.stats.sampling_rate

    ## now calculate the index length of the windowed synthetic event
    mseed_len = int(dme_sps * win)

    ## initilize the wig_array object
    wig_array = np.zeros((len(mseed_files), mseed_len))
    ## loop over the individaul synthetic waves 
    i=0
    while i < len(mseed_files):

        cur_file = mseed_files[i]

        ## read in the current mseed event
        cur_stream = obspy.read(mseed_dir + cur_file)

        ## taper the ends
        ## cur_synth = cur_trace.taper(0.1).data


        ## grab the sampling rate
        cur_sps = cur_stream[0].stats.sampling_rate

        ########################
        ### WINDOW THE EVENT ###
        ########################

        ## assume we dont want to shift at all (for random recordings)
        ## not the case for synthetic events
        wig_cent =  cur_stream[0].data

        if center == True:

            ## first taper the ends to make sure there are no breaks
            cur_mseed = cur_stream[0].taper(0.1).data
                
            ## grab the current prop file 
            cur_prop_file = cur_file[:-6] + '.dat'

            ## read in the current prop file
            cur_prop = np.loadtxt(prop_dir + cur_prop_file)

            ## find the time of the peak amplitude
            peak_amp_time = cur_prop[np.argmax(np.abs(cur_prop[:,2])),1] - np.min(cur_prop[:,1])

            ## find the index of this time on the synthetic event
            ##shift_by = np.argmin(np.abs(cur_stream[0].times() - peak_amp_time))
            center_shift = np.argmin(np.abs(cur_stream[0].times() - peak_amp_time))
            shift_range = [center_shift - win*cur_sps/2, center_shift + win*cur_sps/2]
            shift_by = random.randint(int(shift_range[0]), int(shift_range[1]))

            ## center the synthetic event based on the peak amplitude of the prpoagated wave
            wig_cent = cur_mseed.copy()
            wig_cent = tcn.center_wig(wig_cent, shift_by)
        #

 
        ## now window the damn thing
        wig_start_ind = int(len(wig_cent)/2 - (win/2 * cur_sps))
        wig_stop_ind = int(len(wig_cent)/2 + (win/2 * cur_sps))
        windowed_wig = wig_cent[wig_start_ind:wig_stop_ind]

        ## shift the wig randomly
        # rand_shift = random.randint(0,len(windowed_wig))
        # rand_shift_wig = np.roll(windowed_wig, rand_shift)
        

        ## save the centered wiggle
        ##wig = rand_shift_wig
        wig = windowed_wig


        ####################################
        ### CHECK IF WIG HAS ENOUGH DATA ###
        ####################################

        if len(wig) != mseed_len:
            i = i+1
            continue
        #

        ## add current mseed to wig_array
        wig_array[i,:] = wig

        i=i+1
        print(i)
    #

    ## remove any rows with all zeros.
    bad_rows = np.where(np.sum(wig_array, axis=1) == 0)
    wig_array = np.delete(wig_array, bad_rows, axis=0)

    return (wig_array)
#

