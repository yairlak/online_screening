import scipy.io as sio
import os
import numpy as np
import glob


def get_spike_profile(path2data, ch_num, class_num):
    curr_data = sio.loadmat(os.path.join(path2data, f'times_CSC{ch_num}'))
    cluster_class = curr_data['cluster_class']
    spikes = curr_data['spikes']
    IX_class = (cluster_class[:, 0] == class_num)
    return spikes[IX_class, :]


def event2matrix(spike_events, t_start, t_end):
    '''
    Transform a list of spike trains specified in time into a matrix of ones and zeros
    :param spike_events: list of spikes
    :param t_start: (int) in msec
    :param t_end: (int) in msec
    :return:
    '''
    num_trials = len(spike_events)
    # max_t = np.ceil(max([max(sublist) for sublist in spike_events]))
    # min_t = np.floor(min([min(sublist) for sublist in spike_events]))
    # if not max_t:
    #     return np.zeros((num_trials, (int(t_end)-int(t_start))))
    # print(min_t, max_t, t_start, t_end)
    # assert t_end > max_t
    # assert t_start < min_t
    mat = np.zeros((num_trials, (int(t_end)-int(t_start))))
    for i_trial, spike_train in enumerate(spike_events):
        for j_spike in spike_train[0]:
            mat[i_trial, int(j_spike-t_start)] = 1
    return mat

def smooth_with_gaussian(time_series, sfreq=1000, gaussian_width = 50, N=1000):
    # gaussian_width in samples
    # ---------------------
    import math
    from scipy import signal

    norm_factor = np.sqrt(2 * math.pi * gaussian_width ** 2)/sfreq # sanity check: norm_factor = gaussian_window.sum()
    gaussian_window = signal.general_gaussian(M=N, p=1, sig=gaussian_width) # generate gaussian filter
    norm_factor = (gaussian_window/sfreq).sum()
    smoothed_time_series = np.convolve(time_series, gaussian_window/norm_factor, mode="full") # smooth
    smoothed_time_series = smoothed_time_series[int(round(N/2)):-(int(round(N/2))-1)] # trim ends
    return smoothed_time_series


def get_indexed_array(input, indices) : 
    output = []

    for i in indices : 
        output.append(input[i])

    return output


def getSite(site, plot_regions) : 
    
    if site == "RAH" or site == "RMH" :
        site = "RH"
    if site == "LAH" or site == "LMH" :
        site = "LH"

    if plot_regions == "collapse_hemispheres" : 
        site = site[1:]
    elif plot_regions == "hemispheres" : 
        site = site[0]

    return site


