#!/usr/bin/python3
# this analyzer makes a 2d histogram of the low gain waveforms from all cells

import numpy as np
import matplotlib.pyplot as plt
import tqdm

# load the reco file and get the `event_data` array from it
events = np.load('data/reco/run1.reco.npz', allow_pickle=True)['event_data']


# setup the output arrays
nevents = events.shape[0]
yvals = [] # for the values of the lg adc
xvals = [] # for the ts values


# loop over events to fill arrays
for evidx,event in enumerate(tqdm.tqdm(events)):
    for hit in range(event['nhits']):
        yvals.append(event['hit_adc_lg'][hit]) # yvals is a python array of numpy arrays
        xvals.append(np.arange(11)+1)

# combine them into two long arrays
yvals = np.concatenate(yvals)
xvals = np.concatenate(xvals)

# make the 2d histogram
fig,ax = plt.subplots(figsize=(8,6))
ax.hist2d(xvals,yvals,cmin=1,bins=(11,100))
ax.set(xlabel='Time Sample',ylabel='Low Gain ADC',title='Run 1')
plt.savefig('run1_waveforms.pdf')
