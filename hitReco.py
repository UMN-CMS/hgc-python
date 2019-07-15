#!/usr/bin/python3 -O

import numpy as np
import h5py
import argparse
import tqdm



eventType = np.dtype([
        ('hit_adg_hg', 'f'),
        ('hit_adc_lg', 'f'),
        ('hit_toa_fall', 'u2'),
        ('hit_toa_rise', 'u2'),
        ('hit_tot_slow', 'u2'),
        ('hit_tot_fast', 'u2'),

        ('hit_channel', 'u2'),
        ('hit_chip', 'u2'),
        ('hit_layer', 'u2'),

        ('hit_iu', 'i2'),
        ('hit_iv', 'i2'),
        ('hit_iU', 'i2'),
        ('hit_iV', 'i2'),

        ('hit_x', 'f'),
        ('hit_y', 'f'),
        ('hit_z', 'f'),
    ])



unpackedFile = h5py.File('data/unpacked/run1.unpacked.hdf5')

unpackedData = np.array(unpackedFile['event_data'])

for evidx,event in enumerate(unpackedData):
    for hxidx,hexbd in enumerate(event['hexbd']):
        for chipidx,chip in enumerate(hexbd['chip']):
            for chanidx,chan in enumerate(chip['chan']):
                if chanidx == 0: continue
                print(chanidx,chan['lg_adc'])
                exit()
