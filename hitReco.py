#!/usr/bin/python3 -O

import numpy as np
import h5py
import argparse
import tqdm



# create sca to time sample conversion map for a given roll mask
def getSCAConversion(rollMask):
    scaToTS = -np.ones(13, dtype=np.uint8) # start with -1 (aka 255 in uint8)
    if ((rollMask>>11) & 1) and (rollMask & 1):
        scaToTS[0] = 12
        for i in range(1,13): scaToTS[i] = i-1
    else:
        posTrk = -1 # posTrk is the position of the first bit that is set
        for i in range(13):
            if (rollMask>>i) & 1:
                posTrk = i
                break
        for i in range(1, posTrk+2):
            scaToTS[i] = i + 12 - (posTrk+1)
        for i in range(posTrk+2, 12):
            scaToTS[i] = i - 1 - (posTrk+1)
    return scaToTS[scaToTS < 13] # some time samples are invalid -- only use good ones


# very basic hit selection
def checkHit(lgADC_CMSub):
    return lgADC_CMSub[2] > 0 # 3rd time sample was greater than the CM noise


# event datatype
eventRecoType = np.dtype([
            ('event', 'i'),
            ('nhits', 'i'),

            ('hit_adc_lg', 'O'), # 'O' for python objects so we can have variable length arrays
            ('hit_adc_hg', 'O'),
            ('hit_toa_fall', 'O'),
            ('hit_toa_rise', 'O'),
            ('hit_tot_slow', 'O'),
            ('hit_tot_fast', 'O'),

            ('hit_channel', 'O'),
            ('hit_chip', 'O'),
            ('hit_layer', 'O'),

            ('hit_iu', 'O'),
            ('hit_iv', 'O'),
            ('hit_iU', 'O'),
            ('hit_iV', 'O'),

            ('hit_x', 'O'),
            ('hit_y', 'O'),
            ('hit_z', 'O'),
        ])

# emap datatype
emapType = np.dtype([
        ('x','f'),
        ('y','f'),
        ('z','f'),
        ('iu','i'),
        ('iv','i'),
        ('iU','i'),
        ('iV','i'),
    ])



# read the input file
unpackedFile = h5py.File('data/unpacked/run1.unpacked.hdf5', 'r')
unpackedData = np.array(unpackedFile['event_data'])
nHexbds = len(unpackedData[0]['hexbd'])
nPads = nHexbds*4*32 # 32 (active) channels per chip, 4 chips per hexbd


# read the emap
emap = np.zeros(nPads, dtype=emapType)



# fill the event reco array
eventRecoData = np.zeros(len(unpackedData), dtype=eventRecoType)
for evidx,event in enumerate(tqdm.tqdm(unpackedData)):

    # calculate common mode noise for the hexaboards
    lgCommonMode = np.zeros((nHexbds,11), dtype=float)
    hgCommonMode = np.zeros((nHexbds,11), dtype=float)
    for hxidx,hexbd in enumerate(event['hexbd']):
        nChans = 0
        for chipidx,chip in enumerate(hexbd['chip']):
            scaToTS = getSCAConversion(chip['roll_position'])
            for chanidx,chan in enumerate(chip['chan']):
                if chanidx % 2: continue # ignore odd channels
                lgADCTS = chan['lg_adc'][scaToTS][:11] # only have 11 time samples
                hgADCTS = chan['hg_adc'][scaToTS][:11]

                lgCommonMode[hxidx] += lgADCTS
                hgCommonMode[hxidx] += hgADCTS
                nChans += 1
        lgCommonMode[hxidx] /= nChans
        hgCommonMode[hxidx] /= nChans

    # determine which are hits
    hitMask = np.zeros(nPads, dtype=bool)
    for hxidx,hexbd in enumerate(event['hexbd']):
        for chipidx,chip in enumerate(hexbd['chip']):
            scaToTS = getSCAConversion(chip['roll_position'])
            for chanidx,chan in enumerate(chip['chan']):
                if chanidx % 2: continue
                lgADCTS_CMSub = chan['lg_adc'][scaToTS][:11] - lgCommonMode[hxidx]
                hgADCTS_CMSub = chan['hg_adc'][scaToTS][:11] - hgCommonMode[hxidx]

                isHit = checkHit(lgADCTS_CMSub)
                if isHit:
                    arrIdx = hxidx*4*32 + chipidx*32 + chanidx//2
                    hitMask[arrIdx] = True

    # fill basic event info
    nHits = hitMask.sum()
    evData = eventRecoData[evidx]
    evData['event'] = evidx+1
    evData['nhits'] = nHits

    # create the hit arrays
    evData['hit_adc_lg'] = np.zeros((nHits,11), dtype=np.float32)
    evData['hit_adc_hg'] = np.zeros((nHits,11), dtype=np.float32)
    evData['hit_toa_rise'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_toa_fall'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_tot_slow'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_tot_fast'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_channel'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_chip'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_layer'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_iu'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_iv'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_iU'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_iV'] = np.zeros(nHits, dtype=np.int16)
    evData['hit_x'] = np.zeros(nHits, dtype=np.float32)
    evData['hit_y'] = np.zeros(nHits, dtype=np.float32)
    evData['hit_z'] = np.zeros(nHits, dtype=np.float32)

    # fill the hit arrays
    hitIdx = 0
    for hxidx,hexbd in enumerate(event['hexbd']):
        layer = hxidx+1
        for chipidx,chip in enumerate(hexbd['chip']):
            scaToTS = getSCAConversion(chip['roll_position'])
            for chanidx,chan in enumerate(chip['chan']):
                arrIdx = hxidx*4*32 + chipidx*32 + chanidx//2
                if chanidx % 2 or not hitMask[arrIdx]: continue
                evData['hit_adc_lg'][hitIdx,:] = chan['lg_adc'][scaToTS][:11] - lgCommonMode[hxidx]
                evData['hit_adc_hg'][hitIdx,:] = chan['hg_adc'][scaToTS][:11] - hgCommonMode[hxidx]
                evData['hit_toa_rise'][hitIdx] = chan['toa_rise']
                evData['hit_toa_fall'][hitIdx] = chan['toa_fall']
                evData['hit_tot_slow'][hitIdx] = chan['tot_slow']
                evData['hit_tot_fast'][hitIdx] = chan['tot_fast']
                evData['hit_channel'][hitIdx] = chanidx
                evData['hit_chip'][hitIdx] = chipidx
                evData['hit_layer'][hitIdx] = layer
                evData['hit_iu'][hitIdx] = emap[arrIdx]['iu']
                evData['hit_iu'][hitIdx] = emap[arrIdx]['iv']
                evData['hit_iu'][hitIdx] = emap[arrIdx]['iU']
                evData['hit_iu'][hitIdx] = emap[arrIdx]['iV']
                evData['hit_x'][hitIdx] = emap[arrIdx]['x']
                evData['hit_y'][hitIdx] = emap[arrIdx]['y']
                evData['hit_z'][hitIdx] = emap[arrIdx]['z']
                hitIdx += 1

# create output file and write dataset to it
np.savez_compressed('tmp', event_data=eventRecoData, allow_pickle=True)

# close input files
unpackedFile.close()
