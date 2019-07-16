#!/usr/bin/python3 -O

import numpy as np
import h5py
import argparse
import tqdm



# argument processing
def getOpts():
    parser = argparse.ArgumentParser(description='HGCal Data Reconstructor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fileName',
                        type=str,
                        help='unpacked file')
    parser.add_argument('--unpackDir',
                        type=str,
                        default='data/unpacked/',
                        help='directory to look for the unpacked data')
    parser.add_argument('--recoDir',
                        type=str,
                        default='data/reco/',
                        help='directory for the output data')
    parser.add_argument('--eMap',
                        type=str,
                        default='extern/hgcal-tb-emaps/json_emaps/cosmicstand_2layer.emap.txt',
                        help='path to the electronic map')
    parser.add_argument('--layerDistances',
                        type=str,
                        default='extern/hgcal-tb-emaps/json_emaps/cosmicstand_2layer_layer_distances.txt',
                        help='path to the layer distances file')
    return parser.parse_args()


# filename management
def makeOutFileName(dirName, rawFileName):
    outFileName = dirName
    runNum = rawFileName.split('run')[1].rstrip('.unpacked.hdf5')
    outFileName += 'run'+runNum
    outFileName += '.reco'
    return outFileName


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


# emap generation
emapType = np.dtype([
        ('iu','i'),
        ('iv','i'),
        ('iU','i'),
        ('iV','i'),
        ('x','f'),
        ('y','f'),
        ('z','f'),
        ('type','i'),
    ])

def uvToXY(iu,iv,iU,iV):

    a = 0.6496345 # Size in terms of 1 unit of x/y co-ordinates of a cell side which is 0.064 cm
    A = 11*a # One side of a full sensor(neglecting the cut at the MB)
    x_a = np.sqrt(3) / 2 # cosine pi/6
    y_a = 1 / 2. # sine pi/6
    vy_a = 3. / 2

    # Translation in u,v co-ordinates in terms of TB cartesian -x,y.
    x0 = 2 * x_a * a #Translation in Cartesian x for 1 unit of iu
    vx0 = x_a * a # Cartesian x component of translation for 1 unit of iv
    vy0 = vy_a * a # Cartesian y component of translation for 1 unit of iv
    # Translation in Sensor_u, Sensor_v co-ordinates in terms of TB cartesian -x,y.
    X0 = 2 * x_a * A #Translation in Cartesian x for 1 unit of Sensor_iu
    VX0 = x_a * A # Cartesian x component of translation for 1 unit of Sensor_iv
    VY0 = vy_a * A

    # x,y within the sensor
    x = iu * x0 + iv * vx0
    y = iv * vy0

    # sensor offsets
    x += iU*X0 + iV*VX0
    y += iV*VY0

    return x,y

def createEMap(emapName,layDistName):
    # read layer distances
    layerDistances = np.loadtxt(layDistName, delimiter=',', skiprows=1, unpack=True, usecols=1)

    # create emap array
    nLayers = len(layerDistances)
    nPadsPerLayer = 4*32
    emap = np.zeros(nLayers*nPadsPerLayer,dtype=emapType)

    # fill z values
    for lidx in range(nLayers):
        emap[:(lidx+1)*nPadsPerLayer]['z'] = layerDistances[lidx]

    # get iu,iv,iU,iV values from emap file and generate x,y
    with open(emapName,'r') as ef:
        for line in ef:
            if 'CHIP' in line: continue
            chip,chan,lay,iU,iV,iu,iv,typ = [int(x) for x in line.split()]
            chipidx = chip-1
            hxidx = lay-1
            x,y = uvToXY(iu,iv,iU,iV)
            arrIdx = hxidx*nPadsPerLayer + chipidx*32 + chan//2
            emap[arrIdx]['iu'] = iu
            emap[arrIdx]['iv'] = iu
            emap[arrIdx]['iU'] = iU
            emap[arrIdx]['iV'] = iV
            emap[arrIdx]['x'] = x
            emap[arrIdx]['y'] = y

    return emap


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


if __name__ == '__main__':

    # get options
    args = getOpts()

    # file setup
    unpackedFileName = args.unpackDir+args.fileName
    print('Unpacked File: {0}'.format(unpackedFileName))
    unpackedFile = h5py.File(unpackedFileName, 'r')
    outFileName = makeOutFileName(args.recoDir,args.fileName)
    print('Reco File: {0}.npz'.format(outFileName))

    # read the data from the input file
    unpackedData = np.array(unpackedFile['event_data'])
    nHexbds = len(unpackedData[0]['hexbd'])
    nPads = nHexbds*4*32 # 32 (active) channels per chip, 4 chips per hexbd

    # read the emap
    emap = createEMap(args.eMap,args.layerDistances)
    # emap = np.zeros(nPads, dtype=emapType)

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
    np.savez_compressed(outFileName, event_data=eventRecoData, allow_pickle=True)

    # close input files
    unpackedFile.close()
