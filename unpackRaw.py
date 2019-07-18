#!/usr/bin/python3 -O

import numpy as np
import h5py
import argparse
import tqdm



def getOpts():
    parser = argparse.ArgumentParser(description='HGCal Data Unpacker',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fileName',
                        type=str,
                        help='file to unpack')
    parser.add_argument('--rawDir',
                        type=str,
                        default='data/raw/',
                        help='directory to look for the raw data')
    parser.add_argument('--unpackDir',
                        type=str,
                        default='data/unpacked/',
                        help='directory for the output data')
    parser.add_argument('--keepTime',
                        type=bool,
                        default=False,
                        help='keep the datestamp in the filename')
    return parser.parse_args()

def makeOutFileName(dirName, rawFileName, keepTime):
    outFileName = dirName
    if keepTime:
        dateStamp = rawFileName.split('_run')[0]
        outFileName += dateStamp+'_'
    runNum = rawFileName.split('_run')[1].rstrip('.raw.hdf5')
    outFileName += 'run'+runNum
    outFileName += '.unpacked.hdf5'
    return outFileName


# gray to binary for ADC conversion
def grayToBinary12Bit(num):
    num ^= (num >> 8)
    num ^= (num >> 4)
    num ^= (num >> 2)
    num ^= (num >> 1)
    return num

def grayToBinary32Bit(num):
    num ^= (num >> 16)
    num ^= (num >> 4)
    num ^= (num >> 2)
    num ^= (num >> 1)
    return num



# numpy datatypes
chanType = np.dtype([
        ('hg_adc', 'u2', (13,)), ('hg_adc_hit', '?', (13,)),
        ('lg_adc', 'u2', (13,)), ('lg_adc_hit', '?', (13,)),
        ('toa_fall', 'u2'), ('toa_fall_hit', '?'),
        ('toa_rise', 'u2'), ('toa_rise_hit', '?'),
        ('tot_slow', 'u2'), ('tot_slow_hit', '?'),
        ('tot_fast', 'u2'), ('tot_fast_hit', '?'),
    ])
chipType = np.dtype([
        ('chan', chanType, (64,)),
        ('roll_position', 'u2'),
        ('global_ts', 'u4'),
        ('chip_id', 'u1'),
    ])
hexbdType = np.dtype([
        ('chip', chipType, (4,)),
        ('index', 'u1'),
    ])





if __name__ == '__main__':

    # get options
    args = getOpts()

    # file setup
    rawFileName = args.rawDir+args.fileName
    print('Raw File: {0}'.format(rawFileName))
    rawFile = h5py.File(rawFileName,'r')
    outFileName = makeOutFileName(args.unpackDir,args.fileName,args.keepTime)
    print('Unpacked File: {0}'.format(outFileName))
    unpackedFile = h5py.File(outFileName,'w')

    # find which hexaboards were connected
    hexbds = []
    for hx in range(8):
        if (rawFile.attrs['skiroc_mask']>>(hx*4))&0xf == 0xf:
            hexbds.append(hx)

    # create event dtype and array
    eventType = np.dtype([
            ('hexbd', hexbdType, len(hexbds)), # need to know number of hexbds before creating the event type
        ])
    eventData = np.zeros(rawFile.attrs['nevents'], dtype=eventType)

    # event loop
    for evidx in tqdm.trange(rawFile.attrs['nevents']):

        # read the block from file
        block = np.array(rawFile['/raw_events/{0}'.format(evidx)],dtype=np.uint32)

        # unpack hexaboard data into the bitstreams from each skiroc
        # each skiroc should produce an array of 1924 16-bit words
        for hxidx,hx in enumerate(hexbds):
            # set the hexbd index
            eventData[evidx]['hexbd'][hxidx]['index'] = hx

            # get the data from the hexbd we want
            hxData = ((block>>hx) & 0xf)[1:] # drop the header

            # loop over skirocs to unpack their data from the FIFO blocks
            skiWords = np.zeros((4,1924), dtype=np.uint16)
            for ski in range(4):
                skiBits = ((hxData>>ski) & 0x1).reshape((1924,16)) # reshape the array to look like an array of 1924 16-bit words
                # create the 16-bit words by 'bitwise-or'ing the 16 bits together from left to right
                # FIFO bits are MSB first, so we must fill the word from left to right (15-bitidx)
                for bitidx in range(16):
                    skiWords[ski,:] |= skiBits[:,bitidx] << (15-bitidx)

            # apply gray to binary conversion and bit shifts in batch for better speed
            skiWords12BitBinary = grayToBinary12Bit(skiWords & 0xfff)
            skiWordsHitBit = (skiWords>>12) & 1

            # fill the data for each skiroc
            for ski in range(4):
                # fill channel ADC data
                for sca in range(13): # loop over sca first for a better memory access pattern (in theory)
                    for chan in range(1,64,2): # only even channels are connected (loop over odds, because they are flipped in the fifo array)
                        # channel 64 comes out of the hexbd first, so we use 63-chan to fill the arrays (same reason as range(1,64,2) above)
                        eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['lg_adc'][sca] = skiWords12BitBinary[ski][sca*2*64+0*64+chan]
                        eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['lg_adc_hit'] = skiWordsHitBit[ski][sca*2*64+0*64+chan]
                        eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['lg_adc'][sca] = skiWords12BitBinary[ski][sca*2*64+1*64+chan]
                        eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['hg_adc_hit'] = skiWordsHitBit[ski][sca*2*64+1*64+chan]

                # fill channel TOT/TOA data
                offset = 13*2*64
                for chan in range(1,64,2):
                    eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['toa_fall'] = skiWords12BitBinary[ski][offset+chan]
                    eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['toa_fall_hit'] = skiWordsHitBit[ski][offset+chan]
                    eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['toa_rise'] = skiWords12BitBinary[ski][offset+64+chan]
                    eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['toa_rise_hit'] = skiWordsHitBit[ski][offset+64+chan]
                    eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['tot_slow'] = skiWords12BitBinary[ski][offset+2*64+chan]
                    eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['tot_slow_hit'] = skiWordsHitBit[ski][offset+2*64+chan]
                    eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['tot_fast'] = skiWords12BitBinary[ski][offset+3*64+chan]
                    eventData[evidx]['hexbd'][hxidx]['chip'][ski]['chan'][63-chan]['tot_fast_hit'] = skiWordsHitBit[ski][offset+3*64+chan]

                # fill chip roll position, chip id, and global ts
                eventData[evidx]['hexbd'][hxidx]['chip'][ski]['roll_position'] = skiWords[ski][-4] & 0x1fff
                eventData[evidx]['hexbd'][hxidx]['chip'][ski]['global_ts'] = grayToBinary32Bit(((skiWords[ski][-3]&0x3fff)<<12)|(skiWords[ski][-2]&0xfff))

    # done with raw data
    rawFile.close()

    # save the unpacked data to a compressed hdf5 file
    eventDataset = unpackedFile.create_dataset('unpacked_events', data=eventData, compression='gzip')
    unpackedFile.close()
