#!/usr/bin/python -O
# run this file with optimizations (-O) so __debug__ is False!

import uhal
import numpy as np
import h5py
import argparse
import datetime
import os
import tqdm



# argument parsing
def getOpts():
    parser = argparse.ArgumentParser(description='IPBus DAQ for HGCal',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nEvents',
                        type=int,
                        default=1000,
                        help='number of events for this run')
    parser.add_argument('--runNum',
                        type=int,
                        default=-1,
                        help='custom run number for this run')
    parser.add_argument('--dataDir',
                        type=str,
                        default='data/raw/',
                        help='directory for the output data')
    return parser.parse_args()

# find current run number
def getRunNumber(dataDir):
    ''' looks at the data directory to see what the run number should be '''
    if not os.path.isfile(dataDir+'RunNumber'):
        with open(dataDir+'RunNumber','w+') as f:
            f.write('1')
        return 1
    else:
        with open(dataDir+'RunNumber','r') as f:
            last = int(f.read(1))
        with open(dataDir+'RunNumber','w') as f: 
            f.write(str(last+1))
        return last+1

# output file name setup
def setupFileName(args):
    ''' creates filename with format DATE_runX.raw.hdf5 '''
    fname = args.dataDir
    fname += datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    fname += '_run{0}.raw.hdf5'.format(args.runNum)
    return fname



# interface to ipbus hardware
class IPBusHWController():
    def __init__(self, hw):
        self._hw = hw
        self.blockSize = 30785

    def _readRegister(self, reg):
        hdr = self._hw.getNode(reg).read()
        self._hw.dispatch()
        while not hdr.valid():
            hdr = self._hw.getNode(reg).read()
            self._hw.dispatch()
        return hdr.value()

    def _readRegisterN(self, reg, size):
        hdr = self._hw.getNode(reg).readBlock(size)
        self._hw.dispatch()
        while not hdr.valid():
            hdr = self._hw.getNode(reg).readBlock(size)
            self._hw.dispatch()
        return hdr.value()

    def _writeRegister(self, reg, val):
        hdr = self._hw.getNode(reg).write(val)
        self._hw.dispatch()
        while not hdr.valid():
            hdr = self._hw.getNode(reg).write(val)
            self._hw.dispatch()

    def getBlockReady(self):
        return np.uint32(self._readRegister('BLOCK_READY'))

    def readFIFO(self):
        return np.array(self._readRegisterN('FIFO',self.blockSize), dtype=np.uint32)

    def getSkirocMask(self):
        return np.uint32(self._readRegister('SKIROC_MASK'))

    def getClockCount(self):
        return (np.uint64(self._readRegister('CLK_COUNT1'))<<np.uint64(32)) | np.uint64(self._readRegister('CLK_COUNT0'))





if __name__ == '__main__':

    # get program options
    args = getOpts()
    if args.runNum == -1:
        args.runNum = getRunNumber(args.dataDir)

    # uhal config
    uhal.setLogLevelTo(uhal.LogLevel.WARNING)

    # setup ipbus connection
    mgr = uhal.ConnectionManager('file://etc/connections.xml')
    hw = mgr.getDevice('hgcal.rdout0')
    rdout = IPBusHWController(hw)

    # open and setup output file
    fname = setupFileName(args)
    print 'Filename: {0}'.format(fname)
    rawFile = h5py.File(fname,'a')
    skiMask = rdout.getSkirocMask()
    rawFile.attrs['skiroc_mask'] = skiMask
    rawFile.attrs['start_time'] = str(datetime.datetime.now())
    eventData = rawFile.create_group('event_data')

    # event loop
    prevClockCount = np.uint64(0)
    error = False
    stopRun = False
    kwargs = {}
    if __debug__: kwargs['disable']=True # stop progress bar during debug
    for trig in tqdm.trange(args.nEvents,**kwargs):
        if error or stopRun: break
        if __debug__: print 'event {0}'.format(trig)

        # wait until we have an event ready to be read out
        if __debug__: print 'waiting for block ready'
        blockReady = rdout.getBlockReady()
        while not blockReady:
            blockReady = rdout.getBlockReady()

        # get the data and clock count
        if __debug__: print 'reading clock count'
        clockCount = rdout.getClockCount() # must read this before you empty the FIFO!
        if __debug__: print 'reading FIFO'
        block = rdout.readFIFO()

        # make sure the clock count incremented
        if clockCount == prevClockCount:
            print 'Clock count did not increment for event {0}!'.format(trig)
            print 'Curr: 0x{0:012x} Prev: 0x{1:012x}'.format(clockCount,prevClockCount)
            error = True
        prevClockCount = clockCount

        # make sure the data looks OK
        # first bits sent from hexbd MAX10 are always 1
        # so the first word read should match the skiroc mask
        if block[0] != skiMask:
            print 'Header/SkirocMask mismatch at event {0}!'.format(trig)
            print 'Header: 0x{0:08x} SkirocMask: 0x{1:08x}'.format(block[0],skiMask)
            error = True

        # save this to the file
        if __debug__: print 'saving event data to file'
        eventData.create_dataset(str(trig), (rdout.blockSize,), dtype='u4', data=block, compression='gzip')
        eventData[str(trig)].attrs['clock_count'] = clockCount

    # closing actions
    rawFile.attrs['end_time'] = str(datetime.datetime.now())
    rawFile.attrs['nevents'] = max([int(x) for x in list(rawFile['event_data'].keys())])+1
    rawFile.attrs['error'] = error
    rawFile.close()
