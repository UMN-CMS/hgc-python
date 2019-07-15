# create sca to time sample conversion map
scaToTS = -np.ones(13, dtype=np.uint8) # start with -1 (aka 255 in uint8)
if ((rollMask>>11) & 1) and (rollMask & 1):
    scaToTS[0] = 12
    for i in xrange(1,13): scaToTS[i] = i-1
else:
    posTrk = -1 # posTrk is the position of the first bit that is set
    for i in xrange(13):
        if (rollMask>>i) & 1:
            posTrk = i
            break
    for i in xrange(1, posTrk+2):
        scaToTS[i] = i + 12 - (posTrk+1)
    for i in xrange(posTrk+2, 12):
        scaToTS[i] = i - 1 - (posTrk+1)
