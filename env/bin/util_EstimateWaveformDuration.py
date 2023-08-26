#!/Users/mzeeshan/Desktop/codes/ms-project/env/bin/python3.11

import RIFT.lalsimutils as lalsimutils
import RIFT.misc.ourparams as ourparams
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()
lalsimutils.rosDebugMessagesDictionary            = rosDebugMessagesDictionary


Psig = ourparams.PopulatePrototypeSignal(opts)

timeWaveform = lalsimutils.estimateWaveformDuration(Psig)

print(float(timeWaveform))
