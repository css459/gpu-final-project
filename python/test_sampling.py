#
# "Random" Sampling Simulator
# Cole Smith
# The purpose of this program is only to
# gauge the timings between the real CUDA
# implementation and industry best-practices
# using python and Pandas. Therefore, a few
# heuristics are defined to skip the needless
# computation that would be important in the
# real version, but doesn't affect the timing
# much in this simulated circumstance.
#

import pandas as pd
from random import randint

gpus = 4
samplesize = 79739
buffersize = 3000000
sample_buffersize = 300000
samples = []

f = "test_data/xy.csv"

# Simulated resampling rate
# the amount of buffer fills until
# the buffer is resmapled.
# (-1 means no refresh)
resmapling_rate = -1


def load():
    d = pd.read_csv(f)
    while len(d) < buffersize:
        d = d.append(pd.read_csv(f))
        print(len(d) / buffersize)
    out = d.iloc[:buffersize,:].values
    print(out)
    return out

def make_buffers():
    return load(), [0] * buffersize

print("making initial buffers...")
line_buffer, dirty_buffer = make_buffers()
print("done.")

for i in range(gpus):
    print("starting gpu", i)

    total_sampled = 0
    while total_sampled < samplesize:

        sample_buffer = [0] * sample_buffersize
        for s in range(sample_buffersize):
            # No random sampling here, since we
            # are only concerned with timings in
            # this program (not stat accuracy)
            sample_buffer[s] = line_buffer[s]

            # We are simulating the resampling rate,
            # so we will not be using the dirty buffer
            # here.
            dirty_buffer[s] += 1

        # ----
        # Here is where we would load our hypothetical GPU
        # ----

        # Refresh the buffer if necessary
        if resmapling_rate != -1:
            resmapling_rate -= 1
            if resmapling_rate <= 0:
                line_buffer, dirty_buffer = make_buffers()
                resmapling_rate = 4

        total_sampled += sample_buffersize

        print(total_sampled / samplesize)
