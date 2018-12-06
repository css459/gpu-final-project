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
samplesize = 95777830
buffersize = 3000000
sample_buffersize = 300000
samples = []

f = "xy.csv"

# Simulated resampling rate
# the amount of buffer fills until
# the buffer is resmapled.
resmapling_rate = 4


def load():
    d = pd.read_csv(f).values
    while len(d) < buffersize:
        d = d + pd.read_csv(f).values
    return d.iloc[:buffersize,:]

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
        resmapling_rate -= 1
        if resmapling_rate <= 0:
            line_buffer, dirty_buffer = make_buffers()
            resmapling_rate = 4

        total_sampled += sample_buffersize

        print(total_sampled / samplesize)


# for i in range(gpus):
        # print("starting gpu", i)

        # i = 0
        # while i < samplesize:
            # s = 0
            # for ss in dirty:
                # s += ss
            # if s >= buffersize:
                # d = pd.read_csv("x.csv", nrows=).values
                # dirty = [0] * 5000000

            # for j in range(buffersize):
                # r = randint(0, 5000000-1)
                # dirty[r] = 1
                # samples.append(d[r])

            # print(i / samplesize)

            # i += buffersize
            # samples = []

