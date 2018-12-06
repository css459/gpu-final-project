#!/bin/bash

nvcc src/main.cu src/sampling.cu src/cuda_util.cu -I include && ./a.out
