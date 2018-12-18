# Poject Timeline (For Authors)

# Topics to cover
* Transferring data to multiple GPUs efficiently

    - Transferring data from global to shared memory efficiently in parallel

* How to implement or run Random Forests or decision trees on the GPU with tthe data in global memory

    - Figure out how much of shared memory each block or deciion tree needs to use for variables or temp arrays

* Determine if we would like to **optimize CudaTree** or implement an **improved version**

# TODO: Implement all as serial before moving to CUDA

## DATA TRANSFER (TO DEVICE)
* Generate a large (~100 GB) data set for testing

* Implement reservior sampling to load device with data

* Implement multi-threading or streams to load devices concurrently

## DECISION TREES
* Implement kernel to build a decision tree on a block or thread

* Determinte the amount of temporary memory (space complexity) of decision tree

* Implement as BFS only for simplicity

* Choose only a few hyper parameters for simplicity

* We could leverage CudaTree for this

## RANDOM FOREST / BAG OF TREES
* Decide whether or not to use bag-of-trees or random forests

* Create kernel to run decision tree kernels in parallel

* We could leverage CudaTree for this

## DATA TRANSFER (FROM DEVICE)
* Determine what info needs to be sent back to host

* Determine an async way to do this as devices complete operations

## ADDITIONAL TUNINGS / CONSIDERATIONS
* Parallelize tree-growing / training with transferring data
    (i.e.: While one device is still loading ,the other has started training or while one device is still training, start loading up another device)

* Data efficiency considerations when loading a GPU with data for a second, third, or Nth time

## BENCHMARKING
* Compare data transfer tecniques only (sequential load, parallel multi-device load)

* Compare decision tree training / random forest training only (compare to SKLearn CPU benchmarking (4 cores), CudaTree (maybe?) for GPU)

# Timeline

# Week 1
* Cole: Work on data set creating and implementing reservoir sampling, data loading to single GPU

* Andrew: Research Decision Tree / Random Forest Appoarches (CudaTree, namely) for GPU and CPU and determine if we should use CudaTree as a dependency or implement the way that is described under "Proposal".

# Week 2

* Cole: Work on parallelizing data loading and reservior sampling, data loading to N number of GPUs

* Andrew: Begin working to implement decision tree using approach decided from week 1. Determine the shared memory space complxity (so Cole can figure out how to load global memory into shared memory)

# Week 3

* Cole: Work on K-fold stratified split, GPU-side. Load data into shared memory in the amount that Andrew determined

* Andrew: Work on implementing Random Forest Kernel for single device that calls decision tree kernel

# Week 4

* Cole: PERFORMANCE EVALUATION: Data loading and stratification: Compare to sequential methods

* Andrew: Work on implementing Random Forest Kernel for single device, or apply to multiple devices (on multiple streams)

* Cole: Determine device-to-host transfer of Andrew's tree OOB scores and voting scores

REGROUP TO DETERMINE REST OF PROJECT FLOW (About 6 extra weeks now remain)

# RESERVE 1-2 WEEKS FOR PAPER WRITING, BUG REMOVAL, AND PROFILING


# PROPOSAL

We would like to propose a method for training a grid search of Random Forests on multiple GPUs. When performing a grid search (a set of models with different possible parameters) on a Random Forest, you will mainly be making different style decision trees (different branching criterion, max depth, etc.) or different amount of trees in the forest

Where our project comes in is when we would like to perform this on a data set that is so large, that we could never hope to load it all into RAM, or hope to ever have any single decision tree see every single data point. For this, we propose the following method:

1. Split the data into subsamples that can fit into each GPU’s memory using Replacement Bootstrapping (Reservoir Sampling for unknown data size N) (maybe using OpenMP here to load all GPUs at the same time) to achieve maximum parallelism.

2. From the Grid in the Grid Search (call the set of all model possibilities in the grid T), determine the number of each type of tree you need, and assign each of these trees a membership in one of the grid search possibilities: T_i.

3. Give each GPU a number of trees to grow for each T_i in T, and grow the random forest estimators in parallel on the subsample placed onto that GPU

4. Assign a block to each T_i

5. For each block, perform a k-fold stratified split of the GPU’s loaded data, and load the required data (how much will fit we will need to work out) into shared block memory

6. Now that each block has a train and test set, grow the random forest in each block in parallel for each T_i

7. Each GPU now has a model for each T_i, which is cross validated on different subsamples of the giant data set

8. Move the cross validation scores for each T_i from each GPU, and get the average accuracy of each T_i.

9. Return the trained Random Forest from the GPU with the highest average accuracy, and highest single accuracy (I.e.: If T_1 model had scores 72, 89, 99 from 3 GPUs, with its average score being the highest of all models at 86.33, return the T_1 model from the GPU that calculated the 99 CV score.

