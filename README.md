# GPU Final Project
### Cole Smith
### Andrew Dobroshynskyi

## Compiling

Your GCC compiler must support OpenMP, and CUDA must be install,
and of a recent version of at least CUDA 6.

To compile and run this project, simply run `./build.sh`. This will
load __4__ GPUs with the proper amount of data to 1% occupancy. It will
then run a Grid Search of Random Forests across all GPUs.

## Tuning

The following parameters can be tuned in `src/sampling.cu`:

    #define LINE_BUFFER_SIZE 2056
    #define GLOBAL_MEM_RESERVATION_PERCENT 0.999
    #define SAMPLE_BUFFER_SIZE_ELEMENTS 300000
    #define DIRTY_TOLERANCE 0.70
    #define IGNORE_PRINTF

The default values work well for `cuda5` and the choice of
only 0.1% occupancy was done as a courtesy to fellow users
of the cluster. The buffer sizes should be tuned to use the maximum
amount of RAM you are willing to occupy with these intermediate buffers.

Keep in mind that there will be a separate sample-buffer for __every GPU__
you wish to load if running this code with OpenMP.

Additionally, the following properties list can be tuned in `src/main.cu`
which dictates certain aspects about how you want the program to read the
input CSV file. These are tunde to work with the mini test CSV file `xy.csv`
provided in this project.

    SamplingProperties props = make_properties(
            "test_data/xy.csv",        // csv_file_path
            11209389,                  // file_size
            3000000,                   // line_buffer_size
            1000,                      // random_chunk_size
            21,                        // elements_per_line
            sizeof(float),             // element_size_bytes
            4                          // cuda_device_count
            );

## Data Layout

This project assumes a single binary classifier put into the last position
of the CSV file to load. As in, the last column should be only __0__ or __1__
for each row.

