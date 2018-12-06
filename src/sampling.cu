//
// Created by cole on 10/28/18.
//

#include "sampling.h"
#include "cuda_util.h"

#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// MIN() Helper Function
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// The buffer in bytes of the block that will store a line
// in the line buffer. This should be sufficiently large, but
// not needlessly so
#define LINE_BUFFER_SIZE 2056

// The amount of memory (in percent) of global memory
// to save for other purposes in each GPU
#define GLOBAL_MEM_RESERVATION_PERCENT 0.999

// The size of the second intermediate buffer in
// element count for each data packet send to a GPU
#define SAMPLE_BUFFER_SIZE_ELEMENTS 300000

// The percent of the line buffer that is marked dirty
// before forcing a refresh of the line buffer
#define DIRTY_TOLERANCE 0.70

// Ignore print statements
//#define IGNORE_PRINTF
#ifdef IGNORE_PRINTF
#define printf(fmt, ...) (0)
#endif

// Method to check for CUDA errors
#define cudaCheckError(err) {                                                                    \
    if (err != cudaSuccess) {                                                                    \
        fprintf(stderr,"[ ERR ] CUDA: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                                                                 \
    }                                                                                            \
}

//
// Sampling Properties
//

SamplingProperties make_properties(const char *csv_file_path, unsigned long file_size,
        int line_buffer_size, int random_chunk_size, int elements_per_line,
        size_t element_size_bytes, int cuda_device_count)
{
    SamplingProperties props;
    props.csv_file_path = csv_file_path;
    props.file_size = file_size;
    props.line_buffer_size = line_buffer_size;
    props.random_chunk_size = random_chunk_size;
    props.element_size_bytes = element_size_bytes;
    props.elements_per_line = elements_per_line;
    props.cuda_device_count = cuda_device_count;

    return props;
}

//
// File IO Functions
//

unsigned long seek_nearest_newline(unsigned long loc, FILE *fp) {

    // Begin seek from loc
    fseek(fp, loc, SEEK_SET);

    // Check if this value is a newline
    fseek(fp, loc - 1L, SEEK_SET);
    if (fgetc(fp) == '\n') {
        return loc;
    }

    int sign = 1;
    int offset = 1;

    // Loop until we find a new line
    while (1) {

        // Seek Backward
        fseek(fp, loc - offset, SEEK_SET);
        int c = fgetc(fp);
        if (c == '\n') {
            sign = -1;
            break;
        }

        // Seek Forward
        fseek(fp, loc + offset, SEEK_SET);
        c = fgetc(fp);
        if (c == '\n') {
            break;
        }

        offset += 1L;
    }

    return loc + (sign * offset) + 1;
}

//
// CSV File Buffer
//

// Skip around a CSV file randomly in "random_chunk_size" intervals and load lines of the file into a line buffer
char **buffer_csv(const char *csv_file_path, int line_buffer_size, int random_chunk_size, unsigned long file_size) {

    // Load File
    FILE *fp;
    fp = fopen(csv_file_path, "r");

    // Init Buffer
    char **line_buffer = (char **) malloc(line_buffer_size * sizeof(char *));

    // Loop and load chunks into line buffer
    unsigned long r = rand() % (file_size + 1);
#pragma omp parallel for
    for (int k = 0; k < line_buffer_size; k++) {

        if (k % random_chunk_size == 0) {
            // Pick a random spot and read in data until EOF or buffer full
            r = rand() % (file_size + 1);

            // Find nearest new line at r
            unsigned long nl = seek_nearest_newline(r, fp);
            fseek(fp, nl, SEEK_SET);
        }

        char *line = (char *) malloc(LINE_BUFFER_SIZE * sizeof(char));
        if (fgets(line, LINE_BUFFER_SIZE, fp) == NULL) {
            printf("[ ERR ] Could not allocate memory for line in line buffer\n");
            return NULL;
        }

        line_buffer[k] = line;
    }

    fclose(fp);
    return line_buffer;
}

int* make_dirty_buffer(int line_buffer_size) {
    int *dirty = (int *) calloc(line_buffer_size, sizeof(int));
    if (dirty != NULL) {
        return dirty;
    } else {
        printf("[ ERR ] Unable to alloc dirty buffer\n");
        return 0;
    }
}

void refresh_buffer(const char *csv_file_path, int line_buffer_size, int random_chunk_size,
        unsigned long file_size, char ***line_buffer, int **dirty_buffer, int num_devices) {

    char **l = *line_buffer;
    int *d = *dirty_buffer;

    // Get the needed amount of new data
    int sum = 0;
#pragma omp parallel for reduction (+:sum)
    for (int i = 0; i < line_buffer_size; i++)
        if (d[i] >= num_devices)
            sum = sum + 1;

    printf("Refreshes Needed:%d\n", sum);

    char **refresh = buffer_csv(csv_file_path, sum, random_chunk_size, file_size);

    int r = 0;
#pragma omp parallel for
    for(int i = 0; i < line_buffer_size; i++) {
        if (d[i] >= num_devices) {
            free(l[i]);
            l[i] = refresh[r];
            d[i] = 0;
#pragma omp atomic
            r++;
        }
    }

    free(refresh);
}


void dealloc_csv_buffer(char **buffer, int *dirty_buffer, int buffer_size) {
    for (int i = 0; i < buffer_size; i++) free(buffer[i]);
    free(buffer);
    free(dirty_buffer);
}

//
// Random Sub-Sampling
//

bool random_subsample(float **output_sample_buffer, char **buffer, int *dirty_buffer, int fill_start, int line_buffer_size,
        int samples, int elements_per_line, int num_devices)
{
    // Create the output_sample_buffer
    *output_sample_buffer = (float *) malloc(samples * elements_per_line * sizeof(float));
    if (*output_sample_buffer == NULL) {
        printf("[ ERR ] Could not make sample buffer.\n");
        return -1;
    }

    // Counts the number of duplicated values extracted from the sample buffer
    // If this value exceeds the DIRTY_TOLERANCE then the function will break
    // execution and return the i-th iteration that the execution stopped on
    //
    // The mechanism here is that, once the dirty hit tolerance is reached for all
    // buffer readers, then the buffer will be refreshed.

    int dirty_hits = 0;
    bool needs_refresh = false;

#pragma omp parallel for
    for (int i = fill_start; i < samples; i++) {

        // Get a random point in the buffer
        int rand_sample = rand() % (line_buffer_size);
        char *sample = (char *) malloc(sizeof(char) * LINE_BUFFER_SIZE);
        strcpy(sample, buffer[rand_sample]);

        // If we reach our tolerance, refresh buffer on next invocation.
        if (dirty_hits >= (DIRTY_TOLERANCE * samples) && !needs_refresh) {
            needs_refresh = true;
            printf("[ WRN ] Needs refresh after %d cycles, %d dirty hits\n", i, dirty_hits);
        }

        // Detect duplicate hit
        if (dirty_buffer[rand_sample] >= num_devices) {
            dirty_hits++;
            i--;
            continue;
        }

#pragma omp atomic
        dirty_buffer[rand_sample]++;

        // Read tokens into samples array
        int j = 0;
        char *tok;
        while((tok = strsep(&sample, ",")) != NULL) {
            int indexer = (i * elements_per_line) + j++;
            (*output_sample_buffer)[indexer] = (float) atof(tok);
        }
        free(sample);
        free(tok);
    }

    printf("Dirty hits:%d\n", dirty_hits);

    // Return refresh indicator (samples)
    return needs_refresh;
}

//
// CUDA Device Loading
//

int get_sample_size_for_device(int device_number, int elements_per_sample, size_t element_size_bytes) {
    float occupancy = (float) (1.0 - GLOBAL_MEM_RESERVATION_PERCENT);
    size_t total_global_mem_size = get_global_memory_size_for_device(device_number);

    size_t size_to_alloc = (size_t) (total_global_mem_size * occupancy);
    size_t samples_possible = size_to_alloc / (elements_per_sample * element_size_bytes);

    return (unsigned int) samples_possible;
}

float** load_devices(SamplingProperties *props) {

    int num_devices = props->cuda_device_count;

    // Get number of samples per device, store in SamplingProperties
    int *samples_per_device = (int *) malloc(sizeof(int) * num_devices);
    if (samples_per_device == NULL) {
        printf("[ ERR ] Could not allocate required memory: load_devices\n");
        return NULL;
    }

    for (int i = 0; i < num_devices; i++) {
        samples_per_device[i] = get_sample_size_for_device(i, props->elements_per_line, props->element_size_bytes);
    }

    // Assign this property back to the struct for the user
    props->samples_per_device = samples_per_device;

    // Init CSV row buffer and dirty buffer array
    char **buffer = buffer_csv(props->csv_file_path, props->line_buffer_size, props->random_chunk_size, props->file_size);
    int *dirty = make_dirty_buffer(props->line_buffer_size);

    // Track the number of threads who have finished loading
    int done_threads = 0;

    // Store a flag that threads may raise when they request
    // the line buffer cache be refreshed (from too many dirty hits).
    bool refresh_cache = false;

    // For the following arrays, each entry corresponds
    // to the device number "i"

    // Store the last value that the sub sampling stopped at
    // This will be important for refreshing the cache or writing
    // sub-sub samples to the GPU
    int stopped_at[num_devices];

    // Store the number of samples to go for each GPU until it's totally filled
    int samples_to_go[num_devices];

    // Create CUDA streams for async transfer on each thread
    cudaStream_t streams[num_devices];

    // Store the pointers to CUDA memory space where
    // the data sets will be stored
    float **cuda_data = (float **) malloc(num_devices * sizeof(float *));
    if (cuda_data == NULL) {
        printf("Could not alloc space for CUDA samples\n");
        return NULL;
    }

    // Intialize to default values
    for (int i = 0; i < num_devices; i++) {
        stopped_at[i] = 0;
        /*cuda_last_load[i] = 0;*/
        samples_to_go[i] = samples_per_device[i];

        // Alloc CUDA Space
        float *d;
        size_t memsize = samples_per_device[i] * props->elements_per_line * sizeof(float);
        cudaError_t err = cudaMalloc((void **) &d, memsize);
        cudaCheckError(err);
        cuda_data[i] = d;

        // Make CUDA streams
        cudaStream_t s;
        cudaStreamCreate(&s);
        streams[i] = s;
    }

    // omp_set_dynamic(1);
    // omp_set_num_threads(12);

    // Continue loading and refreshing until all threads are complete
    while (done_threads < num_devices) {

        // For each device, start a thread and begin creating sub samples from
#pragma omp parallel for
        for (int i = 0; i < num_devices; i++) {

            if (refresh_cache) break;

            /*unsigned int sample_size = samples_per_device[i];*/
            int sample_buffer_size = MIN(SAMPLE_BUFFER_SIZE_ELEMENTS, samples_to_go[i]);

            // float *interm_buffer;
            float *interm;

            // Alloc sub sub sample
            bool refresh = random_subsample(&interm, buffer, dirty, 0, props->line_buffer_size,
                    sample_buffer_size, props->elements_per_line, num_devices);

            for (int k = 0; k < 21; k++ ) printf("%f ", interm[k]);
            printf("\n");

            // ----------------------------------------------------------------------------------------------------
            // LOAD GPU MEM
            // ----------------------------------------------------------------------------------------------------
            printf("(%d) Pushing to GPU...\n", i);
            cudaError_t err;

            err = cudaSetDevice(i);
            cudaCheckError(err);

            /*float *dest = cuda_data[i] + cuda_last_load[i];*/
            float *dest = cuda_data[i] + stopped_at[i];
            float *src  = interm;
            size_t memsize = sample_buffer_size * props->elements_per_line * sizeof(float);

            err = cudaMemcpyAsync((void *) dest, (void *) src, memsize, cudaMemcpyHostToDevice, streams[i]);
            cudaCheckError(err);
            err = cudaStreamSynchronize(streams[i]);
            cudaCheckError(err);

            printf("(%d) Done.\n", i);
            // ----------------------------------------------------------------------------------------------------

            // Update refresh cache flag and last written location
            refresh_cache = refresh_cache || refresh;
            stopped_at[i] += sample_buffer_size;
            samples_to_go[i] -= sample_buffer_size;

            // Free temp 2nd buffer
            free(interm);

            if (stopped_at[i] >= samples_per_device[i]) {
#pragma omp atomic
                done_threads++;
                cudaStreamDestroy(streams[i]);
            }

            printf("(%d) Stopped at: %d out of %d\n", i, stopped_at[i], samples_per_device[i]);
        }

        // If all threads are done, exit
        // otherwise, refresh the cache and go again
        if (done_threads == num_devices) {
            printf("All threads done!\n");
            break;
        } else if (refresh_cache) {
            printf("Threads refused to finish, refreshing cache...\n");

            refresh_buffer(props->csv_file_path, props->line_buffer_size, props->random_chunk_size,
                    props->file_size, &buffer, &dirty, num_devices);

            refresh_cache = false;
            printf("Done\n");
        } else {}
    }

    // Cleanup Memory
    dealloc_csv_buffer(buffer, dirty, props->line_buffer_size);

    return cuda_data;
}

