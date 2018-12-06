//
// Created by cole on 10/28/18.
//

#ifndef GPU_FINAL_PROJECT_SAMPLING_H
#define GPU_FINAL_PROJECT_SAMPLING_H

//
// Sampling Properties
//

struct sampling_properties {
    const char *csv_file_path;
    int line_buffer_size;
    int random_chunk_size;
    unsigned long file_size;
    size_t element_size_bytes;
    int elements_per_line;
    int cuda_device_count;
    int *samples_per_device;
};

typedef struct sampling_properties SamplingProperties;

SamplingProperties make_properties(const char *csv_file_path, unsigned long file_size,
        int line_buffer_size, int random_chunk_size, int elements_per_line,
        size_t element_size_bytes, int cuda_device_count);

//SamplingProperties make_properties(const char *csv_file_path, unsigned long file_size,
        //unsigned int line_buffer_size, unsigned int random_chunk_size, unsigned int elements_per_line,
        //size_t element_size_bytes, unsigned int cuda_device_count);

//
// Buffers
//

// char **buffer_csv(const char *csv_file_path, unsigned int line_buffer_size,
//         unsigned int random_chunk_size, unsigned long file_size);
// 
// bool* make_dirty_buffer(unsigned int line_buffer_size);
// 
// void dealloc_csv_buffer(char **buffer, bool *dirty_buffer, int buffer_size);

//
// Sub Sampling
//

/*
 * Randomly draws a sub sample from the row buffer (char **buffer) into
 * output_sample_buffer. Reads from the dirty_buffer whether or not a random
 * sample has already been accessed by a reader. Dirty hits (duplicates) are
 * tolerated up to DIRTY_TOLERANCE, above which the function will fail to fill
 * the rest of the output_sample_buffer, requiring that the csv row buffer be
 * refreshed.
 *
 * Upon dirty tolerance failture, the function returns the i-th iteration,
 * where 0 <= i < samples, of where to resume filling output_sample_buffer
 * Upon memory error, reutrns -1
 * Upon success, returns samples (unsigned int samples)
 *
 * Parameters
 *      output_sample_buffer : The sub sample drawn from buffer, if NULL,
 *                             will be created here.
 *      buffer               : The csv row buffer from which to draw a random
 *                             sample
 *      dirty_buffer         : Boolean array index-aligned the buffer which
 *                             indicates whether or not the row in buffer has
 *                             already been read by a buffer reader
 *      fill_start           : Where to resume in filling output_sample_buffer
 *                             This should be 0 when first calling
 *      line_buffer_size     : The number of rows (elements) in buffer
 *      samples              : The number of rows (elements) in output_sample_buffer
 *      elements_per_line    : The number of floats per row in buffer (data dimensions)
 */
// bool random_subsample(float ***output_sample_buffer, char **buffer, bool *dirty_buffer, int *dirty_hits,
//         unsigned int fill_start, unsigned int line_buffer_size,
//         unsigned int samples, unsigned int elements_per_line);
// 
// void dealloc_subsample(float **buffer, int buffer_size);

//
// GPU Device Functions
//

int get_sample_size_for_device(int device_number, int elements_per_sample, size_t element_size_bytes);

float** load_devices(SamplingProperties *props);

#endif //GPU_FINAL_PROJECT_SAMPLING_H
