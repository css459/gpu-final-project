#include <stdio.h>
#include "cuda_util.h"
#include "sampling.h"

int main() {

    printf("GPU Mem: %zu\n", get_global_memory_size_for_device(0));
    printf("Samples Possible: %d\n", get_sample_size_for_device(0, 20, sizeof(float)));

    // Set the Random Seed to Time
    srand((unsigned int) time(NULL));

    SamplingProperties props = make_properties(
            "test_data/xy.csv",        // csv_file_path
            11209389,       // file_size
            3000000,         // line_buffer_size
            1000,           // random_chunk_size
            21,             // elements_per_line
            sizeof(float),  // element_size_bytes
            4               // cuda_device_count
            );

    float** cuda_samples = load_devices(&props);

    for (int i = 0; i < props.cuda_device_count; i++) cudaFree((void *) cuda_samples[i]);
    free(cuda_samples);

    return 0;
}
