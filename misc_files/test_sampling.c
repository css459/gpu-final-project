#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LINE_BUFFER_SIZE 2048
#define FILE_BUFFER_SIZE 2048 * 10000

int seek_nearest_newline(int loc, FILE *fp) {

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
    while(1) {

        // Seek Backward
        fseek(fp, loc - offset, SEEK_SET);
        int c = fgetc(fp);
        if ( c == '\n') {
            sign = -1;
            break;
        }

        // Seek Forward
        fseek(fp, loc + offset, SEEK_SET);
        c = fgetc(fp);
        if ( c == '\n') {
            break;
        }

        offset += 1L;
    }

    return loc + (sign * offset) + 1;
}

void get_line(float *floats, FILE *fp, int loc, int line_size) {
    int i = 0;

    // Read Line and Tokenize
    char line[line_size];
    fseek(fp, loc, SEEK_SET);
    fgets(line, line_size, fp);

    char *tok = strtok(line, ",");

    // Read tokens into floats array
    while(tok != NULL) {
        floats[i++] = atof(tok);
        tok = strtok(NULL, ",");
    }
}

void random_sample(float* buffer[], unsigned int elements_per_line, unsigned int buffer_size, FILE *fp, unsigned int file_size) {
    for (unsigned int i = 0; i < buffer_size; i++) {

        // Go to a random line in the file and seek nearest new line
        int r = rand() % (file_size + 1);

        // Find nearest new line at r
        int nl = seek_nearest_newline(r, fp);

        // Get the line starting at this index
        float *sample = (float *) malloc(elements_per_line * sizeof(float));
        get_line(sample, fp, nl, LINE_BUFFER_SIZE);

        buffer[i] = sample;
    }
}

// Skip around a CSV file randomly in "random_chunk_size" intervals and load lines of the file into a line buffer
char** buffer_csv(const char *csv_file_path, unsigned int line_buffer_size, unsigned int random_chunk_size, unsigned int elements_per_line, unsigned long file_size) {

    // Load File
    FILE *fp;
    fp = fopen(csv_file_path, "r");

    // Init Buffer
    char **line_buffer = (char **) malloc(line_buffer_size * sizeof(char *));

    // Loop and load chunks into line buffer
    int r = rand() % (file_size + 1);
    for (unsigned int k = 0; k < line_buffer_size; k++) {

        if (k % random_chunk_size == 0) {
            // Pick a random spot and read in data until EOF or buffer full
            r = rand() % (file_size + 1);

            // Find nearest new line at r
            int nl = seek_nearest_newline(r, fp);
            fseek(fp, nl, SEEK_SET);
        }

        char * line = malloc(LINE_BUFFER_SIZE * sizeof(char));
        if (fgets(line, LINE_BUFFER_SIZE, fp) != NULL) {
            line_buffer[k] = line;
            /*printf("%d\n", k);*/
        }
    }
    
    fclose(fp);
    return line_buffer;
}

void dealloc_csv_buffer(char ** buffer, int buffer_size) {
    for (int i = 0; i < buffer_size; i++) free(buffer[i]);
    free(buffer);
}

// Returns a list of random rows from the buffer of size "samples"
float** random_subsample(char **buffer, unsigned int samples, unsigned int elements_per_line) {

    float **sample_buffer = (float **) malloc(samples * sizeof(float *));

#pragma omp parallel for
    for (unsigned int i = 0; i < samples; i++) {

        // Get a random point in the buffer
        int rand_sample = rand() % (samples + 1);
        char *tok = strtok(buffer[rand_sample], ",");

        float *sample = (float *) malloc(elements_per_line * sizeof(float));

        // Read tokens into samples array
        int j = 0;
        while(tok != NULL) {
            sample[j++] = atof(tok);
            tok = strtok(NULL, ",");
        }

        sample_buffer[i] = sample;
    }

    return sample_buffer;
}

void dealloc_subsample(float ** buffer, int buffer_size) {
    for (int i = 0; i < buffer_size; i++) free(buffer[i]);
    free(buffer);
}

int main () {

    // Set the Random Seed to Time
    srand(time(NULL));

    int lines_in_line_buffer = 300000;
    int random_chunk_size = 1000;
    int elements_per_line = 20;
    int approx_file_size = 11209389;
    
    char **buffer = buffer_csv("../x.csv", lines_in_line_buffer, random_chunk_size, elements_per_line, approx_file_size);
    
    float **samples = random_subsample(buffer, 10000, 20);


    for (int i = 0; i < 20; i++) {
        printf("%f\n", samples[0][i]);
    }
    
    dealloc_csv_buffer(buffer, 1000);
    dealloc_subsample(samples, 1);

    return(0);
}

