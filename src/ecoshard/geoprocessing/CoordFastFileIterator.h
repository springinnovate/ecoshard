#ifndef SRC_GEOPROCESSING_COORDFASTFILEITERATOR_H_
#define SRC_GEOPROCESSING_COORDFASTFILEITERATOR_H_

#include <stdio.h>
#include <cstddef>
#include <iostream>
#include <fstream>

template <class DATA_T> class CoordFastFileIterator{
 private:
    DATA_T* buffer = NULL;
    long long* coord_buffer = NULL;
    double* area_buffer = NULL;
    char* file_path = NULL;
    char* coord_file_path = NULL;
    char* area_file_path = NULL;
    // these offsets and sizes are in numbers of items instead of bytes, or
    // number of bytes / sizeof(DATA_T)
    size_t global_offset;
    size_t local_offset;
    size_t cache_size;
    size_t buffer_size;
    size_t file_length;

    void update_buffer() {
        if (this->local_offset >= this->cache_size) {
            this->global_offset += this->local_offset;
            this->local_offset = 0;
            if (this->buffer_size > (this->file_length - this->global_offset)) {
                this->cache_size = this->file_length - this->global_offset;
            } else {
                this->cache_size = this->buffer_size;
            }
            if (this->buffer != NULL) {
                free(this->buffer);
            }
            this->buffer = reinterpret_cast<DATA_T*>(malloc(
                this->cache_size * sizeof(DATA_T)));
            this->coord_buffer = reinterpret_cast<long long*>(malloc(
                this->cache_size * sizeof(long long)));
            this->area_buffer = reinterpret_cast<double*>(malloc(
                this->cache_size * sizeof(double)));
            FILE *fptr = fopen(this->file_path, "rb");
            fseek(fptr, this->global_offset * sizeof(DATA_T), SEEK_SET);

            FILE *coord_fptr = fopen(this->coord_file_path, "rb");
            fseek(coord_fptr, this->global_offset * sizeof(long long), SEEK_SET);

            FILE *area_fptr = fopen(this->area_file_path, "rb");
            fseek(area_fptr, this->global_offset * sizeof(double), SEEK_SET);

            size_t elements_to_read = this->cache_size;
            size_t elements_read = 0;
            size_t coord_read = 0, area_read = 0;
            while (elements_to_read) {
                coord_read += fread(
                    reinterpret_cast<void*>(
                        this->coord_buffer+elements_read*sizeof(long long)),
                    sizeof(long long), elements_to_read, coord_fptr);
                area_read += fread(
                    reinterpret_cast<void*>(
                        this->area_buffer+elements_read*sizeof(double)),
                    sizeof(double), elements_to_read, area_fptr);
                elements_read += fread(
                    reinterpret_cast<void*>(
                        this->buffer+elements_read*sizeof(DATA_T)),
                    sizeof(DATA_T), elements_to_read, fptr);
                if (coord_read != elements_read) {
                    perror("coord read not equal to elements read");
                }
                if (area_read != elements_read) {
                    perror("area read not equal to elements read");
                }

                if (ferror(fptr)) {
                    perror("error occured");
                    elements_to_read = 0;
                    break;
                } else if (feof(fptr)) {
                    printf("end of file\n");
                    break;
                } else {
                    elements_to_read = this->cache_size - elements_read;
                }
            }
            fclose(fptr);
            fclose(coord_fptr);
            fclose(area_fptr);
        }
    }

 public:
    CoordFastFileIterator(const char *file_path, const char *coord_file_path, const char *area_file_path, size_t buffer_size) {
        global_offset = 0;
        local_offset = 0;
        cache_size = 0;
        this->buffer_size = buffer_size;
        this->file_path = reinterpret_cast<char*>(malloc(
            (strlen(file_path)+1)*sizeof(char)));
        strncpy(this->file_path, file_path, strlen(file_path)+1);
        std::ifstream is(this->file_path, std::ifstream::binary);
        is.seekg(0, is.end);
        this->file_length = is.tellg() / sizeof(DATA_T);
        printf("file length is %d", this->file_length);
        if (this->buffer_size > this->file_length) {
            this->buffer_size = this->file_length;
        }
        is.close();

        this->coord_file_path = reinterpret_cast<char*>(malloc(
            (strlen(coord_file_path)+1)*sizeof(char)));
        strncpy(this->coord_file_path, coord_file_path, strlen(coord_file_path)+1);

        this->area_file_path = reinterpret_cast<char*>(malloc(
            (strlen(area_file_path)+1)*sizeof(char)));
        strncpy(this->area_file_path, area_file_path, strlen(area_file_path)+1);

        update_buffer();
    }

    ~CoordFastFileIterator() {
        if (this->buffer != NULL) {
            free(this->buffer);
            free(this->file_path);
            free(this->coord_buffer);
            free(this->coord_file_path);
            free(this->area_buffer);
            free(this->area_file_path);
        }
    }

    DATA_T const peek() {
        return this->buffer[this->local_offset];
    }

    size_t const size() {
        return this->file_length - (this->global_offset + this->local_offset);
    }

    long long const coord() {
        return this->coord_buffer[this->local_offset];
    }

    double const area() {
        return this->area_buffer[this->local_offset];
    }

    DATA_T next() {
        if (size() > 0) {
            DATA_T val = this->buffer[this->local_offset];
            this->local_offset += 1;
            update_buffer();
            return val;
        } else {
            return -1;
        }
    }
};

template <class DATA_T>
int CoordFastFileIteratorCompare(CoordFastFileIterator<DATA_T>* a,
                            CoordFastFileIterator<DATA_T>* b) {
    return a->peek() < b->peek();
}

#endif  // SRC_GEOPROCESSING_COORDFASTFILEITERATOR_H_
