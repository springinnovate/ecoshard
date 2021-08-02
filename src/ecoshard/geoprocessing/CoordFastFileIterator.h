#ifndef SRC_GEOPROCESSING_COORDFASTFILEITERATOR_H_
#define SRC_GEOPROCESSING_COORDFASTFILEITERATOR_H_

#include <stdio.h>
#include <cstddef>
#include <iostream>
#include <fstream>

template <class DATA_T> class CoordFastFileIterator{
 private:
    DATA_T* buffer = NULL;
    long* coord_buffer = NULL;
    char* file_path = NULL;
    char* coord_file_path = NULL;
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
            this->coord_buffer = reinterpret_cast<long*>(malloc(
                this->cache_size * sizeof(long)));
            FILE *fptr = fopen(this->file_path, "rb");
            fseek(fptr, this->global_offset * sizeof(DATA_T), SEEK_SET);

            FILE *coord_fptr = fopen(this->coord_file_path, "rb");
            fseek(coord_fptr, this->global_offset * sizeof(long), SEEK_SET);

            size_t elements_to_read = this->cache_size;
            size_t elements_read = 0;
            while (elements_to_read) {
                fread(
                    reinterpret_cast<void*>(
                        this->coord_buffer+elements_read*sizeof(long)),
                    sizeof(long), elements_to_read, coord_fptr);
                elements_read += fread(
                    reinterpret_cast<void*>(
                        this->buffer+elements_read*sizeof(DATA_T)),
                    sizeof(DATA_T), elements_to_read, fptr);
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
        }
    }

 public:
    CoordFastFileIterator(const char *file_path, const char *coord_file_path, size_t buffer_size) {
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
        if (this->buffer_size > this->file_length) {
            this->buffer_size = this->file_length;
        }
        is.close();

        this->coord_file_path = reinterpret_cast<char*>(malloc(
            (strlen(coord_file_path)+1)*sizeof(char)));
        strncpy(this->coord_file_path, coord_file_path, strlen(coord_file_path)+1);
        update_buffer();
    }

    ~CoordFastFileIterator() {
        if (this->buffer != NULL) {
            free(this->buffer);
            free(this->file_path);
            free(this->coord_buffer);
            free(this->coord_file_path);
        }
    }

    DATA_T const peek() {
        return this->buffer[this->local_offset];
    }

    size_t const size() {
        return this->file_length - (this->global_offset + this->local_offset);
    }

    long const coord() {
        return this->coord_buffer[this->local_offset];
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
    return a->peek() > b->peek();
}

#endif  // SRC_GEOPROCESSING_COORDFASTFILEITERATOR_H_
