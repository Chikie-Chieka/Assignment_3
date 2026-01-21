#pragma once
#include <stdio.h>
#include <pthread.h>
#include "bench.h"

typedef struct {
    FILE *fp;
    pthread_mutex_t mu;
} csv_writer_t;

int csv_open(csv_writer_t *w, const char *path);
void csv_write_header(csv_writer_t *w);
void csv_write_row(csv_writer_t *w, const csv_row_t *r);
void csv_close(csv_writer_t *w);