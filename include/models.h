#pragma once
#include "bench.h"
#include "csv.h"

typedef enum {
    MODEL_KYBER512_ASCON128A,
    MODEL_BIKE_L1_ASCON128A,
    MODEL_X25519_ASCON128A,
    MODEL_ASCON80PQ_TAGONLY
} model_kind_t;

typedef struct {
    model_kind_t kind;
    const char *name;
    const bench_config_t *cfg;
    csv_writer_t *csv;
} model_thread_arg_t;

void *model_thread_main(void *arg);