#pragma once
#include "bench.h"
#include "csv.h"

typedef enum {
    MODEL_STANDALONE_ASCON80PQ_TAGONLY,
    MODEL_STANDALONE_BIKE_L1,
    MODEL_STANDALONE_KYBER512,
    MODEL_STANDALONE_FRODOKEM_640_AES,
    MODEL_STANDALONE_HQC_128,
    MODEL_STANDALONE_CLASSIC_MCELIECE_348864,
    MODEL_STANDALONE_X25519,
    MODEL_HYBRID_CLASSIC_MCELIECE_348864_ASCON128A,
    MODEL_HYBRID_FRODOKEM_640_AES_ASCON128A,
    MODEL_HYBRID_HQC_128_ASCON128A,
    MODEL_KYBER512_ASCON128A,
    MODEL_BIKE_L1_ASCON128A,
    MODEL_X25519_ASCON128A,
} model_kind_t;

typedef struct {
    model_kind_t kind;
    const char *name;
    const bench_config_t *cfg;
    csv_writer_t *csv;
} model_thread_arg_t;

void *model_thread_main(void *arg);
