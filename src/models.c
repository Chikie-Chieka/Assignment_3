#include "models.h"

// forward decls
void run_kyber512_ascon128a(const bench_config_t*, csv_writer_t*);
void run_bike_l1_ascon128a(const bench_config_t*, csv_writer_t*);
void run_x25519_ascon128a(const bench_config_t*, csv_writer_t*);
void run_ascon80pq_tagonly(const bench_config_t*, csv_writer_t*);

void *model_thread_main(void *arg) {
    model_thread_arg_t *a = (model_thread_arg_t*)arg;
    switch (a->kind) {
        case MODEL_KYBER512_ASCON128A: run_kyber512_ascon128a(a->cfg, a->csv); break;
        case MODEL_BIKE_L1_ASCON128A:  run_bike_l1_ascon128a(a->cfg, a->csv); break;
        case MODEL_X25519_ASCON128A:   run_x25519_ascon128a(a->cfg, a->csv); break;
        case MODEL_ASCON80PQ_TAGONLY:  run_ascon80pq_tagonly(a->cfg, a->csv); break;
        default: break;
    }
    return NULL;
}