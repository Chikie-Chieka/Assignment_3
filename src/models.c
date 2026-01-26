#include "models.h"

// forward decls
void run_standalone_bike_l1(const bench_config_t*, csv_writer_t*);
void run_standalone_kyber512(const bench_config_t*, csv_writer_t*);
void run_standalone_frodokem_640_aes(const bench_config_t*, csv_writer_t*);
void run_standalone_hqc_128(const bench_config_t*, csv_writer_t*);
void run_standalone_classic_mceliece_348864(const bench_config_t*, csv_writer_t*);
void run_standalone_x25519(const bench_config_t*, csv_writer_t*);
void run_hybrid_bike_l1_ascon128a(const bench_config_t*, csv_writer_t*);
void run_hybrid_kyber512_ascon128a(const bench_config_t*, csv_writer_t*);
void run_hybrid_classic_mceliece_348864_ascon128a(const bench_config_t*, csv_writer_t*);
void run_hybrid_frodokem_640_aes_ascon128a(const bench_config_t*, csv_writer_t*);
void run_hybrid_hqc_128_ascon128a(const bench_config_t*, csv_writer_t*);
void run_hybrid_x25519_ascon128a(const bench_config_t*, csv_writer_t*);
void run_ascon80pq_tagonly(const bench_config_t*, csv_writer_t*);

void *model_thread_main(void *arg) {
    model_thread_arg_t *a = (model_thread_arg_t*)arg;
    switch (a->kind) {
        case MODEL_STANDALONE_ASCON80PQ_TAGONLY: run_ascon80pq_tagonly(a->cfg, a->csv); break;
        case MODEL_STANDALONE_BIKE_L1: run_standalone_bike_l1(a->cfg, a->csv); break;
        case MODEL_STANDALONE_KYBER512: run_standalone_kyber512(a->cfg, a->csv); break;
        case MODEL_STANDALONE_FRODOKEM_640_AES: run_standalone_frodokem_640_aes(a->cfg, a->csv); break;
        case MODEL_STANDALONE_HQC_128: run_standalone_hqc_128(a->cfg, a->csv); break;
        case MODEL_STANDALONE_CLASSIC_MCELIECE_348864: run_standalone_classic_mceliece_348864(a->cfg, a->csv); break;
        case MODEL_STANDALONE_X25519: run_standalone_x25519(a->cfg, a->csv); break;
        case MODEL_HYBRID_BIKE_L1_ASCON128A: run_hybrid_bike_l1_ascon128a(a->cfg, a->csv); break;
        case MODEL_HYBRID_KYBER512_ASCON128A: run_hybrid_kyber512_ascon128a(a->cfg, a->csv); break;
        case MODEL_HYBRID_FRODOKEM_640_AES_ASCON128A: run_hybrid_frodokem_640_aes_ascon128a(a->cfg, a->csv); break;
        case MODEL_HYBRID_HQC_128_ASCON128A: run_hybrid_hqc_128_ascon128a(a->cfg, a->csv); break;
        case MODEL_HYBRID_CLASSIC_MCELIECE_348864_ASCON128A: run_hybrid_classic_mceliece_348864_ascon128a(a->cfg, a->csv); break;
        case MODEL_HYBRID_X25519_ASCON128A: run_hybrid_x25519_ascon128a(a->cfg, a->csv); break;
        default: break;
    }
    return NULL;
}
