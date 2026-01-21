#define _GNU_SOURCE
#include "bench.h"
#include "csv.h"
#include "models.h"
#include "util.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>

int run_ent_phase(const bench_config_t *cfg, const char *ent_path);

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [--iterations N] [--payload_bytes N] [--aad STR] [--seed STR]\n"
        "          [--csv_out PATH] [--skip_latency] [--skip_ent]\n", argv0);
}

int main(int argc, char **argv) {
    bench_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    // Defaults must match python once you paste argparse.
    cfg.iterations = 100;
    cfg.warmup = 100;
    cfg.payload_bytes = 32;
    cfg.aad = "";
    cfg.seed = "0";
    cfg.csv_out = "testing_process.csv";

    static struct option long_opts[] = {
        {"iterations", required_argument, 0, 1},
        {"payload_bytes", required_argument, 0, 2},
        {"aad", required_argument, 0, 3},
        {"seed", required_argument, 0, 4},
        {"csv_out", required_argument, 0, 5},
        {"skip_latency", no_argument, 0, 6},
        {"skip_ent", no_argument, 0, 7},
        {0,0,0,0}
    };

    int opt, idx=0;
    while ((opt = getopt_long(argc, argv, "", long_opts, &idx)) != -1) {
        switch (opt) {
            case 1: cfg.iterations = atoi(optarg); break;
            case 2: cfg.payload_bytes = atoi(optarg); break;
            case 3: cfg.aad = optarg; break;
            case 4: cfg.seed = optarg; break;
            case 5: cfg.csv_out = optarg; break;
            case 6: cfg.skip_latency = true; break;
            case 7: cfg.skip_ent = true; break;
            default: usage(argv[0]); return 2;
        }
    }

    cfg.payload_len = (size_t)cfg.payload_bytes;
    cfg.payload = (uint8_t*)malloc(cfg.payload_len);
    if (!cfg.payload) return 1;
    memset(cfg.payload, 0x42, cfg.payload_len);

    psk20_from_seed_sha256(cfg.seed, cfg.psk20);

    // Phase 1 CSV
    csv_writer_t w;
    if (csv_open(&w, cfg.csv_out) != 0) {
        fprintf(stderr, "Failed to open %s\n", cfg.csv_out);
        free(cfg.payload);
        return 1;
    }
    csv_write_header(&w);

    if (!cfg.skip_latency) {
        pthread_t th[4];
        model_thread_arg_t args[4] = {
            { MODEL_KYBER512_ASCON128A, "ModelA_Kyber512", &cfg, &w },
            { MODEL_X25519_ASCON128A,   "ModelB_X25519",  &cfg, &w },
            { MODEL_BIKE_L1_ASCON128A,  "ModelC_BIKE_L1", &cfg, &w },
            { MODEL_ASCON80PQ_TAGONLY,  "ModelD_Ascon80pq", &cfg, &w },
        };

        for (int i = 0; i < 4; i++) pthread_create(&th[i], NULL, model_thread_main, &args[i]);
        for (int i = 0; i < 4; i++) pthread_join(th[i], NULL);
    }

    csv_close(&w);

    // ENT phase
    if (!cfg.skip_ent) {
        char ent_path[1024];
        snprintf(ent_path, sizeof(ent_path), "%s_ent.csv", cfg.csv_out);
        if (run_ent_phase(&cfg, ent_path) != 0) {
            fprintf(stderr, "ENT phase failed\n");
        }
    }

    free(cfg.payload);
    return 0;
}