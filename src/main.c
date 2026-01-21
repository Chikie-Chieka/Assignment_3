#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <pthread.h>
#include "bench.h"
#include "csv.h"
#include "util.h"
#include "models.h" // Requires model_kind_t and model_thread_arg_t

void bench_parse_args(int argc, char **argv, bench_config_t *cfg) {
    // Defaults matching experiment_final.py
    cfg->iterations = 100;
    cfg->warmup = 100;
    cfg->payload_bytes = 4096;
    cfg->aad = "";
    cfg->seed = 1337;
    cfg->ent_payload_mb = 1;
    cfg->skip_latency = false;
    cfg->skip_ent = false;

    static struct option long_options[] = {
        {"iterations", required_argument, 0, 'i'},
        {"payload-bytes", required_argument, 0, 'p'},
        {"aad", required_argument, 0, 'a'},
        {"seed", required_argument, 0, 's'},
        {"ent-payload-mb", required_argument, 0, 'e'},
        {"skip-latency", no_argument, 0, 'L'},
        {"skip-ent", no_argument, 0, 'E'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:p:a:s:e:LEh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i': cfg->iterations = atoi(optarg); break;
            case 'p': cfg->payload_bytes = atoi(optarg); break;
            case 'a': cfg->aad = optarg; break;
            case 's': cfg->seed = atoi(optarg); break;
            case 'e': cfg->ent_payload_mb = atoi(optarg); break;
            case 'L': cfg->skip_latency = true; break;
            case 'E': cfg->skip_ent = true; break;
            case 'h':
                printf("usage: %s [-h] [--iterations ITERATIONS] [--payload-bytes PAYLOAD_BYTES]\n", argv[0]);
                printf("          [--aad AAD] [--seed SEED]\n");
                printf("          [--ent-payload-mb ENT_PAYLOAD_MB] [--skip-latency] [--skip-ent]\n\n");
                printf("optional arguments:\n");
                printf("  -h, --help            show this help message and exit\n");
                printf("  --iterations ITERATIONS\n");
                printf("                        (default: 100)\n");
                printf("  --payload-bytes PAYLOAD_BYTES\n");
                printf("                        (default: 4096)\n");
                printf("  --aad AAD             (default: )\n");
                printf("  --seed SEED           (default: 1337)\n");
                printf("  --ent-payload-mb ENT_PAYLOAD_MB\n");
                printf("                        ENT test payload size in MB (default: 1)\n");
                printf("  --skip-latency        Skip latency/memory benchmarking phase (testing_process.csv + results.json)\n");
                printf("  --skip-ent            Skip ENT randomness testing phase (ENT_Test.csv)\n");
                exit(0);
            default: exit(1);
        }
    }
}

// Forward declaration for the ENT phase entry point
int run_ent_phase(const bench_config_t *cfg, const char *ent_path);

// Helper to run a model thread sequentially for Latency Phase
static void run_latency_model(const bench_config_t *cfg, csv_writer_t *csv, 
                              const char *name, model_kind_t kind) {
    printf("Benchmarking %s... ", name);
    fflush(stdout);

    model_thread_arg_t arg;
    arg.cfg = cfg;
    arg.csv = csv;
    arg.kind = kind;

    // We run in a thread to match the structure, though we join immediately
    pthread_t th;
    if (pthread_create(&th, NULL, model_thread_main, &arg) != 0) {
        fprintf(stderr, "Failed to create thread for %s\n", name);
        return;
    }
    pthread_join(th, NULL);
    printf("✓\n");
}

int main(int argc, char **argv) {
    bench_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    bench_parse_args(argc, argv, &cfg);

    // 1. Setup Global Seed
    srand(cfg.seed);

    // 2. Prepare Payload and PSK
    cfg.payload_len = cfg.payload_bytes;
    cfg.payload = (uint8_t*)malloc(cfg.payload_len);
    if (!cfg.payload) {
        fprintf(stderr, "OOM: payload\n");
        return 1;
    }
    // Fill payload with 0x42 to match Python
    memset(cfg.payload, 0x42, cfg.payload_len);

    // Derive PSK from seed
    char seed_str[32];
    snprintf(seed_str, sizeof(seed_str), "%d", cfg.seed);
    psk20_from_seed_sha256(seed_str, cfg.psk20);

    // 3. Phase 1: Latency Benchmarking
    if (!cfg.skip_latency) {
        printf("Running %d valid iterations per model (+ %d warmup)...\n", cfg.iterations, cfg.warmup);
        printf("Payload: %d bytes\n", cfg.payload_bytes);
        const char *latency_csv_path = "testing_process.csv";
        printf("CSV output: %s\n\n", latency_csv_path);

        csv_writer_t csv;
        if (csv_open(&csv, latency_csv_path) != 0) {
            fprintf(stderr, "Failed to open CSV: %s\n", latency_csv_path);
            free(cfg.payload);
            return 1;
        }
        csv_write_header(&csv);

        run_latency_model(&cfg, &csv, "ModelA_Kyber512", MODEL_KYBER512_ASCON128A);
        run_latency_model(&cfg, &csv, "ModelB_X25519", MODEL_X25519_ASCON128A);
        run_latency_model(&cfg, &csv, "ModelC_BIKE_L1", MODEL_BIKE_L1_ASCON128A);
        run_latency_model(&cfg, &csv, "ModelD_Ascon80pq", MODEL_ASCON80PQ_TAGONLY);

        csv_close(&csv);
        printf("✓ Closed %s (Phase 1 complete)\n\n", latency_csv_path);
    } else {
        printf("⊘ Skipping Phase 1 (latency benchmarking)\n\n");
    }

    // 4. Phase 2: ENT Entropy Testing
    if (!cfg.skip_ent) {
        // Derive ENT filename: <csv_out>_ent.csv
        const char *ent_csv_path = "testing_process_ent.csv";

        printf("Running ENT randomness tests (50 iterations, parallel processing)...\n");
        if (run_ent_phase(&cfg, ent_csv_path) != 0) {
            fprintf(stderr, "ENT phase failed\n");
        } else {
            printf("✓ Closed %s (Phase 2 complete)\n\n", ent_csv_path);
        }
    } else {
        printf("⊘ Skipping Phase 2 (ENT entropy testing)\n\n");
    }

    // Cleanup
    free(cfg.payload);

    printf("Execution Summary:\n");
    printf("Done.\n");
    return 0;
}