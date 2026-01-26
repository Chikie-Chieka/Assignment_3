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

typedef struct {
    int id;
    const char *name;
    model_kind_t kind;
} bench_model_entry_t;

static const bench_model_entry_t bench_models[] = {
    {1, "Standalone_Ascon_80pq", MODEL_STANDALONE_ASCON80PQ_TAGONLY},
    {2, "Standalone_BIKE_L1", MODEL_STANDALONE_BIKE_L1},
    {3, "Standalone_Kyber512", MODEL_STANDALONE_KYBER512},
    {4, "Standalone_FrodoKEM_640_AES", MODEL_STANDALONE_FRODOKEM_640_AES},
    {5, "Standalone_HQC_128", MODEL_STANDALONE_HQC_128},
    {6, "Standalone_ClassicMcEliece_348864", MODEL_STANDALONE_CLASSIC_MCELIECE_348864},
    {7, "Standalone_X25519", MODEL_STANDALONE_X25519},
    {8, "Hybrid_BIKE_L1_Ascon128a", MODEL_HYBRID_BIKE_L1_ASCON128A},
    {9, "Hybrid_Kyber512_Ascon128a", MODEL_HYBRID_KYBER512_ASCON128A},
    {10, "Hybrid_FrodoKEM_640_AES_Ascon128a", MODEL_HYBRID_FRODOKEM_640_AES_ASCON128A},
    {11, "Hybrid_HQC_128_Ascon128a", MODEL_HYBRID_HQC_128_ASCON128A},
    {12, "Hybrid_ClassicMcEliece_348864_Ascon128a", MODEL_HYBRID_CLASSIC_MCELIECE_348864_ASCON128A},
    {13, "Hybrid_X25519_Ascon128a", MODEL_HYBRID_X25519_ASCON128A},
};

static const char *bench_model_name(int id) {
    size_t model_count = sizeof(bench_models) / sizeof(bench_models[0]);
    for (size_t i = 0; i < model_count; i++) {
        if (bench_models[i].id == id) return bench_models[i].name;
    }
    return NULL;
}

static void bench_print_help(const char *prog) {
    printf("usage: %s [-h] [--iterations ITERATIONS] [--payload-bytes PAYLOAD_BYTES]\n", prog);
    printf("          [--aad AAD] [--seed SEED]\n");
    printf("          [--ent-payload-mb ENT_PAYLOAD_MB] [--skip-latency] [--skip-ent]\n");
    printf("          [--ent-iterations ENT_ITERATIONS] [--no-csv | --output none]\n");
    printf("          [--model N]\n\n");
    printf("optional arguments:\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --iterations ITERATIONS\n");
    printf("                        (default: 100)\n");
    printf("  --payload-bytes PAYLOAD_BYTES\n");
    printf("                        (default: 4096)\n");
    printf("  --aad AAD             (default: )\n");
    printf("  --seed SEED           (default: 1337)\n");
    printf("  --ent-iterations ENT_ITERATIONS\n");
    printf("                        (default: 50)\n");
    printf("  --ent-payload-mb ENT_PAYLOAD_MB\n");
    printf("                        ENT test payload size in MB (default: 1)\n");
    printf("  --skip-latency        Skip latency/memory benchmarking phase (testing_process.csv + results.json)\n");
    printf("  --skip-ent            Skip ENT randomness testing phase (ENT_Test.csv)\n");
    printf("  --no-csv, --output none\n");
    printf("                        Disable all CSV output files\n");
    printf("  --single-thread {none|full|partial}\n");
    printf("                        none: Phase 1 + Phase 2 use worker threads (default)\n");
    printf("                        full: no worker threads in Phase 1 or Phase 2\n");
    printf("                        partial: Phase 1 sequential, Phase 2 parallel\n");
    printf("  --model N             Run exactly one model:\n");
    printf("                        1=Standalone_Ascon_80pq\n");
    printf("                        2=Standalone_BIKE_L1\n");
    printf("                        3=Standalone_Kyber512\n");
    printf("                        4=Standalone_FrodoKEM_640_AES\n");
    printf("                        5=Standalone_HQC_128\n");
    printf("                        6=Standalone_ClassicMcEliece_348864\n");
    printf("                        7=Standalone_X25519\n");
    printf("                        8=Hybrid_BIKE_L1_Ascon128a\n");
    printf("                        9=Hybrid_Kyber512_Ascon128a\n");
    printf("                        10=Hybrid_FrodoKEM_640_AES_Ascon128a\n");
    printf("                        11=Hybrid_HQC_128_Ascon128a\n");
    printf("                        12=Hybrid_ClassicMcEliece_348864_Ascon128a\n");
    printf("                        13=Hybrid_X25519_Ascon128a\n");
}

void bench_parse_args(int argc, char **argv, bench_config_t *cfg) {
    // Defaults matching experiment_final.py
    cfg->iterations = 100;
    cfg->warmup = 100;
    cfg->payload_bytes = 4096;
    cfg->aad = "";
    cfg->seed = 1337;
    cfg->ent_payload_mb = 1;
    cfg->ent_iterations = 50;
    cfg->skip_latency = false;
    cfg->skip_ent = false;
    cfg->no_csv = false;
    cfg->model_id = 0;
    cfg->single_thread_mode = SINGLE_THREAD_NONE;

    static struct option long_options[] = {
        {"iterations", required_argument, 0, 'i'},
        {"payload-bytes", required_argument, 0, 'p'},
        {"aad", required_argument, 0, 'a'},
        {"seed", required_argument, 0, 's'},
        {"ent-iterations", required_argument, 0, 'I'},
        {"ent-payload-mb", required_argument, 0, 'e'},
        {"skip-latency", no_argument, 0, 'L'},
        {"skip-ent", no_argument, 0, 'E'},
        {"no-csv", no_argument, 0, 'n'},
        {"output", required_argument, 0, 'o'},
        {"single-thread", required_argument, 0, 't'},
        {"model", required_argument, 0, 'm'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:p:a:s:I:e:LEno:t:m:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i': cfg->iterations = atoi(optarg); break;
            case 'p': cfg->payload_bytes = atoi(optarg); break;
            case 'a': cfg->aad = optarg; break;
            case 's': cfg->seed = atoi(optarg); break;
            case 'I': cfg->ent_iterations = atoi(optarg); break;
            case 'e': cfg->ent_payload_mb = atoi(optarg); break;
            case 'L': cfg->skip_latency = true; break;
            case 'E': cfg->skip_ent = true; break;
            case 'n': cfg->no_csv = true; break;
            case 'o':
                if (strcmp(optarg, "none") == 0) {
                    cfg->no_csv = true;
                } else {
                    bench_print_help(argv[0]);
                    exit(1);
                }
                break;
            case 't':
                if (strcmp(optarg, "none") == 0) {
                    cfg->single_thread_mode = SINGLE_THREAD_NONE;
                } else if (strcmp(optarg, "full") == 0) {
                    cfg->single_thread_mode = SINGLE_THREAD_FULL;
                } else if (strcmp(optarg, "partial") == 0) {
                    cfg->single_thread_mode = SINGLE_THREAD_PARTIAL;
                } else {
                    bench_print_help(argv[0]);
                    exit(1);
                }
                break;
            case 'm': {
                int model_id = atoi(optarg);
                if (model_id < 1 || model_id > 13) {
                    bench_print_help(argv[0]);
                    exit(1);
                }
                cfg->model_id = model_id;
                break;
            }
            case 'h':
                bench_print_help(argv[0]);
                exit(0);
            default: exit(1);
        }
    }
}

// Forward declaration for the ENT phase entry point
int run_ent_phase(const bench_config_t *cfg, const char *ent_path);

// Helper to run a model thread sequentially for Latency Phase
static int run_latency_model(const bench_config_t *cfg, csv_writer_t *csv, 
                             const char *name, model_kind_t kind) {
    printf("Benchmarking %s... ", name);
    fflush(stdout);

    model_thread_arg_t arg;
    arg.cfg = cfg;
    arg.csv = csv;
    arg.kind = kind;

    if (cfg->single_thread_mode != SINGLE_THREAD_NONE) {
        model_thread_main(&arg);
        printf("✓\n");
        return 0;
    }

    // We run in a thread to match the structure, though we join immediately
    pthread_t th;
    if (pthread_create(&th, NULL, model_thread_main, &arg) != 0) {
        fprintf(stderr, "Failed to create thread for %s\n", name);
        return -1;
    }
    pthread_join(th, NULL);
    printf("✓\n");
    return 0;
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
        char latency_csv_path[256] = "testing_process.csv";
        if (cfg.model_id != 0) {
            const char *model_name = bench_model_name(cfg.model_id);
            if (model_name) {
                snprintf(latency_csv_path, sizeof(latency_csv_path),
                         "testing_process_%s.csv", model_name);
            }
        }
        if (cfg.no_csv) {
            printf("CSV output: none\n\n");
        } else {
            printf("CSV output: %s\n\n", latency_csv_path);
        }

        csv_writer_t csv;
        csv_writer_t *csvp = NULL;
        if (!cfg.no_csv) {
            if (csv_open(&csv, latency_csv_path) != 0) {
                fprintf(stderr, "Failed to open CSV: %s\n", latency_csv_path);
                free(cfg.payload);
                return 1;
            }
            csv_write_header(&csv);
            csvp = &csv;
        }

        size_t model_count = sizeof(bench_models) / sizeof(bench_models[0]);
        int model_fail = 0;
        for (size_t i = 0; i < model_count; i++) {
            if (cfg.model_id != 0 && cfg.model_id != bench_models[i].id) continue;
            if (run_latency_model(&cfg, csvp, bench_models[i].name, bench_models[i].kind) != 0) {
                model_fail = 1;
            }
        }

        if (!cfg.no_csv) {
            csv_close(&csv);
            printf("✓ Closed %s (Phase 1 complete)\n\n", latency_csv_path);
        } else {
            printf("✓ Phase 1 complete (no CSV)\n\n");
        }
        if (model_fail) {
            fprintf(stderr, "One or more model threads failed to start.\n");
        }
    } else {
        printf("⊘ Skipping Phase 1 (latency benchmarking)\n\n");
    }

    // 4. Phase 2: ENT Entropy Testing
    if (!cfg.skip_ent) {
        // Derive ENT filename: <csv_out>_ent.csv
        char ent_csv_path[256] = "testing_process_ent.csv";
        if (cfg.model_id != 0) {
            const char *model_name = bench_model_name(cfg.model_id);
            if (model_name) {
                snprintf(ent_csv_path, sizeof(ent_csv_path),
                         "testing_process_ent_%s.csv", model_name);
            }
        }

        printf("Running ENT randomness tests (%d iterations, %d MB payload each)...\n", cfg.ent_iterations, cfg.ent_payload_mb);
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
