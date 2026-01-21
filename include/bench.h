#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    // CLI (names must be aligned to python; adjust after you paste argparse)
    int iterations;          // valid iterations per model
    int warmup;              // warmup iterations (python prints +100 warmup)
    int payload_bytes;
    const char *aad;
    int seed;
    int ent_payload_mb;
    int ent_iterations;
    bool skip_latency;
    bool skip_ent;

    // Derived
    uint8_t *payload;
    size_t payload_len;
    uint8_t psk20[20];       // for Ascon-80pq PSK-based mode, if needed
} bench_config_t;

void bench_parse_args(int argc, char **argv, bench_config_t *cfg);

typedef struct {
    char Model[64];
    int Iteration;
    uint64_t KeyGen_ns, Encaps_ns, Decaps_ns;
    uint64_t KDF_ns, Encryption_ns, Decryption_ns;
    uint64_t Total_ns;
    double Total_s;
    int Failed;
    long Peak_Alloc_KB;      // Linux ru_maxrss in KB
} csv_row_t;