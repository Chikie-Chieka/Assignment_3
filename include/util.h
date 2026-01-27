#pragma once
#include <stdint.h>

uint64_t now_ns_monotonic_raw(void);
double effective_ncpu(void);
int perf_cycles_open(void);
uint64_t perf_cycles_read(int fd);
void perf_cycles_close(int fd);

// matches python: sha256(f"psk|{seed}").digest()[:20]
void psk20_from_seed_sha256(const char *seed, uint8_t out20[20]);
