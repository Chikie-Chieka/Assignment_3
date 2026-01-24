#pragma once
#include <stdint.h>

uint64_t now_ns_monotonic_raw(void);
long peak_rss_kb(void);
long current_rss_kb(void);
double effective_ncpu(void);

// matches python: sha256(f"psk|{seed}").digest()[:20]
void psk20_from_seed_sha256(const char *seed, uint8_t out20[20]);
