#define _GNU_SOURCE
#include "util.h"
#include <time.h>
#include <sys/resource.h>
#include <openssl/sha.h>
#include <string.h>
#include <stdio.h>

uint64_t now_ns_monotonic_raw(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

long peak_rss_kb(void) {
    struct rusage u;
    if (getrusage(RUSAGE_SELF, &u) != 0) return -1;
    return u.ru_maxrss; // Linux: KB
}

void psk20_from_seed_sha256(const char *seed, uint8_t out20[20]) {
    uint8_t d[SHA256_DIGEST_LENGTH];
    char buf[512];
    snprintf(buf, sizeof(buf), "psk|%s", seed ? seed : "");
    SHA256((const unsigned char*)buf, strlen(buf), d);
    memcpy(out20, d, 20);
}