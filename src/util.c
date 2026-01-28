#define _GNU_SOURCE
#include "util.h"
#include <time.h>
#include <sys/resource.h>
#include <openssl/sha.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>

uint64_t now_ns_monotonic_raw(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static long parse_cpuset_count(const char *s) {
    long count = 0;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;
        char *end = NULL;
        long a = strtol(p, &end, 10);
        if (end == p) return -1;
        long b = a;
        p = end;
        if (*p == '-') {
            p++;
            b = strtol(p, &end, 10);
            if (end == p) return -1;
            p = end;
        }
        if (b < a) return -1;
        count += (b - a + 1);
        if (*p == ',') p++;
    }
    return count;
}

double effective_ncpu(void) {
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    double fallback = (ncpu > 0) ? (double)ncpu : 1.0;

    FILE *cg = fopen("/proc/self/cgroup", "r");
    if (!cg) return fallback;

    char line[256];
    char cgroup_path[256] = {0};
    while (fgets(line, sizeof(line), cg)) {
        if (strncmp(line, "0::", 3) == 0) {
            char *path = line + 3;
            char *nl = strchr(path, '\n');
            if (nl) *nl = '\0';
            snprintf(cgroup_path, sizeof(cgroup_path), "%s", path);
            break;
        }
    }
    fclose(cg);
    if (cgroup_path[0] == '\0') return fallback;

    char cpuset_path[512];
    char cpu_max_path[512];
    snprintf(cpuset_path, sizeof(cpuset_path), "/sys/fs/cgroup%s/cpuset.cpus.effective", cgroup_path);
    snprintf(cpu_max_path, sizeof(cpu_max_path), "/sys/fs/cgroup%s/cpu.max", cgroup_path);

    long cpus = -1;
    FILE *cpuf = fopen(cpuset_path, "r");
    if (cpuf) {
        char buf[256];
        if (fgets(buf, sizeof(buf), cpuf)) cpus = parse_cpuset_count(buf);
        fclose(cpuf);
    }
    if (cpus <= 0) cpus = (long)fallback;

    double effective = (double)cpus;
    FILE *maxf = fopen(cpu_max_path, "r");
    if (maxf) {
        char q[64], p[64];
        if (fscanf(maxf, "%63s %63s", q, p) >= 1) {
            if (strcmp(q, "max") != 0) {
                long long quota = atoll(q);
                long long period = (p[0] != '\0') ? atoll(p) : 0;
                if (quota > 0 && period > 0) {
                    double limit = (double)quota / (double)period;
                    if (limit < effective) effective = limit;
                }
            }
        }
        fclose(maxf);
    }
    if (effective <= 0.0) effective = fallback;
    return effective;
}

int perf_cycles_open(void) {
	/*
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 0;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    int fd = (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd < 0) return -1;
    return fd;
*/
    return -1;
}

uint64_t perf_cycles_read(int fd) {
    uint64_t val = 0;
    if (fd >= 0) read(fd, &val, sizeof(val));
    return val;
}

void perf_cycles_close(int fd) {
    if (fd >= 0) close(fd);
}

void psk20_from_seed_sha256(const char *seed, uint8_t out20[20]) {
    uint8_t d[SHA256_DIGEST_LENGTH];
    char buf[512];
    snprintf(buf, sizeof(buf), "psk|%s", seed ? seed : "");
    SHA256((const unsigned char*)buf, strlen(buf), d);
    memcpy(out20, d, 20);
}
