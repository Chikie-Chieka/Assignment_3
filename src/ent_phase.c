#include "bench.h"
#include "util.h"
#include <stdio.h>
// (placeholder; needs your python ENT semantics)
int run_ent_phase(const bench_config_t *cfg, const char *ent_path) {
    (void)cfg;
    FILE *fp = fopen(ent_path, "w");
    if (!fp) return -1;

    // TODO: match python ENT CSV columns exactly.
    fprintf(fp, "Iteration,Total_ns,Total_s,Peak_Alloc_KB\n");
    for (int i = 0; i < 50; i++) {
        uint64_t t0 = now_ns_monotonic_raw();
        // TODO: whatever ENT does in python
        uint64_t t1 = now_ns_monotonic_raw();
        uint64_t ns = t1 - t0;
        fprintf(fp, "%d,%llu,%.9f,%ld\n", i,
                (unsigned long long)ns, (double)ns/1e9, peak_rss_kb());
    }
    fclose(fp);
    return 0;
}