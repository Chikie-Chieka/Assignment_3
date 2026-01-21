#include "bench.h"
#include "csv.h"
#include "util.h"
#include "ascon_api.h"
#include <string.h>

void run_ascon80pq_tagonly(const bench_config_t *cfg, csv_writer_t *csv) {
    int total = cfg->warmup + cfg->iterations;

    for (int i = 0; i < total; i++) {
        csv_row_t r; memset(&r, 0, sizeof(r));
        strncpy(r.Model, "ModelD_Ascon80pq", sizeof(r.Model)-1);
        r.Iteration = i - cfg->warmup;

        uint64_t t_begin = now_ns_monotonic_raw();
        r.Failed = 0;

        // No KEM overhead
        r.KeyGen_ns = 0;
        r.Encaps_ns = 0;
        r.Decaps_ns = 0;

        // KDF: if python derives Ascon80pq key from PSK/seed, do it here.
        // For now: use cfg->psk20 directly and count KDF as 0.
        r.KDF_ns = 0;

        uint8_t tag[16];
        uint64_t te0 = now_ns_monotonic_raw();
        if (ascon80pq_tag16_compute(tag,
                                   cfg->payload, cfg->payload_len,
                                   (const uint8_t*)cfg->aad, strlen(cfg->aad),
                                   cfg->psk20) != 0) {
            r.Failed = 1;
        }
        uint64_t te1 = now_ns_monotonic_raw();

        // “Decryption” for tag-only could be verify; measure similarly if you have it.
        // TODO: implement verify() and measure it; for now set 0.
        r.Encryption_ns = te1 - te0;
        r.Decryption_ns = 0;

        uint64_t t_end = now_ns_monotonic_raw();
        r.Total_ns = t_end - t_begin;
        r.Total_s = (double)r.Total_ns / 1e9;
        r.Peak_Alloc_KB = peak_rss_kb();

        if (i >= cfg->warmup) csv_write_row(csv, &r);
    }
}