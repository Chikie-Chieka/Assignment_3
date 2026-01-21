#include "bench.h"
#include "csv.h"
#include "util.h"
#include "AsconAPI.h"
#include <stdlib.h>
#include <string.h>

void run_ascon80pq_tagonly(const bench_config_t *cfg, csv_writer_t *csv) {
    int total = cfg->warmup + cfg->iterations;

    size_t peak_heap = 0;
    for (int i = 0; i < total; i++) {
        csv_row_t r; memset(&r, 0, sizeof(r));
        strncpy(r.Model, "ModelD_Ascon80pq", sizeof(r.Model)-1);
        r.Iteration = (i - cfg->warmup) + 1;

        r.Failed = 0;

        // No KEM overhead
        r.KeyGen_ns = 0;
        r.Encaps_ns = 0;
        r.Decaps_ns = 0;

        // KDF: if python derives Ascon80pq key from PSK/seed, do it here.
        // For now: use cfg->psk20 directly and count KDF as 0.
        r.KDF_ns = 0;

        // Prepare buffers
        // Ciphertext length = plaintext len + tag len (16)
        size_t clen = cfg->payload_len + 16;
        uint8_t *ct = (uint8_t*)malloc(clen);
        uint8_t *pt = (uint8_t*)malloc(cfg->payload_len);
        uint8_t nonce[16]; 
        // Python: nonce = os.urandom(16) (outside timing)
        // We use a dummy nonce here or rand() to avoid I/O overhead, matching logic that nonce exists.
        memset(nonce, 0xAA, 16); 

        uint64_t te0 = now_ns_monotonic_raw();
        if (ascon80pq_aead_encrypt(ct, &clen,
                                   cfg->payload, cfg->payload_len,
                                   (const uint8_t*)cfg->aad, strlen(cfg->aad),
                                   nonce,
                                   cfg->psk20) != 0) {
            r.Failed = 1;
        }
        uint64_t te1 = now_ns_monotonic_raw();

        uint64_t td0 = now_ns_monotonic_raw();
        size_t mlen = 0;
        if (!r.Failed && ascon80pq_aead_decrypt(pt, &mlen, ct, clen,
                                               (const uint8_t*)cfg->aad, strlen(cfg->aad),
                                               nonce, cfg->psk20) != 0) {
            r.Failed = 1;
        }
        uint64_t td1 = now_ns_monotonic_raw();

        r.Encryption_ns = te1 - te0;
        r.Decryption_ns = td1 - td0;

        if (!r.Failed && (mlen != cfg->payload_len || memcmp(pt, cfg->payload, mlen) != 0)) {
            r.Failed = 1;
        }

        // 2. Calculate Total_ns as a sum (matches Python)
        r.Total_ns = r.Encryption_ns + r.Decryption_ns;
        r.Total_s = (double)r.Total_ns / 1e9;
        r.Peak_Alloc_KB = peak_rss_kb();
        
        size_t current_heap = current_heap_bytes();
        if (current_heap > peak_heap) peak_heap = current_heap;
        r.Heap_Used_Bytes = current_heap;
        r.Heap_Used_Peak_Bytes = peak_heap;

        free(ct); free(pt);

        if (i >= cfg->warmup) csv_write_row(csv, &r);
    }
}