#include "bench.h"
#include "csv.h"
#include "util.h"
#include "AsconAPI.h"
#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static double cpu_pct_process_window(const struct timespec *w0, const struct timespec *w1,
                                     const struct timespec *c0, const struct timespec *c1,
                                     double ncpu) {
    double wall = (double)(w1->tv_sec - w0->tv_sec) +
                  (double)(w1->tv_nsec - w0->tv_nsec) / 1e9;
    double cpu = (double)(c1->tv_sec - c0->tv_sec) +
                 (double)(c1->tv_nsec - c0->tv_nsec) / 1e9;
    if (wall <= 0.0 || ncpu <= 0.0) return 0.0;
    return 100.0 * (cpu / (wall * (double)ncpu));
}

static int x25519_keypair(EVP_PKEY **out) {
    EVP_PKEY_CTX *pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_X25519, NULL);
    if (!pctx) return 0;
    int ok = 0;
    if (EVP_PKEY_keygen_init(pctx) <= 0) goto out;
    if (EVP_PKEY_keygen(pctx, out) <= 0) goto out;
    ok = 1;
out:
    EVP_PKEY_CTX_free(pctx);
    return ok;
}

static int x25519_derive(EVP_PKEY *priv, EVP_PKEY *peer,
                         uint8_t *ss, size_t ss_cap, size_t *ss_len) {
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(priv, NULL);
    if (!ctx) return 0;
    int ok = 0;
    if (EVP_PKEY_derive_init(ctx) <= 0) goto out;
    if (EVP_PKEY_derive_set_peer(ctx, peer) <= 0) goto out;
    size_t len = 0;
    if (EVP_PKEY_derive(ctx, NULL, &len) <= 0) goto out;
    if (len > ss_cap) goto out;
    if (EVP_PKEY_derive(ctx, ss, &len) <= 0) goto out;
    *ss_len = len;
    ok = 1;
out:
    EVP_PKEY_CTX_free(ctx);
    return ok;
}

static int kdf_hkdf_sha256(const uint8_t *ikm, size_t ikm_len,
                           uint8_t out_key[16], uint64_t *kdf_ns) {
    uint64_t t0 = now_ns_monotonic_raw();
    int ret = -1;
    EVP_PKEY_CTX *pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF, NULL);
    if (!pctx) goto out;

    if (EVP_PKEY_derive_init(pctx) <= 0) goto out;
    if (EVP_PKEY_CTX_set_hkdf_md(pctx, EVP_sha256()) <= 0) goto out;
    if (EVP_PKEY_CTX_set1_hkdf_salt(pctx, (const unsigned char*)"hybrid-bench-salt", 17) <= 0) goto out;
    if (EVP_PKEY_CTX_set1_hkdf_key(pctx, ikm, ikm_len) <= 0) goto out;
    if (EVP_PKEY_CTX_add1_hkdf_info(pctx, (const unsigned char*)"ascon-128a-key", 14) <= 0) goto out;

    size_t len = 16;
    if (EVP_PKEY_derive(pctx, out_key, &len) <= 0) goto out;

    ret = 0;
out:
    if (pctx) EVP_PKEY_CTX_free(pctx);
    uint64_t t1 = now_ns_monotonic_raw();
    *kdf_ns = t1 - t0;
    return ret;
}

void run_x25519_ascon128a(const bench_config_t *cfg, csv_writer_t *csv) {
    int total = cfg->warmup + cfg->iterations;

    size_t ccap = cfg->payload_len + 16;
    uint8_t *c = malloc(ccap);
    uint8_t *m = malloc(cfg->payload_len);
    if (!c || !m) return;

    // 1. Measure KeyGen once, outside the loop
    uint64_t t_kg0 = now_ns_monotonic_raw();
    EVP_PKEY *srv = NULL, *cli = NULL;
    int kg_ok = 1;
    if (!x25519_keypair(&srv) || !x25519_keypair(&cli)) kg_ok = 0;
    uint64_t t_kg1 = now_ns_monotonic_raw();
    uint64_t keygen_ns = t_kg1 - t_kg0;

    double ncpu = effective_ncpu();
    for (int i = 0; i < total; i++) {
        csv_row_t r; memset(&r, 0, sizeof(r));
        strncpy(r.Model, "ModelB_X25519", sizeof(r.Model)-1);
        r.Iteration = (i - cfg->warmup) + 1;
        r.Failed = !kg_ok;
        r.KeyGen_ns = keygen_ns;
        long rss_peak_kb = current_rss_kb();
        if (rss_peak_kb < 0) rss_peak_kb = 0;

        struct timespec w0, w1, c0, c1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &w0);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0);

        uint64_t t1 = now_ns_monotonic_raw();

        uint8_t ss_cli[64], ss_srv[64];
        size_t ss_cli_len=0, ss_srv_len=0;

        if (!r.Failed && !x25519_derive(cli, srv, ss_cli, sizeof(ss_cli), &ss_cli_len)) r.Failed = 1;
        uint64_t t2 = now_ns_monotonic_raw();
        long rss_kb = current_rss_kb();
        if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;
        if (!r.Failed && !x25519_derive(srv, cli, ss_srv, sizeof(ss_srv), &ss_srv_len)) r.Failed = 1;
        uint64_t t3 = now_ns_monotonic_raw();
        rss_kb = current_rss_kb();
        if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;

        r.Encaps_ns = t2 - t1;
        r.Decaps_ns = t3 - t2;

        if (!r.Failed && (ss_cli_len != ss_srv_len || memcmp(ss_cli, ss_srv, ss_cli_len) != 0)) r.Failed = 1;

        if (!r.Failed) {
            uint8_t k16[16];
            if (kdf_hkdf_sha256(ss_srv, ss_srv_len, k16, &r.KDF_ns) != 0) {
                r.Failed = 1;
            }
            rss_kb = current_rss_kb();
            if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;

            uint8_t nonce16[16];
            for(int k=0; k<16; ++k) nonce16[k] = rand() & 0xFF;
            
            uint64_t te0 = now_ns_monotonic_raw();
            size_t clen = 0;
            if (ascon128a_aead_encrypt(c, &clen, cfg->payload, cfg->payload_len,
                                      (const uint8_t*)cfg->aad, strlen(cfg->aad),
                                      nonce16, k16) != 0) {
                r.Failed = 1;
            }
            uint64_t te1 = now_ns_monotonic_raw();
            rss_kb = current_rss_kb();
            if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;

            uint64_t td0 = now_ns_monotonic_raw();
            size_t mlen = 0;
            if (!r.Failed && ascon128a_aead_decrypt(m, &mlen, c, clen,
                                                   (const uint8_t*)cfg->aad, strlen(cfg->aad),
                                                   nonce16, k16) != 0) {
                r.Failed = 1;
            }
            uint64_t td1 = now_ns_monotonic_raw();
            rss_kb = current_rss_kb();
            if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;

            r.Encryption_ns = te1 - te0;
            r.Decryption_ns = td1 - td0;

        if (!r.Failed && (mlen != cfg->payload_len || memcmp(m, cfg->payload, cfg->payload_len) != 0))
            r.Failed = 1;
        }

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &w1);

        // 2. Calculate Total_ns as a sum (matches Python)
        r.Total_ns = r.Encaps_ns + r.Decaps_ns + r.KDF_ns + r.Encryption_ns + r.Decryption_ns;
        r.Total_s = (double)r.Total_ns / 1e9;
        r.Cpu_Pct = cpu_pct_process_window(&w0, &w1, &c0, &c1, ncpu);
        r.Peak_Alloc_KB = peak_rss_kb();
        r.Peak_RSS_KB = rss_peak_kb;

        if (i >= cfg->warmup) csv_write_row(csv, &r);
    }

    EVP_PKEY_free(srv);
    EVP_PKEY_free(cli);
    free(c);
    free(m);
}

void run_standalone_x25519(const bench_config_t *cfg, csv_writer_t *csv) {
    int total = cfg->warmup + cfg->iterations;

    // 1. Measure KeyGen once, outside the loop
    uint64_t t_kg0 = now_ns_monotonic_raw();
    EVP_PKEY *srv = NULL, *cli = NULL;
    int kg_ok = 1;
    if (!x25519_keypair(&srv) || !x25519_keypair(&cli)) kg_ok = 0;
    uint64_t t_kg1 = now_ns_monotonic_raw();
    uint64_t keygen_ns = t_kg1 - t_kg0;

    double ncpu = effective_ncpu();
    for (int i = 0; i < total; i++) {
        csv_row_t r; memset(&r, 0, sizeof(r));
        strncpy(r.Model, "Standalone_X25519", sizeof(r.Model)-1);
        r.Iteration = (i - cfg->warmup) + 1;
        r.Failed = !kg_ok;
        r.KeyGen_ns = keygen_ns;
        long rss_peak_kb = current_rss_kb();
        if (rss_peak_kb < 0) rss_peak_kb = 0;

        struct timespec w0, w1, c0, c1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &w0);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0);

        uint64_t t1 = now_ns_monotonic_raw();

        uint8_t ss_cli[64], ss_srv[64];
        size_t ss_cli_len = 0, ss_srv_len = 0;

        if (!r.Failed && !x25519_derive(cli, srv, ss_cli, sizeof(ss_cli), &ss_cli_len)) r.Failed = 1;
        uint64_t t2 = now_ns_monotonic_raw();
        long rss_kb = current_rss_kb();
        if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;
        if (!r.Failed && !x25519_derive(srv, cli, ss_srv, sizeof(ss_srv), &ss_srv_len)) r.Failed = 1;
        uint64_t t3 = now_ns_monotonic_raw();
        rss_kb = current_rss_kb();
        if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;

        r.Encaps_ns = t2 - t1;
        r.Decaps_ns = t3 - t2;

        if (!r.Failed && (ss_cli_len != ss_srv_len || memcmp(ss_cli, ss_srv, ss_cli_len) != 0)) r.Failed = 1;

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &w1);

        r.Total_ns = r.Encaps_ns + r.Decaps_ns;
        r.Total_s = (double)r.Total_ns / 1e9;
        r.Cpu_Pct = cpu_pct_process_window(&w0, &w1, &c0, &c1, ncpu);
        r.Peak_Alloc_KB = peak_rss_kb();
        r.Peak_RSS_KB = rss_peak_kb;

        if (i >= cfg->warmup) csv_write_row(csv, &r);
    }

    EVP_PKEY_free(srv);
    EVP_PKEY_free(cli);
}
