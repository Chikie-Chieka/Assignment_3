#include "bench.h"
#include "csv.h"
#include "util.h"
#include "AsconAPI.h"

#include <oqs/oqs.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <openssl/evp.h>
#include <openssl/kdf.h>

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

static void run_oqs_kem_dem(const char *model_name, const char *kem_alg,
                            const bench_config_t *cfg, csv_writer_t *csv) {

    OQS_KEM *kem = OQS_KEM_new(kem_alg);
    if (!kem) return;

    uint8_t *pk = malloc(kem->length_public_key);
    uint8_t *sk = malloc(kem->length_secret_key);
    uint8_t *ct = malloc(kem->length_ciphertext);
    uint8_t *ss_e = malloc(kem->length_shared_secret);
    uint8_t *ss_d = malloc(kem->length_shared_secret);
    uint8_t *c = NULL;
    uint8_t *m = NULL;
    if (!pk || !sk || !ct || !ss_e || !ss_d) goto out;

    int total = cfg->warmup + cfg->iterations;

    // ciphertext buffer for DEM: payload + 16B tag (typical AEAD)
    size_t ccap = cfg->payload_len + 16;
    c = malloc(ccap);
    m = malloc(cfg->payload_len);
    if (!c || !m) goto out;

    // 1. Measure KeyGen once, outside the loop (matches Python)
    uint64_t t_kg0 = now_ns_monotonic_raw();
    int kg_rc = OQS_KEM_keypair(kem, pk, sk);
    uint64_t t_kg1 = now_ns_monotonic_raw();
    uint64_t keygen_ns = t_kg1 - t_kg0;

    double ncpu = effective_ncpu();
    for (int i = 0; i < total; i++) {
        csv_row_t r; memset(&r, 0, sizeof(r));
        strncpy(r.Model, model_name, sizeof(r.Model)-1);
        r.Iteration = (i - cfg->warmup) + 1;
        r.Failed = (kg_rc != OQS_SUCCESS) ? 1 : 0;
        r.KeyGen_ns = keygen_ns;
        long rss_peak_kb = current_rss_kb();
        if (rss_peak_kb < 0) rss_peak_kb = 0;

        struct timespec w0, w1, c0, c1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &w0);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0);

        uint64_t t1 = now_ns_monotonic_raw();
        if (!r.Failed && OQS_KEM_encaps(kem, ct, ss_e, pk) != OQS_SUCCESS) r.Failed = 1;
        uint64_t t2 = now_ns_monotonic_raw();
        long rss_kb = current_rss_kb();
        if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;
        if (!r.Failed && OQS_KEM_decaps(kem, ss_d, ct, sk) != OQS_SUCCESS) r.Failed = 1;
        uint64_t t3 = now_ns_monotonic_raw();
        rss_kb = current_rss_kb();
        if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;

        r.Encaps_ns = t2 - t1;
        r.Decaps_ns = t3 - t2;

        if (!r.Failed && memcmp(ss_e, ss_d, kem->length_shared_secret) != 0) r.Failed = 1;

        uint8_t k16[16];
        if (!r.Failed) {
            if (kdf_hkdf_sha256(ss_d, kem->length_shared_secret, k16, &r.KDF_ns) != 0) {
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

out:
    if (c) free(c);
    if (m) free(m);
    if (pk) free(pk);
    if (sk) free(sk);
    if (ct) free(ct);
    if (ss_e) free(ss_e);
    if (ss_d) free(ss_d);
    OQS_KEM_free(kem);
}

static void run_oqs_kem_only(const char *model_name, const char *kem_alg,
                             const bench_config_t *cfg, csv_writer_t *csv) {

    OQS_KEM *kem = OQS_KEM_new(kem_alg);
    if (!kem) return;

    uint8_t *pk = malloc(kem->length_public_key);
    uint8_t *sk = malloc(kem->length_secret_key);
    uint8_t *ct = malloc(kem->length_ciphertext);
    uint8_t *ss_e = malloc(kem->length_shared_secret);
    uint8_t *ss_d = malloc(kem->length_shared_secret);
    if (!pk || !sk || !ct || !ss_e || !ss_d) goto out;

    int total = cfg->warmup + cfg->iterations;

    // 1. Measure KeyGen once, outside the loop (matches Python)
    uint64_t t_kg0 = now_ns_monotonic_raw();
    int kg_rc = OQS_KEM_keypair(kem, pk, sk);
    uint64_t t_kg1 = now_ns_monotonic_raw();
    uint64_t keygen_ns = t_kg1 - t_kg0;

    double ncpu = effective_ncpu();
    for (int i = 0; i < total; i++) {
        csv_row_t r; memset(&r, 0, sizeof(r));
        strncpy(r.Model, model_name, sizeof(r.Model)-1);
        r.Iteration = (i - cfg->warmup) + 1;
        r.Failed = (kg_rc != OQS_SUCCESS) ? 1 : 0;
        r.KeyGen_ns = keygen_ns;
        long rss_peak_kb = current_rss_kb();
        if (rss_peak_kb < 0) rss_peak_kb = 0;

        struct timespec w0, w1, c0, c1;
        clock_gettime(CLOCK_MONOTONIC_RAW, &w0);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c0);

        uint64_t t1 = now_ns_monotonic_raw();
        if (!r.Failed && OQS_KEM_encaps(kem, ct, ss_e, pk) != OQS_SUCCESS) r.Failed = 1;
        uint64_t t2 = now_ns_monotonic_raw();
        long rss_kb = current_rss_kb();
        if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;
        if (!r.Failed && OQS_KEM_decaps(kem, ss_d, ct, sk) != OQS_SUCCESS) r.Failed = 1;
        uint64_t t3 = now_ns_monotonic_raw();
        rss_kb = current_rss_kb();
        if (rss_kb > rss_peak_kb) rss_peak_kb = rss_kb;

        r.Encaps_ns = t2 - t1;
        r.Decaps_ns = t3 - t2;

        if (!r.Failed && memcmp(ss_e, ss_d, kem->length_shared_secret) != 0) r.Failed = 1;

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &c1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &w1);

        r.Total_ns = r.Encaps_ns + r.Decaps_ns;
        r.Total_s = (double)r.Total_ns / 1e9;
        r.Cpu_Pct = cpu_pct_process_window(&w0, &w1, &c0, &c1, ncpu);
        r.Peak_Alloc_KB = peak_rss_kb();
        r.Peak_RSS_KB = rss_peak_kb;

        if (i >= cfg->warmup) csv_write_row(csv, &r);
    }

out:
    if (pk) free(pk);
    if (sk) free(sk);
    if (ct) free(ct);
    if (ss_e) free(ss_e);
    if (ss_d) free(ss_d);
    OQS_KEM_free(kem);
}

void run_hybrid_bike_l1_ascon128a(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_dem("Hybrid_BIKE_L1_Ascon128a", OQS_KEM_alg_bike_l1, cfg, csv);
}
void run_hybrid_kyber512_ascon128a(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_dem("Hybrid_Kyber512_Ascon128a", OQS_KEM_alg_kyber_512, cfg, csv);
}
void run_standalone_bike_l1(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_only("Standalone_BIKE_L1", OQS_KEM_alg_bike_l1, cfg, csv);
}
void run_standalone_kyber512(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_only("Standalone_Kyber512", OQS_KEM_alg_kyber_512, cfg, csv);
}
void run_standalone_frodokem_640_aes(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_only("Standalone_FrodoKEM_640_AES", OQS_KEM_alg_frodokem_640_aes, cfg, csv);
}
void run_standalone_hqc_128(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_only("Standalone_HQC_128", OQS_KEM_alg_hqc_128, cfg, csv);
}
void run_standalone_classic_mceliece_348864(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_only("Standalone_ClassicMcEliece_348864", OQS_KEM_alg_classic_mceliece_348864, cfg, csv);
}
void run_hybrid_classic_mceliece_348864_ascon128a(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_dem("Hybrid_ClassicMcEliece_348864_Ascon128a", OQS_KEM_alg_classic_mceliece_348864, cfg, csv);
}
void run_hybrid_frodokem_640_aes_ascon128a(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_dem("Hybrid_FrodoKEM_640_AES_Ascon128a", OQS_KEM_alg_frodokem_640_aes, cfg, csv);
}
void run_hybrid_hqc_128_ascon128a(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_dem("Hybrid_HQC_128_Ascon128a", OQS_KEM_alg_hqc_128, cfg, csv);
}
