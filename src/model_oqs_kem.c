#include "bench.h"
#include "csv.h"
#include "util.h"
#include "ascon_api.h"

#include <oqs/oqs.h>
#include <string.h>
#include <stdlib.h>

// TODO: replace with your python-equivalent KDF.
// For now: derive Ascon-128a key/nonce by hashing ss (placeholder).
#include <openssl/sha.h>

static void kdf_placeholder_sha256(const uint8_t *ss, size_t ss_len,
                                  uint8_t k16[16], uint8_t nonce16[16],
                                  uint64_t *kdf_ns) {
    uint8_t d[32];
    uint64_t t0 = now_ns_monotonic_raw();
    SHA256(ss, ss_len, d);
    memcpy(k16, d, 16);
    memcpy(nonce16, d + 16, 16);
    uint64_t t1 = now_ns_monotonic_raw();
    *kdf_ns = t1 - t0;
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
    if (!pk || !sk || !ct || !ss_e || !ss_d) goto out;

    int total = cfg->warmup + cfg->iterations;

    // ciphertext buffer for DEM: payload + 16B tag (typical AEAD)
    size_t ccap = cfg->payload_len + 16;
    uint8_t *c = malloc(ccap);
    uint8_t *m = malloc(cfg->payload_len);
    if (!c || !m) goto out;

    for (int i = 0; i < total; i++) {
        csv_row_t r; memset(&r, 0, sizeof(r));
        strncpy(r.Model, model_name, sizeof(r.Model)-1);
        r.Iteration = i - cfg->warmup;
        r.Failed = 0;

        uint64_t t_begin = now_ns_monotonic_raw();

        uint64_t t0 = now_ns_monotonic_raw();
        if (OQS_KEM_keypair(kem, pk, sk) != OQS_SUCCESS) r.Failed = 1;
        uint64_t t1 = now_ns_monotonic_raw();
        if (!r.Failed && OQS_KEM_encaps(kem, ct, ss_e, pk) != OQS_SUCCESS) r.Failed = 1;
        uint64_t t2 = now_ns_monotonic_raw();
        if (!r.Failed && OQS_KEM_decaps(kem, ss_d, ct, sk) != OQS_SUCCESS) r.Failed = 1;
        uint64_t t3 = now_ns_monotonic_raw();

        r.KeyGen_ns = t1 - t0;
        r.Encaps_ns = t2 - t1;
        r.Decaps_ns = t3 - t2;

        if (!r.Failed && memcmp(ss_e, ss_d, kem->length_shared_secret) != 0) r.Failed = 1;

        uint8_t k16[16], nonce16[16];
        if (!r.Failed) {
            kdf_placeholder_sha256(ss_d, kem->length_shared_secret, k16, nonce16, &r.KDF_ns);

            uint64_t te0 = now_ns_monotonic_raw();
            size_t clen = 0;
            // TODO: replace with your Ascon-128a implementation + exact aad handling
            if (ascon128a_aead_encrypt(c, &clen, cfg->payload, cfg->payload_len,
                                      (const uint8_t*)cfg->aad, strlen(cfg->aad),
                                      nonce16, k16) != 0) {
                r.Failed = 1;
            }
            uint64_t te1 = now_ns_monotonic_raw();

            uint64_t td0 = now_ns_monotonic_raw();
            size_t mlen = 0;
            if (!r.Failed && ascon128a_aead_decrypt(m, &mlen, c, clen,
                                                   (const uint8_t*)cfg->aad, strlen(cfg->aad),
                                                   nonce16, k16) != 0) {
                r.Failed = 1;
            }
            uint64_t td1 = now_ns_monotonic_raw();

            r.Encryption_ns = te1 - te0;
            r.Decryption_ns = td1 - td0;

            if (!r.Failed && (mlen != cfg->payload_len || memcmp(m, cfg->payload, cfg->payload_len) != 0))
                r.Failed = 1;
        }

        uint64_t t_end = now_ns_monotonic_raw();
        r.Total_ns = t_end - t_begin;
        r.Total_s = (double)r.Total_ns / 1e9;
        r.Peak_Alloc_KB = peak_rss_kb();

        if (i >= cfg->warmup) csv_write_row(csv, &r);
    }

out:
    if (pk) free(pk); if (sk) free(sk); if (ct) free(ct);
    if (ss_e) free(ss_e); if (ss_d) free(ss_d);
    OQS_KEM_free(kem);
}

void run_kyber512_ascon128a(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_dem("ModelA_Kyber512", OQS_KEM_alg_kyber_512, cfg, csv);
}
void run_bike_l1_ascon128a(const bench_config_t *cfg, csv_writer_t *csv) {
    run_oqs_kem_dem("ModelC_BIKE_L1", OQS_KEM_alg_bike_l1, cfg, csv);
}