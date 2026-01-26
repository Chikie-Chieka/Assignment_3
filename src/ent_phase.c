#define _GNU_SOURCE
#include "util.h"
#include "models.h"   // for model_kind_t if you reuse it
#include "AsconAPI.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --------------------
// CSV writing (ENT)
// --------------------
typedef struct {
    FILE *fp;
    pthread_mutex_t mu;
} ent_writer_t;

static int ent_open(ent_writer_t *w, const char *path) {
    memset(w, 0, sizeof(*w));
    w->fp = fopen(path, "w");
    if (!w->fp) return -1;
    pthread_mutex_init(&w->mu, NULL);
    return 0;
}
static void ent_close(ent_writer_t *w) {
    if (w->fp) fclose(w->fp);
    pthread_mutex_destroy(&w->mu);
}
static void ent_write_header(ent_writer_t *w) {
    if (!w || !w->fp) return;
    pthread_mutex_lock(&w->mu);
    fprintf(w->fp, "Model,Iteration,Entropy_Bytes,Entropy_Bits_Per_Byte,Serial_Correlation_Coefficient,Status\n");
    fflush(w->fp);
    pthread_mutex_unlock(&w->mu);
}
static void ent_write_row(ent_writer_t *w,
                          const char *model, int iter,
                          size_t entropy_bytes, double h_bits_per_byte,
                          double scc, const char *status) {
    if (!w || !w->fp) return;
    pthread_mutex_lock(&w->mu);
    fprintf(w->fp, "%s,%d,%zu,%.12f,%.12f,%s\n",
            model, iter, entropy_bytes, h_bits_per_byte, scc, status);
    fflush(w->fp);
    pthread_mutex_unlock(&w->mu);
}

// --------------------
// Metrics
// --------------------
static double shannon_entropy_bits_per_byte(const uint8_t *buf, size_t n) {
    if (n == 0) return 0.0;
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n; i++) hist[buf[i]]++;

    double h = 0.0;
    const double inv_n = 1.0 / (double)n;
    for (int b = 0; b < 256; b++) {
        if (!hist[b]) continue;
        double p = (double)hist[b] * inv_n;
        h -= p * (log(p) / log(2.0));
    }
    return h; // bits per symbol (byte)
}

// Standard “serial correlation coefficient” with wrap-around:
// scc = (n*sum(x_i*x_{i+1}) - sum(x)^2) / (n*sum(x_i^2) - sum(x)^2)
static double serial_correlation_coeff(const uint8_t *buf, size_t n) {
    if (n < 2) return 0.0;

    double sum = 0.0, sumsq = 0.0, sumprod = 0.0;
    for (size_t i = 0; i < n; i++) {
        double x = (double)buf[i];
        double y = (double)buf[(i + 1) % n];
        sum += x;
        sumsq += x * x;
        sumprod += x * y;
    }
    double denom = (double)n * sumsq - sum * sum;
    if (denom == 0.0) return 0.0;
    double num = (double)n * sumprod - sum * sum;
    return num / denom;
}

// --------------------
// Byte generation (MUST MATCH PYTHON)
// --------------------
// Common choice is to generate a fresh encryption output and test:
// - For KEM+DEM models: (kem_ct || dem_ct || dem_tag) or (dem_ct || dem_tag)
// - For X25519+DEM: (dem_ct || dem_tag)
// - For Ascon80pq: (tag only) or (ct||tag) depending on python.
static int ent_generate_bytes_for_model(const bench_config_t *cfg,
                                       const char *model_name,
                                       int iteration,
                                       uint8_t **out, size_t *out_len) {
    (void)iteration;

    // Reproduce Python: payload = configurable MB of 0x00 (default 1MB)
    size_t mlen = (size_t)cfg->ent_payload_mb * 1024 * 1024;
    uint8_t *m = (uint8_t*)calloc(1, mlen);
    if (!m) return -1;

    // Output buffer: Ciphertext + Tag (16B)
    size_t clen_buf = mlen + 16;
    uint8_t *c = (uint8_t*)malloc(clen_buf);
    if (!c) { free(m); return -1; }

    // Random 16B key and 16B nonce
    // Python uses os.urandom(16). We use rand() for reproduction.
    uint8_t k[20] = {0}; // Max key size (20B for Ascon80pq)
    uint8_t n[16] = {0};
    for (int i = 0; i < 16; i++) k[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < 16; i++) n[i] = (uint8_t)(rand() & 0xFF);

    // Encrypt
    size_t clen_out = 0;
    int res = -1;

    if (strcmp(model_name, "Standalone_Ascon_80pq") == 0) {
        // Ascon-80pq (20B key). We use the 16 random bytes + 4 zero bytes (from init)
        // to match Python passing 16B key to 80pq variant.
        res = ascon80pq_aead_encrypt(c, &clen_out, m, mlen, NULL, 0, n, k);
    } else {
        // Ascon-128a (16B key)
        res = ascon128a_aead_encrypt(c, &clen_out, m, mlen, NULL, 0, n, k);
    }

    free(m);

    if (res != 0) {
        free(c);
        return -1;
    }

    // Return ONLY ciphertext (exclude 16B tag)
    // Ascon ciphertext is same length as plaintext.
    *out = c;
    *out_len = mlen; 
    return 0;
}

// --------------------
// Thread worker
// --------------------
typedef struct {
    const bench_config_t *cfg;
    const char *model_name;
    ent_writer_t *writer;
} ent_thread_arg_t;

static void *ent_thread_main(void *p) {
    ent_thread_arg_t *a = (ent_thread_arg_t*)p;
    // Python: N iterations, 1-based index
    for (int i = 1; i <= a->cfg->ent_iterations; i++) {
        uint8_t *buf = NULL;
        size_t n = 0;
        int rc = ent_generate_bytes_for_model(a->cfg, a->model_name, i, &buf, &n);

        double h = 0.0, scc = 0.0;
        const char *status = "ent test failed"; // Default error

        if (rc == 0) {
            h = shannon_entropy_bits_per_byte(buf, n);
            scc = serial_correlation_coeff(buf, n);
            // Python sets "success" if metrics are computed
            status = "success";
        }

        ent_write_row(a->writer, a->model_name, i, n, h, scc, status);
        if (buf) free(buf);
    }
    return NULL;
}

// --------------------
// Public entry
// --------------------
int run_ent_phase(const bench_config_t *cfg, const char *ent_path) {
    ent_writer_t w;
    ent_writer_t *wp = NULL;
    if (!cfg->no_csv) {
        if (ent_open(&w, ent_path) != 0) return -1;
        ent_write_header(&w);
        wp = &w;
    }

    struct {
        int id;
        ent_thread_arg_t arg;
    } models[] = {
        { 1, { cfg, "Standalone_Ascon_80pq", wp } },
        { 2, { cfg, "Standalone_BIKE_L1", wp } },
        { 3, { cfg, "Standalone_Kyber512", wp } },
        { 4, { cfg, "Standalone_FrodoKEM_640_AES", wp } },
        { 5, { cfg, "Standalone_HQC_128", wp } },
        { 6, { cfg, "Standalone_ClassicMcEliece_348864", wp } },
        { 7, { cfg, "Standalone_X25519", wp } },
        { 8, { cfg, "Hybrid_BIKE_L1_Ascon128a", wp } },
        { 9, { cfg, "Hybrid_Kyber512_Ascon128a", wp } },
        { 10, { cfg, "Hybrid_FrodoKEM_640_AES_Ascon128a", wp } },
        { 11, { cfg, "Hybrid_HQC_128_Ascon128a", wp } },
        { 12, { cfg, "Hybrid_ClassicMcEliece_348864_Ascon128a", wp } },
        { 13, { cfg, "Hybrid_X25519_Ascon128a", wp } },
    };

    if (cfg->single_thread_mode == SINGLE_THREAD_FULL) {
        int model_count = (int)(sizeof(models) / sizeof(models[0]));
        for (int i = 0; i < model_count; i++) {
            if (cfg->model_id != 0 && cfg->model_id != models[i].id) continue;
            ent_thread_main(&models[i].arg);
        }
        if (!cfg->no_csv) ent_close(&w);
        return 0;
    }

    pthread_t th[13];
    int th_count = 0;
    int create_failed = 0;
    int model_count = (int)(sizeof(models) / sizeof(models[0]));
    for (int i = 0; i < model_count; i++) {
        if (cfg->model_id != 0 && cfg->model_id != models[i].id) continue;
        if (pthread_create(&th[th_count], NULL, ent_thread_main, &models[i].arg) != 0) {
            create_failed = 1;
            continue;
        }
        th_count++;
    }
    for (int i = 0; i < th_count; i++) pthread_join(th[i], NULL);

    if (!cfg->no_csv) ent_close(&w);
    return create_failed ? -1 : 0;
}
