#include "csv.h"
#include <string.h>

int csv_open(csv_writer_t *w, const char *path) {
    memset(w, 0, sizeof(*w));
    w->fp = fopen(path, "w");
    if (!w->fp) return -1;
    pthread_mutex_init(&w->mu, NULL);
    return 0;
}

void csv_write_header(csv_writer_t *w) {
    pthread_mutex_lock(&w->mu);
    fprintf(w->fp,
        "Model,Iteration,KeyGen_ns,Encaps_ns,Decaps_ns,KDF_ns,"
        "Encryption_ns,Decryption_ns,Total_ns,Total_s,Failed,Peak_Alloc_KB,"
        "Heap_Used_Bytes,Heap_Used_Peak_Bytes\n");
    fflush(w->fp);
    pthread_mutex_unlock(&w->mu);
}

void csv_write_row(csv_writer_t *w, const csv_row_t *r) {
    pthread_mutex_lock(&w->mu);
    fprintf(w->fp,
        "%s,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.9f,%d,%ld,%zu,%zu\n",
        r->Model, r->Iteration,
        (unsigned long long)r->KeyGen_ns,
        (unsigned long long)r->Encaps_ns,
        (unsigned long long)r->Decaps_ns,
        (unsigned long long)r->KDF_ns,
        (unsigned long long)r->Encryption_ns,
        (unsigned long long)r->Decryption_ns,
        (unsigned long long)r->Total_ns,
        r->Total_s,
        r->Failed,
        r->Peak_Alloc_KB,
        r->Heap_Used_Bytes,
        r->Heap_Used_Peak_Bytes
    );
    fflush(w->fp);
    pthread_mutex_unlock(&w->mu);
}

void csv_close(csv_writer_t *w) {
    if (!w) return;
    if (w->fp) fclose(w->fp);
    pthread_mutex_destroy(&w->mu);
}