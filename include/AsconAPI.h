#pragma once
#include <stddef.h>
#include <stdint.h>

// Ascon-128a AEAD: key=16B, nonce=16B, tag=16B 
int ascon128a_aead_encrypt(
    uint8_t *c, size_t *clen,
    const uint8_t *m, size_t mlen,
    const uint8_t *ad, size_t adlen,
    const uint8_t *npub16,
    const uint8_t *k16);

int ascon128a_aead_decrypt(
    uint8_t *m, size_t *mlen,
    const uint8_t *c, size_t clen,
    const uint8_t *ad, size_t adlen,
    const uint8_t *npub16,
    const uint8_t *k16);

// Ascon-80pq AEAD: key=20B, nonce=16B, tag=16B
int ascon80pq_aead_encrypt(
    uint8_t *c, size_t *clen,
    const uint8_t *m, size_t mlen,
    const uint8_t *ad, size_t adlen,
    const uint8_t *npub16,
    const uint8_t *k20);

int ascon80pq_aead_decrypt(
    uint8_t *m, size_t *mlen,
    const uint8_t *c, size_t clen,
    const uint8_t *ad, size_t adlen,
    const uint8_t *npub16,
    const uint8_t *k20);