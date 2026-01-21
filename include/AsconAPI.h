#pragma once
#include <stddef.h>
#include <stdint.h>
// (adapter placeholder)
// You must choose/provide an Ascon C implementation.
// Provide these functions as wrappers around that implementation.

// Ascon-128a AEAD: key=16B, nonce=16B, tag=16B (typical)
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

// Ascon-80pq tag-only (your requirement: “16B tag only”, no ciphertext).
// Define semantics: tag = MAC(key, aad||payload||...), or AEAD with zero-length plaintext.
// Return 0 on success.
int ascon80pq_tag16_compute(
    uint8_t tag16[16],
    const uint8_t *msg, size_t msglen,
    const uint8_t *ad, size_t adlen,
    const uint8_t key20[20]); // Ascon-80pq key is 20B