#!/usr/bin/env python3
"""
main_experiment.py

Fair KEM -> KDF -> DEM benchmarking harness.

Models:
- ModelA_Kyber512 (liboqs KEM)
- ModelB_BIKE_L1 (liboqs KEM)
- ModelC_X25519 (PyNaCl)
- ModelD_PSK_Ascon80pq (no KEM; PSK + Ascon-80pq)

Pipeline (per iteration):
1) Establish shared secret (KEM encaps/decaps or PSK)
2) KDF (HKDF-SHA256) -> 16-byte key (Models A-C), none for D
3) DEM: Ascon encrypt/decrypt, verify correctness
4) Measure per-phase timings; output JSON with mean/stddev and failures.

Notes:
- This script expects `python-ascon` for Ascon AEAD. There are multiple Ascon
  Python packages with different APIs. We try to import a common one and provide
  a clear error if not found.
- liboqs Python bindings must be installed and support Kyber512 and BIKE-L1.
- PyNaCl must be installed for X25519.
"""

from __future__ import annotations

import abc
import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Dependency helpers
# ----------------------------

def _import_liboqs():
    try:
        import oqs  # type: ignore
        return oqs
    except Exception as e:
        raise RuntimeError(
            "liboqs-python not available. Install liboqs + Python bindings.\n"
            "Typical: pip install oqs (or build from source depending on platform).\n"
            f"Import error: {e}"
        )

def _import_pynacl():
    try:
        from nacl.public import PrivateKey  # type: ignore
        from nacl.bindings import crypto_scalarmult  # type: ignore
        return PrivateKey, crypto_scalarmult
    except Exception as e:
        raise RuntimeError(
            "PyNaCl not available. Install with: pip install pynacl\n"
            f"Import error: {e}"
        )

def _import_ascon():
    """
    Try to locate an Ascon AEAD implementation.

    We support a common `ascon` module shape where:
      - ascon.encrypt(key, nonce, ad, plaintext, variant="Ascon-128a") -> ciphertext+tag
      - ascon.decrypt(key, nonce, ad, ciphertext_and_tag, variant="Ascon-128a") -> plaintext

    If your Ascon library uses a different API, adapt the AsconAdapter below.
    """
    try:
        ascon = importlib.import_module("ascon")
        # Very lightweight sanity check
        if not hasattr(ascon, "encrypt") or not hasattr(ascon, "decrypt"):
            raise AttributeError("ascon module found but missing encrypt/decrypt")
        return ascon
    except Exception as e:
        raise RuntimeError(
            "Ascon Python module not available or incompatible.\n"
            "Install an Ascon AEAD implementation, e.g. `pip install ascon` (varies by distro),\n"
            "or provide an `ascon.py` with encrypt/decrypt in the same directory.\n"
            f"Import error: {e}"
        )

# ----------------------------
# HKDF (RFC 5869) - SHA256
# ----------------------------

def hkdf_sha256(ikm: bytes, length: int, salt: bytes = b"", info: bytes = b"") -> bytes:
    if length <= 0 or length > 32 * 255:
        raise ValueError("Invalid HKDF length")
    if salt == b"":
        salt = b"\x00" * 32
    prk = hmac_sha256(salt, ikm)
    t = b""
    okm = b""
    counter = 1
    while len(okm) < length:
        t = hmac_sha256(prk, t + info + bytes([counter]))
        okm += t
        counter += 1
    return okm[:length]

def hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", data, key, 1, dklen=32)

# ----------------------------
# Ascon adapter
# ----------------------------

class AsconAdapter:
    """
    Normalizes encrypt/decrypt to return (nonce, ct, tag) and accept plaintext/aad.

    This adapter assumes an ascon API:
      out = ascon.encrypt(key, nonce, ad, pt, variant="Ascon-128a")
      pt  = ascon.decrypt(key, nonce, ad, ct_and_tag, variant="Ascon-128a")

    where ct_and_tag is ct||tag (tag length 16 for Ascon-128a and also for Ascon-80pq in common specs).
    """
    def __init__(self):
        self.ascon = _import_ascon()

    @staticmethod
    def _split_ct_tag(ct_and_tag: bytes, tag_len: int = 16) -> Tuple[bytes, bytes]:
        if len(ct_and_tag) < tag_len:
            raise ValueError("Ciphertext too short for tag")
        return ct_and_tag[:-tag_len], ct_and_tag[-tag_len:]

    @staticmethod
    def _join_ct_tag(ct: bytes, tag: bytes) -> bytes:
        return ct + tag

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"", variant: str = "Ascon-128a") -> Tuple[bytes, bytes, bytes]:
        # 128-bit nonce for Ascon AEAD variants in many implementations
        nonce = os.urandom(16)
        ct_and_tag = self.ascon.encrypt(key, nonce, aad, plaintext, variant=variant)
        ct, tag = self._split_ct_tag(ct_and_tag, tag_len=16)
        return nonce, ct, tag

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"", variant: str = "Ascon-128a") -> bytes:
        ct_and_tag = self._join_ct_tag(ct, tag)
        pt = self.ascon.decrypt(key, nonce, aad, ct_and_tag, variant=variant)
        return pt

# ----------------------------
# Benchmark stats
# ----------------------------

@dataclass
class PhaseStats:
    times: List[float]

    def mean(self) -> float:
        return statistics.fmean(self.times) if self.times else 0.0

    def stdev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) >= 2 else 0.0

@dataclass
class ModelResult:
    name: str
    iterations: int
    failures: int
    encaps_ms: Dict[str, float]
    decaps_ms: Dict[str, float]
    kdf_ms: Dict[str, float]
    enc_ms: Dict[str, float]
    dec_ms: Dict[str, float]

# ----------------------------
# HybridModel base class
# ----------------------------

class HybridModel(abc.ABC):
    def __init__(self, name: str, ascon_adapter: AsconAdapter):
        self.name = name
        self.ascon = ascon_adapter

    @abc.abstractmethod
    def setup(self) -> None:
        """Initialize any crypto contexts/keys required for repeated runs."""

    @abc.abstractmethod
    def establish_shared_secret(self) -> Tuple[bytes, Dict[str, float]]:
        """
        Returns (shared_secret, timings_ms_dict).

        timings_ms_dict must include:
          - encaps_ms: float
          - decaps_ms: float
        For PSK models, set both to 0.0.
        """

    @abc.abstractmethod
    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, float]:
        """Returns (dem_key, kdf_time_ms)."""

    @abc.abstractmethod
    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes, float]:
        """Returns (nonce, ct, tag, enc_time_ms)."""

    @abc.abstractmethod
    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> Tuple[bytes, float]:
        """Returns (plaintext, dec_time_ms)."""

# ----------------------------
# Model implementations
# ----------------------------

class ModelA_Kyber512(HybridModel):
    def __init__(self, ascon_adapter: AsconAdapter):
        super().__init__("ModelA_Kyber512", ascon_adapter)
        self.oqs = _import_liboqs()
        self.kem_name = "Kyber512"
        self._server = None
        self._server_public_key = None
        self._server_secret_key = None

    def setup(self) -> None:
        with self.oqs.KeyEncapsulation(self.kem_name) as kem:
            pk = kem.generate_keypair()
            sk = kem.export_secret_key()
        self._server_public_key = pk
        self._server_secret_key = sk

    def establish_shared_secret(self) -> Tuple[bytes, Dict[str, float]]:
        t0 = time.perf_counter()
        with self.oqs.KeyEncapsulation(self.kem_name) as client_kem:
            ct, ss_client = client_kem.encap_secret(self._server_public_key)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        with self.oqs.KeyEncapsulation(self.kem_name, secret_key=self._server_secret_key) as server_kem:
            ss_server = server_kem.decap_secret(ct)
        t3 = time.perf_counter()

        if ss_client != ss_server:
            raise ValueError("KEM shared secret mismatch")

        return ss_client, {"encaps_ms": (t1 - t0) * 1e3, "decaps_ms": (t3 - t2) * 1e3}

    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, float]:
        t0 = time.perf_counter()
        key = hkdf_sha256(ss, length=16, salt=b"hybrid-bench-salt", info=b"ascon-128a-key")
        t1 = time.perf_counter()
        return key, (t1 - t0) * 1e3

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes, float]:
        t0 = time.perf_counter()
        nonce, ct, tag = self.ascon.encrypt(key, plaintext, aad=aad, variant="Ascon-128a")
        t1 = time.perf_counter()
        return nonce, ct, tag, (t1 - t0) * 1e3

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> Tuple[bytes, float]:
        t0 = time.perf_counter()
        pt = self.ascon.decrypt(key, nonce, ct, tag, aad=aad, variant="Ascon-128a")
        t1 = time.perf_counter()
        return pt, (t1 - t0) * 1e3


class ModelB_BIKE_L1(ModelA_Kyber512):
    def __init__(self, ascon_adapter: AsconAdapter):
        super().__init__(ascon_adapter)
        self.name = "ModelB_BIKE_L1"
        self.kem_name = "BIKE-L1"


class ModelC_X25519(HybridModel):
    def __init__(self, ascon_adapter: AsconAdapter):
        super().__init__("ModelC_X25519", ascon_adapter)
        PrivateKey, crypto_scalarmult = _import_pynacl()
        self.PrivateKey = PrivateKey
        self.crypto_scalarmult = crypto_scalarmult
        self._alice_sk = None
        self._alice_pk = None
        self._bob_sk = None
        self._bob_pk = None

    def setup(self) -> None:
        self._alice_sk = self.PrivateKey.generate()
        self._bob_sk = self.PrivateKey.generate()
        self._alice_pk = bytes(self._alice_sk.public_key)
        self._bob_pk = bytes(self._bob_sk.public_key)

    def establish_shared_secret(self) -> Tuple[bytes, Dict[str, float]]:
        # For fairness with KEM, treat "encaps" as Alice computing DH, "decaps" as Bob computing DH.
        t0 = time.perf_counter()
        ss_alice = self.crypto_scalarmult(bytes(self._alice_sk), self._bob_pk)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        ss_bob = self.crypto_scalarmult(bytes(self._bob_sk), self._alice_pk)
        t3 = time.perf_counter()

        if ss_alice != ss_bob:
            raise ValueError("X25519 shared secret mismatch")

        return ss_alice, {"encaps_ms": (t1 - t0) * 1e3, "decaps_ms": (t3 - t2) * 1e3}

    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, float]:
        t0 = time.perf_counter()
        key = hkdf_sha256(ss, length=16, salt=b"hybrid-bench-salt", info=b"ascon-128a-key")
        t1 = time.perf_counter()
        return key, (t1 - t0) * 1e3

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes, float]:
        t0 = time.perf_counter()
        nonce, ct, tag = self.ascon.encrypt(key, plaintext, aad=aad, variant="Ascon-128a")
        t1 = time.perf_counter()
        return nonce, ct, tag, (t1 - t0) * 1e3

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> Tuple[bytes, float]:
        t0 = time.perf_counter()
        pt = self.ascon.decrypt(key, nonce, ct, tag, aad=aad, variant="Ascon-128a")
        t1 = time.perf_counter()
        return pt, (t1 - t0) * 1e3


class ModelD_PSK_Ascon80pq(HybridModel):
    def __init__(self, ascon_adapter: AsconAdapter, psk_20_bytes: bytes):
        super().__init__("ModelD_PSK_Ascon80pq", ascon_adapter)
        if len(psk_20_bytes) != 20:
            raise ValueError("ModelD requires a fixed 20-byte PSK for Ascon-80pq")
        self.psk = psk_20_bytes

    def setup(self) -> None:
        # Nothing to initialize; PSK is fixed.
        pass

    def establish_shared_secret(self) -> Tuple[bytes, Dict[str, float]]:
        # No KEM; shared secret is the PSK.
        return self.psk, {"encaps_ms": 0.0, "decaps_ms": 0.0}

    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, float]:
        # No HKDF here by requirement; key is already 20 bytes for Ascon-80pq.
        return ss, 0.0

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes, float]:
        t0 = time.perf_counter()
        nonce, ct, tag = self.ascon.encrypt(key, plaintext, aad=aad, variant="Ascon-80pq")
        t1 = time.perf_counter()
        return nonce, ct, tag, (t1 - t0) * 1e3

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> Tuple[bytes, float]:
        t0 = time.perf_counter()
        pt = self.ascon.decrypt(key, nonce, ct, tag, aad=aad, variant="Ascon-80pq")
        t1 = time.perf_counter()
        return pt, (t1 - t0) * 1e3

# ----------------------------
# Runner
# ----------------------------

def bench_model(model: HybridModel, payload: bytes, aad: bytes, iterations: int) -> ModelResult:
    encaps_t: List[float] = []
    decaps_t: List[float] = []
    kdf_t: List[float] = []
    enc_t: List[float] = []
    dec_t: List[float] = []
    failures = 0

    model.setup()

    for _ in range(iterations):
        try:
            ss, kem_times = model.establish_shared_secret()
            encaps_t.append(float(kem_times.get("encaps_ms", 0.0)))
            decaps_t.append(float(kem_times.get("decaps_ms", 0.0)))

            key, kdf_ms = model.derive_dem_key(ss)
            kdf_t.append(float(kdf_ms))

            nonce, ct, tag, enc_ms = model.encrypt(key, payload, aad=aad)
            enc_t.append(float(enc_ms))

            pt, dec_ms = model.decrypt(key, nonce, ct, tag, aad=aad)
            dec_t.append(float(dec_ms))

            if pt != payload:
                failures += 1
        except Exception:
            failures += 1

    def summarize(xs: List[float]) -> Dict[str, float]:
        return {
            "mean": statistics.fmean(xs) if xs else 0.0,
            "stdev": statistics.stdev(xs) if len(xs) >= 2 else 0.0,
        }

    return ModelResult(
        name=model.name,
        iterations=iterations,
        failures=failures,
        encaps_ms=summarize(encaps_t),
        decaps_ms=summarize(decaps_t),
        kdf_ms=summarize(kdf_t),
        enc_ms=summarize(enc_t),
        dec_ms=summarize(dec_t),
    )

def versions_blob() -> Dict[str, Any]:
    blob: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for mod in ("oqs", "nacl", "ascon"):
        try:
            m = importlib.import_module(mod)
            blob[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            blob[f"{mod}_version"] = None
    return blob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--payload-bytes", type=int, default=4096)
    ap.add_argument("--aad", type=str, default="")  # optional associated data (utf-8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="results.json")
    args = ap.parse_args()

    # Reproducibility notes:
    # - We seed Python's RNG for any non-crypto randomness we might add later.
    # - Crypto randomness (os.urandom) is intentionally not seeded.
    random.seed(args.seed)

    ascon_adapter = AsconAdapter()

    payload = bytes([0x42]) * args.payload_bytes
    aad = args.aad.encode("utf-8")

    # Fixed 20-byte PSK for Model D (deterministic from seed).
    # (Requirement: fixed PSK; not "secure", but reproducible.)
    psk = hashlib.sha256(f"psk|{args.seed}".encode()).digest()[:20]

    models: List[HybridModel] = [
        ModelA_Kyber512(ascon_adapter),
        ModelB_BIKE_L1(ascon_adapter),
        ModelC_X25519(ascon_adapter),
        ModelD_PSK_Ascon80pq(ascon_adapter, psk_20_bytes=psk),
    ]

    results: Dict[str, Any] = {
        "meta": {
            "iterations": args.iterations,
            "payload_bytes": args.payload_bytes,
            "aad_bytes": len(aad),
            "seed": args.seed,
            "notes": [
                "Models A-C: KEM/DH -> HKDF-SHA256 -> 16-byte key -> Ascon-128a",
                "Model D: fixed 20-byte PSK -> Ascon-80pq (no KEM, no HKDF)",
                "Ascon nonce generated via os.urandom(16) per encryption",
                "Timing via time.perf_counter(); values in milliseconds",
            ],
            "versions": versions_blob(),
        },
        "models": [],
    }

    for m in models:
        r = bench_model(m, payload=payload, aad=aad, iterations=args.iterations)
        results["models"].append(asdict(r))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()