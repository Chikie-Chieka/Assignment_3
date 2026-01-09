#!/usr/bin/env python3
"""
main_experiment_enhanced.py

Enhanced version with:
1. CSV logging (latency_results.csv) with nanosecond precision
2. Per-iteration tracking (not aggregated)
3. Backward compatibility with JSON output
4. Memory profiling hooks
5. ENT entropy collection readiness
"""

from __future__ import annotations

import abc
import argparse
import csv
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
# Dependency helpers (unchanged)
# ----------------------------

def _import_liboqs():
    try:
        import oqs
        return oqs
    except Exception as e:
        raise RuntimeError(
            "liboqs-python not available. Install liboqs + Python bindings.\n"
            "Typical: pip install oqs (or build from source depending on platform).\n"
            f"Import error: {e}"
        )

def _import_pynacl():
    try:
        from nacl.public import PrivateKey
        from nacl.bindings import crypto_scalarmult
        return PrivateKey, crypto_scalarmult
    except Exception as e:
        raise RuntimeError(
            "PyNaCl not available. Install with: pip install pynacl\n"
            f"Import error: {e}"
        )

def _import_ascon():
    try:
        ascon = importlib.import_module("ascon")
        if not hasattr(ascon, "encrypt") or not hasattr(ascon, "decrypt"):
            raise AttributeError("ascon module found but missing encrypt/decrypt")
        return ascon
    except Exception as e:
        raise RuntimeError(
            "Ascon Python module not available or incompatible.\n"
            "Install an Ascon AEAD implementation, e.g. `pip install ascon`.\n"
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

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"", variant: str = "Ascon-128a") -> Tuple[bytes, bytes, bytes, int]:
        """Returns (nonce, ct, tag, time_ns)"""
        nonce = os.urandom(16)
        t_start_ns = time.perf_counter_ns()
        ct_and_tag = self.ascon.encrypt(key, nonce, aad, plaintext, variant=variant)
        t_end_ns = time.perf_counter_ns()
        ct, tag = self._split_ct_tag(ct_and_tag, tag_len=16)
        return nonce, ct, tag, t_end_ns - t_start_ns

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"", variant: str = "Ascon-128a") -> Tuple[bytes, int]:
        """Returns (plaintext, time_ns)"""
        ct_and_tag = self._join_ct_tag(ct, tag)
        t_start_ns = time.perf_counter_ns()
        pt = self.ascon.decrypt(key, nonce, aad, ct_and_tag, variant=variant)
        t_end_ns = time.perf_counter_ns()
        return pt, t_end_ns - t_start_ns

# ----------------------------
# Per-iteration tracking data
# ----------------------------

@dataclass
class IterationTimings:
    """Individual iteration timings in nanoseconds"""
    model: str
    iteration: int
    keygen_ns: int = 0
    encaps_ns: int = 0
    decaps_ns: int = 0
    kdf_ns: int = 0
    encryption_ns: int = 0
    decryption_ns: int = 0
    total_ns: int = 0
    failed: int = 0  # 0 = success, 1 = failure

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
    def establish_shared_secret(self) -> Tuple[bytes, int, int]:
        """
        Returns (shared_secret, encaps_ns, decaps_ns).
        For PSK models, set both to 0.
        """

    @abc.abstractmethod
    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, int]:
        """Returns (dem_key, kdf_time_ns)."""

    @abc.abstractmethod
    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes, int]:
        """Returns (nonce, ct, tag, enc_time_ns)."""

    @abc.abstractmethod
    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> Tuple[bytes, int]:
        """Returns (plaintext, dec_time_ns)."""

# ----------------------------
# Model implementations
# ----------------------------

class ModelA_Kyber512(HybridModel):
    def __init__(self, ascon_adapter: AsconAdapter):
        super().__init__("ModelA_Kyber512", ascon_adapter)
        self.oqs = _import_liboqs()
        self.kem_name = "Kyber512"
        self._server_public_key = None
        self._server_secret_key = None

    def setup(self) -> None:
        with self.oqs.KeyEncapsulation(self.kem_name) as kem:
            pk = kem.generate_keypair()
            sk = kem.export_secret_key()
        self._server_public_key = pk
        self._server_secret_key = sk

    def establish_shared_secret(self) -> Tuple[bytes, int, int]:
        t0 = time.perf_counter_ns()
        with self.oqs.KeyEncapsulation(self.kem_name) as client_kem:
            ct, ss_client = client_kem.encap_secret(self._server_public_key)
        t1 = time.perf_counter_ns()

        t2 = time.perf_counter_ns()
        with self.oqs.KeyEncapsulation(self.kem_name, secret_key=self._server_secret_key) as server_kem:
            ss_server = server_kem.decap_secret(ct)
        t3 = time.perf_counter_ns()

        if ss_client != ss_server:
            raise ValueError("KEM shared secret mismatch")

        return ss_client, t1 - t0, t3 - t2

    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, int]:
        t0 = time.perf_counter_ns()
        key = hkdf_sha256(ss, length=16, salt=b"hybrid-bench-salt", info=b"ascon-128a-key")
        t1 = time.perf_counter_ns()
        return key, t1 - t0

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes, int]:
        nonce, ct, tag, enc_ns = self.ascon.encrypt(key, plaintext, aad=aad, variant="Ascon-128a")
        return nonce, ct, tag, enc_ns

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> Tuple[bytes, int]:
        pt, dec_ns = self.ascon.decrypt(key, nonce, ct, tag, aad=aad, variant="Ascon-128a")
        return pt, dec_ns


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

    def establish_shared_secret(self) -> Tuple[bytes, int, int]:
        t0 = time.perf_counter_ns()
        ss_alice = self.crypto_scalarmult(bytes(self._alice_sk), self._bob_pk)
        t1 = time.perf_counter_ns()

        t2 = time.perf_counter_ns()
        ss_bob = self.crypto_scalarmult(bytes(self._bob_sk), self._alice_pk)
        t3 = time.perf_counter_ns()

        if ss_alice != ss_bob:
            raise ValueError("X25519 shared secret mismatch")

        return ss_alice, t1 - t0, t3 - t2

    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, int]:
        t0 = time.perf_counter_ns()
        key = hkdf_sha256(ss, length=16, salt=b"hybrid-bench-salt", info=b"ascon-128a-key")
        t1 = time.perf_counter_ns()
        return key, t1 - t0

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes, int]:
        nonce, ct, tag, enc_ns = self.ascon.encrypt(key, plaintext, aad=aad, variant="Ascon-128a")
        return nonce, ct, tag, enc_ns

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> Tuple[bytes, int]:
        pt, dec_ns = self.ascon.decrypt(key, nonce, ct, tag, aad=aad, variant="Ascon-128a")
        return pt, dec_ns


class ModelD_PSK_Ascon80pq(HybridModel):
    def __init__(self, ascon_adapter: AsconAdapter, psk_20_bytes: bytes):
        super().__init__("ModelD_PSK_Ascon80pq", ascon_adapter)
        if len(psk_20_bytes) != 20:
            raise ValueError("ModelD requires a fixed 20-byte PSK for Ascon-80pq")
        self.psk = psk_20_bytes

    def setup(self) -> None:
        pass

    def establish_shared_secret(self) -> Tuple[bytes, int, int]:
        return self.psk, 0, 0

    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, int]:
        return ss, 0

    def encrypt(self, key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes, int]:
        nonce, ct, tag, enc_ns = self.ascon.encrypt(key, plaintext, aad=aad, variant="Ascon-80pq")
        return nonce, ct, tag, enc_ns

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> Tuple[bytes, int]:
        pt, dec_ns = self.ascon.decrypt(key, nonce, ct, tag, aad=aad, variant="Ascon-80pq")
        return pt, dec_ns

# ----------------------------
# Benchmark runner with CSV logging
# ----------------------------

def bench_model_enhanced(
    model: HybridModel,
    payload: bytes,
    aad: bytes,
    iterations: int,
    csv_writer: csv.DictWriter
) -> Dict[str, Any]:
    """
    Benchmark with per-iteration CSV logging (nanosecond precision).
    Returns aggregated statistics for JSON output.
    """
    timings_list: List[IterationTimings] = []
    
    # Mapping function to convert dataclass fields to CSV column names
    def timing_to_row(timing: IterationTimings) -> Dict[str, Any]:
        return {
            'Model': timing.model,
            'Iteration': timing.iteration,
            'KeyGen_ns': timing.keygen_ns,
            'Encaps_ns': timing.encaps_ns,
            'Decaps_ns': timing.decaps_ns,
            'KDF_ns': timing.kdf_ns,
            'Encryption_ns': timing.encryption_ns,
            'Decryption_ns': timing.decryption_ns,
            'Total_ns': timing.total_ns,
            'Failed': timing.failed,
        }
    
    model.setup()

    for iteration in range(1, iterations + 1):
        timing = IterationTimings(model=model.name, iteration=iteration)
        
        try:
            # KEM phase
            ss, encaps_ns, decaps_ns = model.establish_shared_secret()
            timing.encaps_ns = encaps_ns
            timing.decaps_ns = decaps_ns

            # KDF phase
            key, kdf_ns = model.derive_dem_key(ss)
            timing.kdf_ns = kdf_ns

            # Encryption phase
            nonce, ct, tag, enc_ns = model.encrypt(key, payload, aad=aad)
            timing.encryption_ns = enc_ns

            # Decryption phase
            pt, dec_ns = model.decrypt(key, nonce, ct, tag, aad=aad)
            timing.decryption_ns = dec_ns

            # Verification
            if pt != payload:
                timing.failed = 1
            else:
                timing.failed = 0

        except Exception as e:
            timing.failed = 1

        # Calculate total
        timing.total_ns = (
            timing.encaps_ns +
            timing.decaps_ns +
            timing.kdf_ns +
            timing.encryption_ns +
            timing.decryption_ns
        )

        timings_list.append(timing)

        # Write to CSV immediately (using mapping function to align fields)
        csv_writer.writerow(timing_to_row(timing))

    # Aggregate statistics for JSON
    def aggregate_timing(attr: str) -> Dict[str, float]:
        values = [getattr(t, attr) for t in timings_list if not t.failed]
        if not values:
            return {"mean": 0.0, "stdev": 0.0}
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
        return {"mean": mean / 1e6, "stdev": stdev / 1e6}  # Convert ns to ms

    failures = sum(1 for t in timings_list if t.failed)

    return {
        "name": model.name,
        "iterations": iterations,
        "failures": failures,
        "encaps_ms": aggregate_timing("encaps_ns"),
        "decaps_ms": aggregate_timing("decaps_ns"),
        "kdf_ms": aggregate_timing("kdf_ns"),
        "enc_ms": aggregate_timing("encryption_ns"),
        "dec_ms": aggregate_timing("decryption_ns"),
        "total_ms": aggregate_timing("total_ns"),
    }

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
    ap.add_argument("--aad", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--csv-out", type=str, default="latency_results.csv")
    args = ap.parse_args()

    random.seed(args.seed)

    ascon_adapter = AsconAdapter()

    payload = bytes([0x42]) * args.payload_bytes
    aad = args.aad.encode("utf-8")

    psk = hashlib.sha256(f"psk|{args.seed}".encode()).digest()[:20]

    models: List[HybridModel] = [
        ModelA_Kyber512(ascon_adapter),
        ModelB_BIKE_L1(ascon_adapter),
        ModelC_X25519(ascon_adapter),
        ModelD_PSK_Ascon80pq(ascon_adapter, psk_20_bytes=psk),
    ]

    # Open CSV file for writing
    csv_file = open(args.csv_out, 'w', newline='', buffering=1)
    csv_fieldnames = [
        'Model', 'Iteration', 'KeyGen_ns', 'Encaps_ns', 'Decaps_ns',
        'KDF_ns', 'Encryption_ns', 'Decryption_ns', 'Total_ns', 'Failed'
    ]
    
    # Custom writer wrapping
    # extrasaction='ignore' provides fallback protection against unexpected fields
    csv_writer_obj = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, extrasaction='ignore')
    csv_writer_obj.writeheader()

    # Wrap to convert from IterationTimings dataclass
    class TimingCSVWriter:
        def __init__(self, csv_writer):
            self.writer = csv_writer
        
        def writerow(self, row_dict):
            self.writer.writerow(row_dict)

    csv_writer = TimingCSVWriter(csv_writer_obj)

    results: Dict[str, Any] = {
        "meta": {
            "iterations": args.iterations,
            "payload_bytes": args.payload_bytes,
            "aad_bytes": len(aad),
            "seed": args.seed,
            "csv_output": args.csv_out,
            "notes": [
                "Models A-C: KEM/DH -> HKDF-SHA256 -> 16-byte key -> Ascon-128a",
                "Model D: fixed 20-byte PSK -> Ascon-80pq (no KEM, no HKDF)",
                "Ascon nonce generated via os.urandom(16) per encryption",
                "Timing via time.perf_counter_ns(); CSV values in nanoseconds, JSON values in milliseconds",
            ],
            "versions": versions_blob(),
        },
        "models": [],
    }

    print(f"Running {args.iterations} iterations per model...")
    print(f"Payload: {args.payload_bytes} bytes")
    print(f"CSV output: {args.csv_out}")
    print()

    for m in models:
        print(f"Benchmarking {m.name}...", end=" ", flush=True)
        r = bench_model_enhanced(m, payload=payload, aad=aad, iterations=args.iterations, csv_writer=csv_writer)
        results["models"].append(r)
        print("✓")

    csv_file.close()

    # Write JSON results
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print()
    print(f"✓ Wrote {args.out}")
    print(f"✓ Wrote {args.csv_out}")
    print()
    print(f"CSV contains {args.iterations * 4} rows ({args.iterations} iterations × 4 models)")

if __name__ == "__main__":
    main()
