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
import json
import os
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# Import randomness tests
try:
    from randomness_tests import shannon_entropy_bits_per_byte, monobit_zscore, chi_square_pvalue
    HAS_RANDOMNESS_TESTS = True
except ImportError:
    HAS_RANDOMNESS_TESTS = False

import math

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
        import ascon
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
        t_start_ns = time.process_time_ns()
        ct_and_tag = self.ascon.encrypt(key, nonce, aad, plaintext, variant=variant)
        t_end_ns = time.process_time_ns()
        ct, tag = self._split_ct_tag(ct_and_tag, tag_len=16)
        return nonce, ct, tag, t_end_ns - t_start_ns

    def decrypt(self, key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"", variant: str = "Ascon-128a") -> Tuple[bytes, int]:
        """Returns (plaintext, time_ns)"""
        ct_and_tag = self._join_ct_tag(ct, tag)
        t_start_ns = time.process_time_ns()
        pt = self.ascon.decrypt(key, nonce, aad, ct_and_tag, variant=variant)
        t_end_ns = time.process_time_ns()
        return pt, t_end_ns - t_start_ns

# ----------------------------
# Per-iteration tracking data
# ----------------------------

@dataclass
class IterationTimings:
    """Individual iteration timings in nanoseconds, with memory and randomness metrics"""
    model: str
    iteration: int
    keygen_ns: int = 0  # Setup time (one-time, replicated to each iteration)
    encaps_ns: int = 0
    decaps_ns: int = 0
    kdf_ns: int = 0
    encryption_ns: int = 0
    decryption_ns: int = 0
    total_ns: int = 0
    total_s: float = 0.0  # Total time in seconds
    failed: int = 0  # 0 = success, 1 = failure
    
    # Memory profiling
    peak_rss_kb: float = float('nan')  # Peak RSS during iteration (KB)
    peak_alloc_kb: float = float('nan')  # Peak allocated during iteration (KB)
    
    # Randomness metrics: Shared Secret
    entropy_sharedsecret: float = float('nan')
    monobit_sharedsecret: float = float('nan')
    chisquare_sharedsecret: float = float('nan')
    
    # Randomness metrics: HKDF output
    entropy_hkdf: float = float('nan')
    monobit_hkdf: float = float('nan')
    chisquare_hkdf: float = float('nan')

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
        t0 = time.process_time_ns()
        with self.oqs.KeyEncapsulation(self.kem_name) as client_kem:
            ct, ss_client = client_kem.encap_secret(self._server_public_key)
        t1 = time.process_time_ns()

        t2 = time.process_time_ns()
        with self.oqs.KeyEncapsulation(self.kem_name, secret_key=self._server_secret_key) as server_kem:
            ss_server = server_kem.decap_secret(ct)
        t3 = time.process_time_ns()

        if ss_client != ss_server:
            raise ValueError("KEM shared secret mismatch")

        return ss_client, t1 - t0, t3 - t2

    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, int]:
        t0 = time.process_time_ns()
        key = hkdf_sha256(ss, length=16, salt=b"hybrid-bench-salt", info=b"ascon-128a-key")
        t1 = time.process_time_ns()
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
        t0 = time.process_time_ns()
        ss_alice = self.crypto_scalarmult(bytes(self._alice_sk), self._bob_pk)
        t1 = time.process_time_ns()

        t2 = time.process_time_ns()
        ss_bob = self.crypto_scalarmult(bytes(self._bob_sk), self._alice_pk)
        t3 = time.process_time_ns()

        if ss_alice != ss_bob:
            raise ValueError("X25519 shared secret mismatch")

        return ss_alice, t1 - t0, t3 - t2

    def derive_dem_key(self, ss: bytes) -> Tuple[bytes, int]:
        t0 = time.process_time_ns()
        key = hkdf_sha256(ss, length=16, salt=b"hybrid-bench-salt", info=b"ascon-128a-key")
        t1 = time.process_time_ns()
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
# Static Bandwidth Cost Definitions (Metric 4)
# ----------------------------

BANDWIDTH_COSTS = {
    "ModelA_Kyber512": {
        "pk_bytes": 800,
        "ct_bytes": 768,
        "tag_bytes": 0,  # Embedded in Ascon AEAD ciphertext
        "total_bytes": 1568,
        "description": "Kyber-512 PK (800B) + CT (768B) for KEM + Ascon-128a AEAD",
    },
    "ModelB_BIKE_L1": {
        "pk_bytes": 1282,
        "ct_bytes": 1282,
        "tag_bytes": 0,
        "total_bytes": 2564,
        "description": "BIKE-L1 PK (1282B) + CT (1282B) for KEM + Ascon-128a AEAD",
    },
    "ModelC_X25519": {
        "pk_bytes": 32,
        "ct_bytes": 32,
        "tag_bytes": 0,
        "total_bytes": 64,
        "description": "X25519 ECDH PK (32B) + SS (32B) for KEM + Ascon-128a AEAD",
    },
    "ModelD_PSK_Ascon80pq": {
        "pk_bytes": 0,
        "ct_bytes": 0,
        "tag_bytes": 16,
        "total_bytes": 16,
        "description": "Ascon-80pq symmetric: no KEM overhead, 16B tag only",
    },
}

def compute_bandwidth_summary() -> Dict[str, Any]:
    """
    Return static bandwidth costs for all models (Metric 4).
    """
    return {
        "description": "Static cryptographic communication overhead per model (Metric 4)",
        "unit": "bytes",
        "formula": "PK_size + CT_size + Tag_size",
        "models": BANDWIDTH_COSTS,
        "comparison_notes": [
            f"Kyber-512: {BANDWIDTH_COSTS['ModelA_Kyber512']['total_bytes']} bytes (PQC)",
            f"X25519: {BANDWIDTH_COSTS['ModelC_X25519']['total_bytes']} bytes (classical ECC)",
            f"Ratio: {BANDWIDTH_COSTS['ModelA_Kyber512']['total_bytes'] / BANDWIDTH_COSTS['ModelC_X25519']['total_bytes']:.1f}× larger",
            f"Ascon-80pq: {BANDWIDTH_COSTS['ModelD_PSK_Ascon80pq']['total_bytes']} bytes (symmetric baseline, no KEM)",
        ],
    }

# ----------------------------
# ENT Entropy Test (Metric 3)
# ----------------------------

def run_ent_entropy_test(model: HybridModel, iterations: int = 50) -> List[Dict[str, Any]]:
    """
    Run ENT entropy test multiple times per model.
    
    For each iteration:
    1. Generate 1MB of encrypted zeros (reduced from 10MB for speed)
    2. Run ent -t via pipe to get serial correlation coefficient
    
    Args:
        model: HybridModel instance to test
        iterations: Number of iterations to run (default 50)
    
    Returns:
        List of dicts, one per iteration:
        {
            "model": str,
            "iteration": int,
            "entropy_bytes": int,
            "serial_correlation_coefficient": float (or None if ent not available),
            "status": str,
        }
    """
    import subprocess
    
    results = []
    
    for iteration_num in range(1, iterations + 1):
        try:
            # Generate 1 MB of zeros (reduced from 10MB for 10x speedup in I/O)
            PAYLOAD_SIZE_MB = 1
            payload = bytes([0x00]) * (PAYLOAD_SIZE_MB * 1024 * 1024)
            
            # Setup model for single encryption
            model.setup()
            
            # Encrypt the payload
            key = os.urandom(16)
            nonce, ct, tag, _ = model.encrypt(key, payload)
            
            # Run ent -t via pipe (stdin) to avoid disk I/O - FIX #4
            try:
                result = subprocess.run(
                    ["ent", "-t"],  # Read from stdin instead of file
                    input=ct,  # Binary ciphertext passed directly
                    capture_output=True,
                    timeout=60,
                )
                
                if result.returncode != 0:
                    results.append({
                        "model": model.name,
                        "iteration": iteration_num,
                        "entropy_bytes": len(ct),
                        "serial_correlation_coefficient": None,
                        "status": "ent tool returned error",
                    })
                else:
                    # FIX #1: Parse ENT -t CSV output correctly
                    # ent -t outputs CSV: Header,Col1,Col2,...,Serial-Correlation
                    # Data row with actual serial correlation value
                    serial_corr = None
                    try:
                        output_text = result.stdout.decode('utf-8') if isinstance(result.stdout, bytes) else result.stdout
                        lines = output_text.strip().split('\n')
                        if len(lines) >= 2:
                            header = lines[0].split(',')
                            data = lines[1].split(',')
                            
                            # Find "Serial-Correlation" column index
                            if "Serial-Correlation" in header:
                                idx = header.index("Serial-Correlation")
                                if idx < len(data):
                                    serial_corr = float(data[idx].strip())
                    except (IndexError, ValueError, AttributeError):
                        serial_corr = None
                    
                    results.append({
                        "model": model.name,
                        "iteration": iteration_num,
                        "entropy_bytes": len(ct),
                        "serial_correlation_coefficient": serial_corr,
                        "status": "success" if serial_corr is not None else "ent ran but parse failed",
                    })
            
            except FileNotFoundError:
                results.append({
                    "model": model.name,
                    "iteration": iteration_num,
                    "serial_correlation_coefficient": None,
                    "status": "ent tool not found",
                })
            except subprocess.TimeoutExpired:
                results.append({
                    "model": model.name,
                    "iteration": iteration_num,
                    "serial_correlation_coefficient": None,
                    "status": "ent tool timeout",
                })
        
        except Exception as e:
            results.append({
                "model": model.name,
                "iteration": iteration_num,
                "serial_correlation_coefficient": None,
                "status": "ent test failed",
            })
    
    return results

# ----------------------------
# Benchmark runner with CSV logging
# ----------------------------

def bench_model_enhanced(
    model: HybridModel,
    payload: bytes,
    aad: bytes,
    iterations: int,
    csv_writer: csv.DictWriter,
) -> Dict[str, Any]:
    """
    Benchmark with per-iteration CSV logging (nanosecond precision).
    
    WARMUP LOGIC:
    - Runs iterations + 100 total (10,100 if user requests 10,000).
    - First 100 iterations are warmup and are NOT written to CSV.
    - Remaining iterations are valid and written to CSV.
    
    MEMORY MEASUREMENT:
    - tracemalloc is reset and measured ONLY around KEM handshake (establish_shared_secret).
    - NOT around entire iteration window.
    
    TIMING:
    - Uses time.process_time_ns() for CPU-only latency (all phases).
    
    Returns aggregated statistics for JSON output (from valid iterations only).
    """
    WARMUP_COUNT = 100
    actual_iterations = iterations + WARMUP_COUNT  # Run 10,100 if user wants 10,000 valid
    
    timings_list: List[IterationTimings] = []
    valid_timings_list: List[IterationTimings] = []
    
    # Mapping function to convert dataclass fields to CSV column names
    def timing_to_row(timing: IterationTimings) -> Dict[str, Any]:
        row = {
            'Model': timing.model,
            'Iteration': timing.iteration,
            'KeyGen_ns': timing.keygen_ns,
            'Encaps_ns': timing.encaps_ns,
            'Decaps_ns': timing.decaps_ns,
            'KDF_ns': timing.kdf_ns,
            'Encryption_ns': timing.encryption_ns,
            'Decryption_ns': timing.decryption_ns,
            'Total_ns': timing.total_ns,
            'Total_s': timing.total_s,
            'Failed': timing.failed,
            'Peak_Alloc_KB': timing.peak_alloc_kb if not (isinstance(timing.peak_alloc_kb, float) and math.isnan(timing.peak_alloc_kb)) else '',
        }
        return row
    
    # Measure model setup (KeyGen) once per model
    keygen_t0 = time.process_time_ns()
    model.setup()
    keygen_t1 = time.process_time_ns()
    keygen_ns = keygen_t1 - keygen_t0

    for iteration in range(1, actual_iterations + 1):
        is_warmup = (iteration <= WARMUP_COUNT)
        timing = IterationTimings(model=model.name, iteration=iteration, keygen_ns=keygen_ns)
        
        try:
            # === KEM Handshake Memory Window (tracemalloc ONLY around KEM) ===
            tracemalloc.reset_peak()
            
            # KEM phase (encaps + decaps)
            ss, encaps_ns, decaps_ns = model.establish_shared_secret()
            timing.encaps_ns = encaps_ns
            timing.decaps_ns = decaps_ns
            
            # Capture peak allocated ONLY during KEM phase
            try:
                _, peak_bytes = tracemalloc.get_traced_memory()
                timing.peak_alloc_kb = peak_bytes / 1024.0
            except:
                pass
            
            # === Rest of iteration (outside memory measurement window) ===
            
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

        # Calculate total and convert to seconds
        timing.total_ns = (
            timing.encaps_ns +
            timing.decaps_ns +
            timing.kdf_ns +
            timing.encryption_ns +
            timing.decryption_ns
        )
        timing.total_s = timing.total_ns / 1_000_000_000

        timings_list.append(timing)
        
        # Write to CSV only if NOT in warmup phase
        if not is_warmup:
            csv_writer.writerow(timing_to_row(timing))
            valid_timings_list.append(timing)
        
        # Progress indicator
        if iteration % 100 == 0:
            status = "warmup" if is_warmup else "valid"
            print(f"  ... iteration {iteration} ({status})", file=sys.stderr)

    # Aggregate statistics for JSON (from valid iterations only, post-warmup)
    def aggregate_timing(attr: str) -> Dict[str, float]:
        values = [getattr(t, attr) for t in valid_timings_list if not t.failed]
        if not values:
            return {"mean": 0.0, "stdev": 0.0}
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
        return {"mean": mean / 1e6, "stdev": stdev / 1e6}  # Convert ns to ms

    failures = sum(1 for t in valid_timings_list if t.failed)

    return {
        "name": model.name,
        "iterations": iterations,  # Report VALID count (post-warmup)
        "total_runs": actual_iterations,  # Total including warmup
        "failures": failures,
        "keygen_ms": aggregate_timing("keygen_ns"),
        "encaps_ms": aggregate_timing("encaps_ns"),
        "decaps_ms": aggregate_timing("decaps_ns"),
        "kdf_ms": aggregate_timing("kdf_ns"),
        "enc_ms": aggregate_timing("encryption_ns"),
        "dec_ms": aggregate_timing("decryption_ns"),
        "total_ms": aggregate_timing("total_ns"),
    }

def generate_summary_csv(input_csv: str, output_csv: str, ent_csv: str = None) -> None:
    """
    Read testing_process.csv and ENT_Test.csv, generate testing_summary.csv with aggregate stats.
    
    Args:
        input_csv: Path to testing_process.csv
        output_csv: Path to testing_summary.csv (output)
        ent_csv: Path to ENT_Test.csv (optional)
    """
    import csv as csv_module
    
    # Read input CSV (latency metrics)
    data_by_model = {}
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            model = row['Model']
            if model not in data_by_model:
                data_by_model[model] = []
            data_by_model[model].append(row)
    
    # Read ENT CSV if available (entropy metrics)
    ent_data_by_model = {}
    if ent_csv and os.path.exists(ent_csv):
        with open(ent_csv, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                model = row['Model']
                if model not in ent_data_by_model:
                    ent_data_by_model[model] = []
                ent_data_by_model[model].append(row)
    
    # Aggregate per model
    numeric_cols = [
        'KeyGen_ns', 'Encaps_ns', 'Decaps_ns', 'KDF_ns',
        'Encryption_ns', 'Decryption_ns', 'Total_ns', 'Peak_Alloc_KB'
    ]
    
    ent_cols = ['Serial_Correlation_Coefficient']
    
    summary_rows = []
    for model in ['ModelA_Kyber512', 'ModelB_BIKE_L1', 'ModelC_X25519', 'ModelD_PSK_Ascon80pq']:
        summary_row = {'Model': model}
        
        # Process latency metrics from testing_process.csv
        if model in data_by_model:
            for col in numeric_cols:
                values = []
                for row in data_by_model[model]:
                    try:
                        val = float(row[col])
                        values.append(val)
                    except (ValueError, KeyError):
                        pass
                
                if values:
                    summary_row[f'{col}_Mean'] = statistics.fmean(values)
                    summary_row[f'{col}_StDev'] = statistics.stdev(values) if len(values) > 1 else 0.0
                    summary_row[f'{col}_Min'] = min(values)
                    summary_row[f'{col}_Max'] = max(values)
                    summary_row[f'{col}_Median'] = statistics.median(values)
                else:
                    summary_row[f'{col}_Mean'] = 0.0
                    summary_row[f'{col}_StDev'] = 0.0
                    summary_row[f'{col}_Min'] = 0.0
                    summary_row[f'{col}_Max'] = 0.0
                    summary_row[f'{col}_Median'] = 0.0
        else:
            for col in numeric_cols:
                summary_row[f'{col}_Mean'] = 0.0
                summary_row[f'{col}_StDev'] = 0.0
                summary_row[f'{col}_Min'] = 0.0
                summary_row[f'{col}_Max'] = 0.0
                summary_row[f'{col}_Median'] = 0.0
        
        # Process ENT metrics from ENT_Test.csv
        if model in ent_data_by_model:
            for col in ent_cols:
                values = []
                for row in ent_data_by_model[model]:
                    try:
                        val = float(row[col])
                        values.append(val)
                    except (ValueError, KeyError):
                        pass
                
                if values:
                    summary_row[f'{col}_Mean'] = statistics.fmean(values)
                    summary_row[f'{col}_StDev'] = statistics.stdev(values) if len(values) > 1 else 0.0
                    summary_row[f'{col}_Min'] = min(values)
                    summary_row[f'{col}_Max'] = max(values)
                    summary_row[f'{col}_Median'] = statistics.median(values)
                else:
                    summary_row[f'{col}_Mean'] = 0.0
                    summary_row[f'{col}_StDev'] = 0.0
                    summary_row[f'{col}_Min'] = 0.0
                    summary_row[f'{col}_Max'] = 0.0
                    summary_row[f'{col}_Median'] = 0.0
        else:
            for col in ent_cols:
                summary_row[f'{col}_Mean'] = 0.0
                summary_row[f'{col}_StDev'] = 0.0
                summary_row[f'{col}_Min'] = 0.0
                summary_row[f'{col}_Max'] = 0.0
                summary_row[f'{col}_Median'] = 0.0
        
        summary_rows.append(summary_row)
    
    # Write summary CSV
    all_fieldnames = ['Model']
    for col in numeric_cols:
        for stat in ['Mean', 'StDev', 'Min', 'Max', 'Median']:
            all_fieldnames.append(f'{col}_{stat}')
    for col in ent_cols:
        for stat in ['Mean', 'StDev', 'Min', 'Max', 'Median']:
            all_fieldnames.append(f'{col}_{stat}')
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv_module.DictWriter(f, fieldnames=all_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--payload-bytes", type=int, default=4096)
    ap.add_argument("--aad", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--csv-out", type=str, default="testing_process.csv")
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

    # Open CSV file for writing (testing_process.csv with per-iteration detail)
    csv_file = open(args.csv_out, 'w', newline='', buffering=1)
    csv_fieldnames = [
        'Model', 'Iteration', 'KeyGen_ns', 'Encaps_ns', 'Decaps_ns',
        'KDF_ns', 'Encryption_ns', 'Decryption_ns', 'Total_ns', 'Total_s', 'Failed',
        'Peak_Alloc_KB',
    ]
    
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
                "Metric 1 (Latency): time.process_time_ns(), 10,100 iterations with first 100 discarded as warmup",
                "Metric 2 (Memory): tracemalloc peak during KEM handshake only (not RSS)",
                "Metric 3 (Entropy): ENT subprocess test (if available)",
                "Metric 4 (Bandwidth): Static cryptographic overhead per model",
                "Models A-C: KEM/DH -> HKDF-SHA256 -> 16-byte key -> Ascon-128a",
                "Model D: fixed 20-byte PSK -> Ascon-80pq (no KEM, encaps/decaps = 0 ns)",
                "Ascon nonce generated via os.urandom(16) per encryption",
            ],
            "versions": {"python": sys.version.replace("\n", " ")},
        },
        "models": [],
        "bandwidth": compute_bandwidth_summary(),  # Metric 4
    }

    print(f"Running {args.iterations} valid iterations per model (+ 100 warmup)...")
    print(f"Payload: {args.payload_bytes} bytes")
    print(f"CSV output: {args.csv_out}")
    print()

    for m in models:
        print(f"Benchmarking {m.name}...", end=" ", flush=True)
        r = bench_model_enhanced(m, payload=payload, aad=aad, iterations=args.iterations, csv_writer=csv_writer)
        results["models"].append(r)
        print(" ✓")

    csv_file.close()

    # Run ENT entropy tests (Metric 3) - 50 iterations per model
    # FIX #3: Parallelize model testing for 4x speedup
    ent_csv_path = "ENT_Test.csv"
    print("\nRunning ENT randomness tests (50 iterations, 1MB per iteration, parallel processing)...")
    ent_iterations = 50
    ent_file = open(ent_csv_path, 'w', newline='', buffering=1)
    ent_fieldnames = ['Model', 'Iteration', 'Entropy_Bytes', 'Serial_Correlation_Coefficient', 'Status']
    ent_writer_obj = csv.DictWriter(ent_file, fieldnames=ent_fieldnames, extrasaction='ignore')
    ent_writer_obj.writeheader()
    
    ent_all_results = {}
    
    # FIX #3: Use ThreadPoolExecutor to run models in parallel
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    csv_lock = threading.Lock()
    
    def run_ent_and_log(m):
        """Run ENT tests for a single model."""
        ent_results = run_ent_entropy_test(m, iterations=ent_iterations)
        
        success_count = 0
        with csv_lock:
            for ent_result in ent_results:
                ent_writer_obj.writerow({
                    'Model': ent_result.get('model'),
                    'Iteration': ent_result.get('iteration'),
                    'Entropy_Bytes': ent_result.get('entropy_bytes', ''),
                    'Serial_Correlation_Coefficient': ent_result.get('serial_correlation_coefficient', ''),
                    'Status': ent_result.get('status'),
                })
                if ent_result.get('status') == 'success' and ent_result.get('serial_correlation_coefficient') is not None:
                    success_count += 1
        
        return m.name, ent_results, success_count
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_ent_and_log, m): m for m in models}
        
        for future in futures:
            try:
                model_name, ent_results, success_count = future.result()
                ent_all_results[model_name] = ent_results
                print(f"  {model_name}...", end=" ", flush=True)
                print(f" ✓ ({success_count}/{ent_iterations} successful)")
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
    
    ent_file.close()
    print(f"✓ Wrote {ent_csv_path}")
    
    # Generate summary CSV: testing_summary.csv with one row per model with means
    summary_csv_path = args.csv_out.replace('.csv', '_summary.csv')
    generate_summary_csv(args.csv_out, summary_csv_path, ent_csv_path)
    print(f"✓ Wrote {summary_csv_path}")
    
    results["ent_entropy"] = ent_all_results  # Metric 3
    
    # Write JSON results
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print()
    print(f"✓ Wrote {args.out}")
    print(f"✓ Wrote {args.csv_out}")
    print(f"✓ Wrote {summary_csv_path}")
    print()
    print(f"Summary: {args.iterations} valid iterations × 4 models = {args.iterations * 4} rows in {args.csv_out}")
    print(f"(Plus 100 warmup iterations per model, discarded from CSV)")


if __name__ == "__main__":
    main()
