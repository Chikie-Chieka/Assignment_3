# PQC Cryptography Benchmarking Suite

Benchmarking framework for post-quantum cryptography (Kyber512, BIKE-L1) versus classical (X25519) and pre-shared-key baselines. Measures KEM → HKDF → AEAD pipeline latency with nanosecond precision.

## Quick Start

```bash
# 1. Clone and setup (one-time)
git clone <repo-url>
cd Assignment_3
python3 -m venv pqc_lab
source pqc_lab/bin/activate  # or: pqc_lab\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 2. Run benchmark
python main_experiment.py --iterations=100

# 3. View results
cat results.json | python -m json.tool
```

**For full setup instructions, troubleshooting, and analysis:** → **[MASTER_GUIDE.md](./MASTER_GUIDE.md)**

## Overview

| Model | Type | Speed | Quantum-Safe |
|-------|------|-------|--------------|
| **ModelA** (Kyber512) | PQC (Lattice) | 1.61 ms | ✅ Yes |
| **ModelB** (BIKE-L1) | PQC (Code) | 2.27 ms | ✅ Yes |
| **ModelC** (X25519) | Classical (ECDH) | 1.70 ms | ❌ No |
| **ModelD** (PSK) | Pre-Shared Key | 1.91 ms | ✅ Yes |

**Pipeline:** Key Agreement (KEM/ECDH) → HKDF-SHA256 → Ascon AEAD

**Timing Precision:** Nanoseconds (via `time.perf_counter_ns()`)

## Entry Points

### `main_experiment.py` — JSON Only
Best for: Quick benchmarks, comparative analysis

```bash
python main_experiment.py \
  --iterations=100 \
  --payload-bytes=256 \
  --seed=1337 \
  --out=results.json
```

**Output:** `results.json` with aggregated stats (mean, stdev per phase)

### `main_experiment_enhanced.py` — JSON + CSV
Best for: Detailed analysis, warm-up studies, statistical validation

```bash
python main_experiment_enhanced.py \
  --iterations=100 \
  --payload-bytes=256 \
  --csv-out=latency_results.csv \
  --out=results.json
```

**Output:** 
- `results.json` (aggregated stats)
- `latency_results.csv` (400 rows: 100 iterations × 4 models)

## Common Commands

```bash
# Standard benchmark (100 iterations)
python main_experiment.py

# High-precision run (500 iterations for publication)
python main_experiment.py --iterations=500

# Custom payload (8 KB instead of 4 KB)
python main_experiment.py --iterations=100 --payload-bytes=8192

# With additional authenticated data
python main_experiment.py --aad="optional-context"

# Reproducible results (same seed)
python main_experiment.py --iterations=100 --seed=42

# CSV output for analysis
python main_experiment_enhanced.py --iterations=100

# For fast testing
python main_experiment.py --iterations=10
```

## Documentation

- **[MASTER_GUIDE.md](./MASTER_GUIDE.md)** ← **START HERE**
  - Installation (Ubuntu/WSL2/macOS)
  - Dependency troubleshooting (oqs/liboqs, Ascon API issues)
  - JSON/CSV metrics explained
  - Iterations guidance (when to use 10 vs 100 vs 500)
  - Warm-up discard methods
  - Plotting templates (4 complete examples with code)
  - .gitignore template

- **[docs/EXPERIMENTAL_PIPELINE_SUMMARY.md](./docs/EXPERIMENTAL_PIPELINE_SUMMARY.md)** (Optional)
  - CLI parameters deep-dive
  - Per-iteration phase breakdown
  - Aggregation methodology
  - Model architecture differences

## Results Format

### JSON (`results.json`)

```json
{
  "meta": {
    "iterations": 100,
    "payload_bytes": 256,
    "seed": 1337
  },
  "models": [
    {
      "name": "ModelA_Kyber512",
      "failures": 0,
      "encaps_ms": { "mean": 0.046, "stdev": 0.012 },
      "decaps_ms": { "mean": 0.023, "stdev": 0.006 },
      "kdf_ms": { "mean": 0.019, "stdev": 0.003 },
      "enc_ms": { "mean": 0.758, "stdev": 0.045 },
      "dec_ms": { "mean": 0.768, "stdev": 0.048 },
      "total_ms": { "mean": 1.614, "stdev": 0.067 }
    }
  ]
}
```

### CSV (`latency_results.csv`)

```csv
Model,Iteration,Encaps_ns,Decaps_ns,KDF_ns,Encryption_ns,Decryption_ns,Total_ns,Failed
ModelA_Kyber512,1,90600,39900,71300,800800,739500,1742100,0
ModelA_Kyber512,2,63500,29600,20900,778799,810300,1703099,0
...
```

**All timing in nanoseconds; easily converted to milliseconds (÷ 1,000,000) for analysis.**

## Files

### Core
- `main_experiment.py` — JSON benchmark harness
- `main_experiment_enhanced.py` — JSON + CSV benchmark harness

### Documentation
- `README.md` — This file (quick start)
- `MASTER_GUIDE.md` — Complete reference (25 KB, all details)
- `docs/EXPERIMENTAL_PIPELINE_SUMMARY.md` — Technical deep-dive

### Configuration
- `requirements.txt` — Pinned Python dependencies
- `.gitignore` — Git exclusions (venv, outputs, etc.)

## System Requirements

- **Python:** 3.8+
- **OS:** Linux, WSL2, or macOS
- **RAM:** ≥2 GB
- **Build tools:** gcc, make (for liboqs compilation)

## Troubleshooting

**Installation issues?** See [MASTER_GUIDE.md § Dependency Troubleshooting](./MASTER_GUIDE.md#dependency-troubleshooting)

**Examples:**
- `ModuleNotFoundError: No module named 'oqs'` → [Solution](./MASTER_GUIDE.md#issue-1-modulenotfounderror-no-module-named-oqs)
- Ascon API mismatch → [Solution](./MASTER_GUIDE.md#issue-4-ascon-api-mismatch)
- liboqs.so.0 not found → [Solution](./MASTER_GUIDE.md#issue-2-importerror-liboqsso0-cannot-open-shared-object-file)

**All questions answered in [MASTER_GUIDE.md](./MASTER_GUIDE.md).**

```bash
python -m json.tool results.json | head -100
```

---

## Installation Details

See **INSTALL.md** for:
- Detailed platform-specific instructions (Linux, macOS, Windows, WSL)
- Troubleshooting common dependency issues
- Verification steps
- Known workarounds

TL;DR:
```bash
pip install liboqs-python PyNaCl ascon
```

---

## Dependencies

| Package | Version | Purpose | Note |
|---------|---------|---------|------|
| `liboqs-python` | ≥0.14.1 | Kyber512, BIKE-L1 KEMs | Required |
| `PyNaCl` | ≥1.5.0 | X25519 key exchange | Required |
| `ascon` | ≥0.0.9 | Ascon-128a, Ascon-80pq AEAD | Required |
| `python` | ≥3.9 | Language | Must match wheel builds |

---

## Usage

### Basic Benchmark

```bash
python main_experiment.py --iterations=100 --payload-bytes=4096
```

### Custom Parameters

```bash
python main_experiment.py \
  --iterations=100           # Number of encrypt/decrypt cycles per model
  --payload-bytes=4096       # Size of plaintext (bytes)
  --aad="metadata-string"    # Additional authenticated data
  --seed=42                  # RNG seed for reproducibility
  --out=results.json         # Output file
```

### Smoke Test (Quick Verification)

```bash
# 1 iteration, 256-byte payload, ~5-10 seconds
python main_experiment.py --iterations=1 --payload-bytes=256
```

### High-Resolution Benchmark

```bash
# 1000 iterations for more stable statistics, ~20-60 minutes
python main_experiment.py --iterations=1000 --payload-bytes=8192
```

---

## Output Format

### JSON Structure

```json
{
  "meta": {
    "aad_bytes": 0,
    "iterations": 100,
    "payload_bytes": 4096,
    "seed": 1337,
    "versions": {
      "python": "3.11.x ...",
      "platform": "Linux-6.x.x...",
      "oqs_version": "0.14.1",
      "nacl_version": "1.6.2",
      "ascon_version": "0.0.9"
    },
    "notes": [...]
  },
  "models": [
    {
      "name": "ModelA_Kyber512",
      "iterations": 100,
      "failures": 0,
      "encaps_ms": {"mean": 2.1, "stdev": 0.3},
      "decaps_ms": {"mean": 2.4, "stdev": 0.2},
      "kdf_ms": {"mean": 0.1, "stdev": 0.01},
      "enc_ms": {"mean": 0.4, "stdev": 0.05},
      "dec_ms": {"mean": 0.5, "stdev": 0.06}
    },
    ...
  ]
}
```

### Interpreting Results

- **encaps_ms / decaps_ms**: Key exchange time in milliseconds
- **kdf_ms**: Key derivation (HKDF) time
- **enc_ms / dec_ms**: Symmetric encryption/decryption time
- **mean / stdev**: Average timing and standard deviation across iterations
- **failures**: Count of iterations where decryption produced wrong plaintext

**Note:** All times are in milliseconds; lower is better.

---

## Models Explained

### ModelA: Kyber512 (Post-Quantum)

- **KEM:** Kyber512 from liboqs (NIST level-1 equivalent)
- **Shared Secret Size:** 32 bytes
- **KDF:** HKDF-SHA256 → 16-byte key
- **AEAD:** Ascon-128a with 128-bit nonce
- **Typical Timing:** Encaps ~2ms, Decaps ~2.5ms, Enc/Dec ~0.5ms

### ModelB: BIKE-L1 (Post-Quantum)

- **KEM:** BIKE-L1 (lattice-based, code-based alternative)
- **Shared Secret Size:** 32 bytes
- **KDF:** HKDF-SHA256 → 16-byte key
- **AEAD:** Ascon-128a with 128-bit nonce
- **Typical Timing:** Encaps ~3ms, Decaps ~3.5ms, Enc/Dec ~0.5ms
- **Note:** May not be available in all liboqs builds

### ModelC: X25519 (Classical ECC)

- **Key Exchange:** X25519 elliptic curve DH
- **Shared Secret Size:** 32 bytes
- **KDF:** HKDF-SHA256 → 16-byte key
- **AEAD:** Ascon-128a with 128-bit nonce
- **Typical Timing:** Encaps ~0.2ms, Decaps ~0.2ms, Enc/Dec ~0.4ms
- **Use as Baseline:** For comparison with PQC alternatives

### ModelD: Pre-Shared Key (PSK) Only

- **Key Exchange:** None (uses fixed 20-byte PSK)
- **KDF:** None (PSK directly used as key)
- **AEAD:** Ascon-80pq with 128-bit nonce
- **Typical Timing:** Enc/Dec ~0.3ms
- **Use as Upper Bound:** Theoretical maximum performance

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'oqs'
```
→ See **INSTALL.md** → "Issue 1"

```
AttributeError: module 'ascon' has no attribute 'encrypt'
```
→ See **INSTALL.md** → "Issue 2"

### Algorithm Unavailable

```
OQSError: Kyber512 is not available on this system
```
→ See **INSTALL.md** → "Issue 3"

### Build Failures

```
error: build failed
```
→ See **INSTALL.md** → "Issue 4"

---

## Code Quality & Usability

### Strengths
- ✅ Clear abstractions (HybridModel base class)
- ✅ Type hints throughout
- ✅ Comprehensive error messages
- ✅ JSON output for automated analysis
- ✅ Reproducible (seed parameter)

### Issues Found & Fixed

See **USABILITY_ANALYSIS.md** for detailed analysis.

**Critical bugs fixed in `main_experiment_fixed.py`:**
1. HMAC-SHA256 implementation (was using wrong function)
2. Ascon API variance (now supports multiple package APIs)
3. liboqs import path fallbacks (handles different distributions)
4. Missing algorithm validation (gracefully skips unavailable KEMs)

---

## Next Steps

1. **First time?** Read **INSTALL.md** for platform-specific setup
2. **Need details?** See **USABILITY_ANALYSIS.md** for comprehensive evaluation
3. **Want to understand patches?** Check **PATCHES.md** for code changes
4. **Ready to benchmark?** Run the smoke test above, then full benchmark
5. **Analyzing results?** See "Interpreting Results" section above

---

## Implementation Notes

### Security Considerations
- Ascon nonce is randomly generated per encryption (secure)
- Model D PSK is derived from seed (deterministic, not cryptographically secure)
- All randomness uses `os.urandom()` (cryptographically secure)

### Performance Notes
- Timings measured via `time.perf_counter()` (high-resolution)
- Ascon encrypt/decrypt includes nonce generation time
- Each model pre-generates keys in `setup()` phase (not timed)
- Results vary significantly based on:
  - CPU model, temperature, load
  - OS scheduler behavior
  - Wheel vs. source builds (liboqs)
  - Payload size (larger = less overhead)

### Reproducibility
- Results reproducible across runs with same `--seed`
- Cross-platform differences expected (different hardware/OS)
- Use `--iterations=1000+` for statistically significant results

---

**Last Updated:** January 2026  
**Status:** ✅ Fully functional with corrected dependencies and error handling
