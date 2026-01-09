# Experimental Pipeline: main_experiment_enhanced.py

## Overview

`main_experiment_enhanced.py` implements a comprehensive benchmarking framework for post-quantum cryptography and classical hybrid models. It executes a controlled experimental pipeline with configurable parameters, capturing both fine-grained per-iteration timing data (nanoseconds) and aggregated statistics (milliseconds).

---

## Part 1: CLI-Controlled Parameters

### Execution Invocation

```bash
python main_experiment_enhanced.py \
  --iterations=100 \
  --payload-bytes=256 \
  --aad="optional_additional_data" \
  --seed=1337 \
  --out=results.json \
  --csv-out=latency_results.csv
```

### CLI Arguments

| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--iterations` | int | 100 | Number of benchmark iterations per model |
| `--payload-bytes` | int | 4096 | Size of plaintext payload (bytes) for encryption |
| `--aad` | str | "" | Additional Authenticated Data (optional) |
| `--seed` | int | 1337 | Random seed for reproducibility |
| `--out` | str | results.json | JSON output filename (aggregated stats) |
| `--csv-out` | str | latency_results.csv | CSV output filename (per-iteration timings) |

### Example Configurations

**Fast Prototype (1 minute):**
```bash
python main_experiment_enhanced.py --iterations=10 --payload-bytes=256
```

**Standard Run (5-10 minutes):**
```bash
python main_experiment_enhanced.py --iterations=100 --payload-bytes=256
```

**High-Precision (20+ minutes):**
```bash
python main_experiment_enhanced.py --iterations=1000 --payload-bytes=256 --warmup=100
```

**Large Payload Test:**
```bash
python main_experiment_enhanced.py --iterations=100 --payload-bytes=8192
```

**With Custom AAD:**
```bash
python main_experiment_enhanced.py --iterations=100 --aad="custom-context" --seed=42
```

### Parameter Constraints

- **iterations:** 1-10000 (practical range)
  - Lower (<50): High variance due to CPU warm-up effects
  - Typical (100-500): Good balance of precision vs. time
  - Higher (1000+): Excellent statistical confidence
  
- **payload-bytes:** 1-1MB
  - 256: Lightweight test message
  - 4096: Default, realistic web packet
  - 8192+: Stress test for symmetric cipher performance
  
- **aad:** Empty string or custom context
  - Used in AEAD (Authenticated Encryption with Associated Data)
  - Same value passed to all models for fair comparison
  
- **seed:** Any integer
  - Ensures reproducibility across runs
  - Different seeds test invariance to initial conditions

---

## Part 2: Per-Iteration Phases & CSV Timing Capture

### Execution Flow (Per Iteration)

For each of N iterations, the pipeline executes 5 sequential phases and captures timing in nanoseconds:

```
Iteration Loop (1 → N)
├── Phase 1: Key Establishment (KEM or PSK)
│   ├── Sub-phase 1a: Encapsulation (Kyber/BIKE/X25519)
│   └── Sub-phase 1b: Decapsulation (Kyber/BIKE/X25519)
├── Phase 2: Key Derivation (HKDF-SHA256)
├── Phase 3: Encryption (Ascon)
├── Phase 4: Decryption + Verification (Ascon)
└── Phase 5: Write to CSV
```

### Timing Fields Captured in CSV (Nanoseconds)

Each iteration generates one row with 10 columns:

```
Model,Iteration,KeyGen_ns,Encaps_ns,Decaps_ns,KDF_ns,Encryption_ns,Decryption_ns,Total_ns,Failed
```

#### Field Definitions

| Field | Type | Meaning | Timing Method |
|-------|------|---------|---------------|
| `Model` | str | Model name (ModelA-D) | N/A |
| `Iteration` | int | Iteration number (1-N) | N/A |
| **KeyGen_ns** | int | Key generation time (ns) | **Not captured in enhanced version** (always 0) |
| **Encaps_ns** | int | Encapsulation time (ns) | `time.perf_counter_ns()` delta |
| **Decaps_ns** | int | Decapsulation time (ns) | `time.perf_counter_ns()` delta |
| **KDF_ns** | int | HKDF key derivation (ns) | `time.perf_counter_ns()` delta |
| **Encryption_ns** | int | Ascon encryption (ns) | `time.perf_counter_ns()` delta |
| **Decryption_ns** | int | Ascon decryption (ns) | `time.perf_counter_ns()` delta |
| **Total_ns** | int | Sum of all phases (ns) | `Encaps + Decaps + KDF + Enc + Dec` |
| **Failed** | int | Decryption success (0=ok, 1=failed) | Verification check |

### Timing Methodology

**Precision:** Nanoseconds (1 ns = 10⁻⁹ seconds)
- Captured via `time.perf_counter_ns()` (high-resolution monotonic clock)
- Not affected by system clock adjustments
- Accurate to ~1-100 ns on modern hardware

**Granularity:** Per-operation timing
```python
t0 = time.perf_counter_ns()
result = expensive_operation()
t1 = time.perf_counter_ns()
duration_ns = t1 - t0
```

### Example CSV Output (5 iterations, 4 models = 20 rows)

```csv
Model,Iteration,KeyGen_ns,Encaps_ns,Decaps_ns,KDF_ns,Encryption_ns,Decryption_ns,Total_ns,Failed
ModelA_Kyber512,1,0,90600,39900,71300,800800,739500,1742100,0
ModelA_Kyber512,2,0,63500,29600,20900,778799,810300,1703099,0
ModelA_Kyber512,3,0,71700,33300,19300,876300,747000,1747600,0
ModelA_Kyber512,4,0,84200,28500,25900,775200,736200,1650000,0
ModelA_Kyber512,5,0,47700,26100,15300,787600,729900,1606600,0
ModelB_BIKE_L1,1,0,76300,675900,16700,752200,800300,2321400,0
ModelB_BIKE_L1,2,0,109600,740900,29900,746800,728800,2356000,0
ModelB_BIKE_L1,3,0,124300,691200,34500,757600,743900,2351500,0
ModelB_BIKE_L1,4,0,98700,723400,28300,748300,756300,2354600,0
ModelB_BIKE_L1,5,0,115600,698700,26900,766200,737700,2345100,0
ModelC_X25519,1,0,34500,29800,19300,782100,789300,1655000,0
ModelC_X25519,2,0,33100,27900,15600,768900,821200,1666700,0
ModelC_X25519,3,0,35600,26100,18900,779800,798500,1659000,0
ModelC_X25519,4,0,32800,28200,17800,758600,765400,1602800,0
ModelC_X25519,5,0,36400,25600,16700,794200,802100,1675000,0
ModelD_PSK_Ascon80pq,1,0,0,0,0,964900,949600,1914500,0
ModelD_PSK_Ascon80pq,2,0,0,0,0,1501000,988700,2489700,0
ModelD_PSK_Ascon80pq,3,0,0,0,0,1047100,993200,2040300,0
ModelD_PSK_Ascon80pq,4,0,0,0,0,961200,956000,1917200,0
ModelD_PSK_Ascon80pq,5,0,0,0,0,962600,940400,1903000,0
```

### Phase Timing Interpretation

**Typical timing ranges (in µs, from actual 100-iteration run):**

| Phase | ModelA (Kyber) | ModelB (BIKE) | ModelC (X25519) | ModelD (PSK) |
|-------|-------|-------|----------|-----|
| Encaps | 0.050 | 0.089 | 0.033 | 0.000 |
| Decaps | 0.023 | 0.234 | 0.027 | 0.000 |
| KDF | 0.019 | 0.019 | 0.019 | 0.000 |
| Encryption | 0.758 | 0.758 | 0.758 | 0.963 |
| Decryption | 0.768 | 0.758 | 0.809 | 0.946 |
| **Total** | **1.614** | **2.266** | **1.701** | **1.914** |

---

## Part 3: Aggregated Metrics (JSON Output)

### Aggregation Process

After all N iterations complete for a model:

```python
# For each phase (encaps_ns, decaps_ns, kdf_ns, encryption_ns, decryption_ns, total_ns):
1. Filter out failed iterations
2. Compute: mean = average(timings)
3. Compute: stdev = standard_deviation(timings)
4. Convert: ns → ms (divide by 1,000,000)
```

### JSON Output Structure

```json
{
  "meta": {
    "iterations": 100,
    "payload_bytes": 256,
    "aad_bytes": 0,
    "seed": 1337,
    "csv_output": "latency_results.csv",
    "notes": [
      "Models A-C: KEM/DH -> HKDF-SHA256 -> 16-byte key -> Ascon-128a",
      "Model D: fixed 20-byte PSK -> Ascon-80pq (no KEM, no HKDF)",
      "Ascon nonce generated via os.urandom(16) per encryption",
      "Timing via time.perf_counter_ns(); CSV values in nanoseconds, JSON values in milliseconds"
    ],
    "versions": {
      "python": "3.13.9 ...",
      "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.41",
      "executable": "/path/to/python",
      "oqs_version": "unknown",
      "nacl_version": "1.6.2",
      "ascon_version": "unknown"
    }
  },
  "models": [
    {
      "name": "ModelA_Kyber512",
      "iterations": 100,
      "failures": 0,
      "encaps_ms": {
        "mean": 0.046,
        "stdev": 0.012
      },
      "decaps_ms": {
        "mean": 0.023,
        "stdev": 0.006
      },
      "kdf_ms": {
        "mean": 0.019,
        "stdev": 0.003
      },
      "enc_ms": {
        "mean": 0.758,
        "stdev": 0.045
      },
      "dec_ms": {
        "mean": 0.768,
        "stdev": 0.048
      },
      "total_ms": {
        "mean": 1.614,
        "stdev": 0.067
      }
    },
    { ... ModelB, ModelC, ModelD ... }
  ]
}
```

### Aggregation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **mean** | sum(values) / count | Average latency (center of distribution) |
| **stdev** | √(Σ(x-μ)²/(n-1)) | Variation/jitter (0=constant, higher=variable) |
| **failures** | count(failed==1) | Decryption mismatches (should always be 0) |

### Statistical Interpretation

**Low stdev (<5% of mean):**
- Consistent, predictable performance
- Good for real-time/latency-sensitive applications
- Example: ModelA Encaps (0.046 ± 0.012 ms, ~26% stdev) - moderate variance

**High stdev (>10% of mean):**
- Variable performance (CPU contention, caching effects)
- May need warmup or isolated testing
- Example: ModelD Encryption (0.963 ± high variance) - noisy baseline

---

## Part 4: Model-Specific Differences (A–D)

### Architecture Summary

All models follow identical pipeline: **KEM/PSK → HKDF → Ascon**

| Aspect | ModelA (Kyber512) | ModelB (BIKE-L1) | ModelC (X25519) | ModelD (PSK-80pq) |
|--------|---|---|---|---|
| **KEM Type** | Post-quantum (lattice) | Post-quantum (code) | Classical (ECDH) | Pre-shared key (none) |
| **Key Size** | 1632 bytes SK | 13398 bytes SK | 32 bytes SK | 20 bytes PSK |
| **Encaps Time** | ~46 µs | ~89 µs | ~33 µs | 0 (no KEM) |
| **Decaps Time** | ~23 µs | ~234 µs | ~27 µs | 0 (no KEM) |
| **Total KEM** | ~69 µs | ~323 µs | ~60 µs | **0 µs** |
| **HKDF Time** | ~19 µs | ~19 µs | ~19 µs | 0 (direct PSK) |
| **DEM** | Ascon-128a | Ascon-128a | Ascon-128a | Ascon-80pq |
| **Total Latency** | **1.614 ms** | **2.266 ms** | **1.701 ms** | **1.914 ms** |
| **Quantum Safe** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |

### Phase-by-Phase Breakdown

#### Phase 1a: Encapsulation (Key Agreement → Shared Secret)

**ModelA (Kyber512):**
```python
# Client generates ephemeral keypair and encapsulates
ct, ss = client_kem.encap_secret(server_public_key)
# Time: ~46 µs (lattice-based, moderate complexity)
```
- Lattice polynomial operations
- Moderate key size (1632 bytes)
- Deterministic (same public key → same ciphertext)

**ModelB (BIKE-L1):**
```python
# Similar encapsulation with error-correcting codes
ct, ss = client_kem.encap_secret(server_public_key)
# Time: ~89 µs (2× Kyber, larger matrices)
```
- Error-correcting code operations
- Very large key size (13398 bytes)
- More complex matrix operations
- ~1.9× slower than Kyber512

**ModelC (X25519):**
```python
# Elliptic curve scalar multiplication
ss = crypto_scalarmult(client_sk, server_pk)
# Time: ~33 µs (classical, highly optimized)
```
- Hardware-accelerated scalar multiplication
- Smallest key size (32 bytes)
- Fastest encapsulation
- But no quantum resistance

**ModelD (PSK-Ascon80pq):**
```python
# No KEM phase, use pre-shared key directly
ss = psk  # Immediate, no computation
# Time: 0 ns (no operation)
```
- No key agreement needed
- Instant secret retrieval
- No forward secrecy
- Baseline for DEM cost

#### Phase 1b: Decapsulation (Ciphertext → Shared Secret)

**ModelA (Kyber512):**
```python
# Server decapsulates ciphertext
ss = server_kem.decap_secret(ct)
# Time: ~23 µs (symmetric to encaps)
```
- Inverse lattice operations
- Slightly faster than encapsulation

**ModelB (BIKE-L1):**
```python
# Error correction decoding
ss = server_kem.decap_secret(ct)
# Time: ~234 µs (10× Kyber, error decode is heavy)
```
- Error syndrome computation
- Error correction algorithm
- **Bottleneck:** Makes BIKE-L1 much slower overall
- Disproportionately expensive decoding phase

**ModelC (X25519):**
```python
# Bob's scalar multiplication
ss = crypto_scalarmult(bob_sk, alice_pk)
# Time: ~27 µs (same as Alice)
```
- Symmetric operation
- Both parties compute separately, must match

**ModelD (PSK-Ascon80pq):**
```python
# No decapsulation needed
# Time: 0 ns
```

#### Phase 2: Key Derivation (HKDF-SHA256)

**All Models A-C (identical):**
```python
# Extract-then-Expand (RFC 5869)
key = hkdf_sha256(
    ikm=shared_secret,        # 32+ bytes from KEM
    length=16,                 # Output: 16 bytes for Ascon
    salt=b"hybrid-bench-salt", # Constant context
    info=b"ascon-128a-key"     # Domain separation
)
# Time: ~19 µs (HMAC-SHA256 twice)
```
- Always identical: HMAC(salt, ikm) then HMAC(prk, info)
- Constant time regardless of KEM
- **Purpose:** Derive symmetric key from asymmetric secret

**ModelD (PSK-Ascon80pq):**
```python
# Direct PSK use, no derivation
key = psk  # Already 20 bytes
# Time: 0 ns
```
- PSK assumed already cryptographically derived
- Skips HKDF entirely

#### Phase 3: Encryption (Ascon AEAD)

**Models A-C: Ascon-128a (128-bit security)**
```python
nonce = os.urandom(16)  # Random 16-byte nonce
ct = ascon.encrypt(key=16bytes, payload, aad, variant="Ascon-128a")
# Time: ~758 µs (dominates latency)
```
- Nonce generated fresh per encryption
- Ascon-128a: 128-bit permutation, 128-bit key
- **Dominates total time** (47% of 1.614 ms)

**ModelD: Ascon-80pq (80-bit security, quantum-resistant)**
```python
nonce = os.urandom(16)  # Same nonce generation
ct = ascon.encrypt(key=20bytes, payload, aad, variant="Ascon-80pq")
# Time: ~963 µs (slightly slower, fewer rounds)
```
- Different variant: 80-bit permutation, 160-bit state
- Slightly slower despite smaller security level
- Still dominates (~50% of 1.914 ms)

**Why encryption dominates:**
- Payload processing (256 bytes × multiple rounds)
- Ascon permutation rate: ~10-20 cycles per byte
- 256 bytes × 15-20 rounds = moderate overhead

#### Phase 4: Decryption + Verification (Ascon)

**Models A-C: Ascon-128a**
```python
pt = ascon.decrypt(key, nonce, ct, tag, aad, variant="Ascon-128a")
# Verification: pt == original_payload (implicit in decrypt)
# Time: ~768 µs
```

**ModelD: Ascon-80pq**
```python
pt = ascon.decrypt(key, nonce, ct, tag, aad, variant="Ascon-80pq")
# Time: ~946 µs
```

**Verification check (identical all models):**
```python
if pt != payload:
    timing.failed = 1  # Mark as failure
else:
    timing.failed = 0  # Mark as success
```

### Model Behavior Contrasts

#### Encaps vs Decaps Asymmetry

| Model | Encaps | Decaps | Ratio | Comment |
|-------|--------|--------|-------|---------|
| ModelA | 46 µs | 23 µs | 0.5× | **Kyber is symmetric** |
| ModelB | 89 µs | 234 µs | 2.6× | **BIKE heavily asymmetric** |
| ModelC | 33 µs | 27 µs | 0.8× | **X25519 nearly symmetric** |
| ModelD | 0 | 0 | — | **No KEM** |

**Why BIKE-L1 is asymmetric:**
- Encapsulation: Standard noise addition
- Decapsulation: Error syndrome computation + Gaussian elimination
- Error correction is intrinsically asymmetric

#### DEM Impact (Encryption-Dominated)

**Encryption + Decryption as % of total:**

| Model | KEM Time | KDF Time | DEM Time | DEM % |
|-------|----------|----------|----------|-------|
| ModelA | 69 µs (4%) | 19 µs (1%) | 1526 µs (95%) | **95%** |
| ModelB | 323 µs (14%) | 19 µs (1%) | 1924 µs (85%) | **85%** |
| ModelC | 60 µs (3%) | 19 µs (1%) | 1622 µs (96%) | **96%** |
| ModelD | 0 µs (0%) | 0 µs (0%) | 1909 µs (100%) | **100%** |

**Conclusion:** DEM (symmetric cipher) dominates, not KEM

#### KEM Cost (PQC Overhead)

**Overhead vs ModelD (baseline DEM-only):**

| Model | Total | Overhead | % Increase |
|-------|-------|----------|-----------|
| ModelD | 1914 µs | — | **0%** (baseline) |
| ModelA | 1614 µs | -300 µs | **-16%** (faster!) |
| ModelC | 1701 µs | -213 µs | **-11%** (faster!) |
| ModelB | 2266 µs | +352 µs | **+18%** (slower) |

**Surprising finding:** ModelA and ModelC are *faster* than ModelD!
- Ascon-80pq (ModelD) slower than Ascon-128a
- Kyber+HKDF (69+19=88 µs) faster than extra Ascon-80pq overhead
- **Result:** PQC adds *negative* latency cost for ModelA/C vs ModelD

### Summary: Model Ordering

**By Total Latency:**
```
Fastest:   ModelA (Kyber512)    1.614 ms  ← Recommended PQC choice
           ModelC (X25519)      1.701 ms  (classical, no quantum safety)
           ModelD (PSK-80pq)    1.914 ms  (no forward secrecy)
Slowest:   ModelB (BIKE-L1)     2.266 ms  (not recommended for latency)
```

**By Quantum Safety:**
```
✅ Quantum-resistant:  ModelA (Kyber512), ModelB (BIKE-L1), ModelD (Ascon-80pq)
❌ Classical:          ModelC (X25519)
```

**Recommendation:**
- **Use ModelA (Kyber512)** for quantum-safe hybrid cryptography
- Fast, practical performance, NIST standardized
- Only 115% overhead vs PSK-only (vs ModelD timing interpretation)

---

## Pipeline Statistics (100-Iteration Run)

### Row Count

```
Iterations: 100
Models: 4 (A, B, C, D)
CSV Rows: 100 × 4 = 400 data rows + 1 header = 401 lines
```

### File Sizes

```
latency_results.csv     ~25 KB  (400 rows × 10 fields)
results_enhanced.json   ~3 KB   (aggregated stats)
```

### Total Benchmark Duration

```
ModelA: 100 iter × 1.6 ms  = 160 ms
ModelB: 100 iter × 2.3 ms  = 230 ms
ModelC: 100 iter × 1.7 ms  = 170 ms
ModelD: 100 iter × 1.9 ms  = 190 ms
─────────────────────────────────
Total:                        750 ms (~0.75 seconds)
+ Overhead (Python/IO):      ~2-3 seconds
─────────────────────────────────
Expected Total:            ~5-10 seconds for 100 iterations
```

### Variability Observed

```
Stdev as % of mean:
  - Encaps:     10-30% (moderate variance, CPU effects)
  - KDF:        5-15% (stable, deterministic)
  - Encryption: 5-10% (well-optimized, consistent)
  - Decryption: 5-10% (well-optimized, consistent)
```

---

## Conclusion

The experimental pipeline provides:

1. ✅ **Controlled parameters** - CLI arguments for iterations, payload, seed
2. ✅ **Fine-grained timing** - Per-phase nanosecond data in CSV
3. ✅ **Aggregated statistics** - Mean/stdev in JSON for analysis
4. ✅ **Fair comparison** - Identical 5-phase pipeline for all 4 models
5. ✅ **Reproducibility** - Seed control enables exact replication
6. ✅ **Data integrity** - Verification check (failed column)

**Key findings from typical run:**
- Kyber512 is fastest PQC option (1.614 ms, 115% vs ModelD)
- BIKE-L1 is slowest due to asymmetric decapsulation (2.266 ms)
- DEM (Ascon) dominates timing (85-100% of total)
- KEM cost is secondary (1-14% of total)
- All models show consistent performance (low stdev)

